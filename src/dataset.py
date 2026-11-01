"""Dataset loading: finds images under Au/Tp (or authentic/tampered) folders,
computes the 2-channel model input (ELA + greyscale) and caches it as uint8
so training is fast and memory use stays low (about 1.6 GB for CASIA v2)."""
import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from config import (AUTHENTIC_DIRS, TAMPERED_DIRS, VALID_EXT, VAL_SPLIT,
                    TEST_SPLIT, SEED, IMG_SIZE)
from ela import model_input_uint8, to_float


def scan_dataset(root):
    """Return (paths, labels). Label 0 = authentic, 1 = tampered."""
    paths, labels = [], []
    for sub in sorted(os.listdir(root)):
        full = os.path.join(root, sub)
        if not os.path.isdir(full):
            continue
        key = sub.lower()
        if key in AUTHENTIC_DIRS:
            label = 0
        elif key in TAMPERED_DIRS:
            label = 1
        else:
            continue
        for dirpath, _, files in os.walk(full):
            for name in sorted(files):
                if name.lower().endswith(VALID_EXT):
                    paths.append(os.path.join(dirpath, name))
                    labels.append(label)
    if not paths:
        raise FileNotFoundError(
            f"No images found in {root}. Expected sub-folders 'Au' and 'Tp' "
            "(or 'authentic' and 'tampered').")
    return paths, labels


def build_features(paths, cache_file=None):
    """Compute model inputs for every image, optionally caching to .npy."""
    if cache_file and os.path.exists(cache_file):
        data = np.load(cache_file)
        if data.shape[0] == len(paths) and data.shape[-1] == IMG_SIZE:
            return data
    feats = np.zeros((len(paths), 2, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    for i, p in enumerate(paths):
        try:
            feats[i] = model_input_uint8(p)
        except Exception as exc:  # corrupt file -> leave as zeros
            print(f"  skipped {p}: {exc}")
        if (i + 1) % 500 == 0:
            print(f"  features computed for {i + 1}/{len(paths)} images")
    if cache_file:
        np.save(cache_file, feats)
    return feats


def split_indices(labels):
    idx = np.arange(len(labels))
    train_val, test = train_test_split(
        idx, test_size=TEST_SPLIT, stratify=labels, random_state=SEED)
    rel_val = VAL_SPLIT / (1 - TEST_SPLIT)
    train, val = train_test_split(
        train_val, test_size=rel_val,
        stratify=np.asarray(labels)[train_val], random_state=SEED)
    return train, val, test


class ELADataset(Dataset):
    def __init__(self, features, labels, indices, augment=False):
        self.x = features
        self.y = np.asarray(labels)
        self.idx = indices
        self.augment = augment

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        x = to_float(self.x[j])
        if self.augment:
            if np.random.rand() < 0.5:
                x = x[:, :, ::-1]
            if np.random.rand() < 0.5:
                x = x[:, ::-1, :]
        return torch.from_numpy(np.ascontiguousarray(x)), int(self.y[j])
