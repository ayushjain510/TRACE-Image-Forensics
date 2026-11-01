"""Error Level Analysis (ELA) and model-input preparation.

When a JPEG is saved again at a known quality, regions that were already
compressed at that quality change very little, while regions pasted in from
another source (with a different compression history) change more. The
absolute difference between the image and its re-saved copy therefore
highlights edited regions.

Important: ELA and noise traces are destroyed by resampling, so images are
never shrunk before ELA. Large images are analysed as 256 x 256 tiles at their
original resolution instead.
"""
import io

import numpy as np
from PIL import Image, ImageChops, ImageEnhance

from config import ELA_QUALITY, ELA_SCALE, ELA_NORM, IMG_SIZE

TILE_STRIDE = IMG_SIZE // 2


def compute_ela(image, quality=ELA_QUALITY):
    """Return the raw ELA difference image (PIL, RGB) for a PIL image or path."""
    if isinstance(image, str):
        image = Image.open(image)
    original = image.convert("RGB")

    buffer = io.BytesIO()
    original.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer).convert("RGB")

    return ImageChops.difference(original, resaved)


def ela_visual(image, quality=ELA_QUALITY, scale=ELA_SCALE):
    """ELA image brightened so differences are visible to a person."""
    diff = compute_ela(image, quality)
    extrema = diff.getextrema()
    max_diff = max(e[1] for e in extrema) or 1
    auto_scale = 255.0 / max_diff
    return ImageEnhance.Brightness(diff).enhance(min(auto_scale, scale * 4))


def _ensure_min_size(img, size=IMG_SIZE):
    """Upscale only images smaller than the tile size (never shrink)."""
    w, h = img.size
    if w < size or h < size:
        img = img.resize((max(w, size), max(h, size)), Image.BILINEAR)
    return img


def _channels(img, quality=ELA_QUALITY):
    """Full-resolution uint8 ELA magnitude and greyscale arrays."""
    ela = np.asarray(compute_ela(img, quality), dtype=np.float32).mean(axis=2)
    grey = np.asarray(img.convert("L"), dtype=np.uint8)
    return np.clip(ela, 0, 255).astype(np.uint8), grey


def model_input_uint8(image, quality=ELA_QUALITY, size=IMG_SIZE):
    """Training input: compact (2, size, size) uint8 array.

    channel 0: ELA magnitude (mean over RGB)
    channel 1: greyscale image (the CNN applies SRM noise filters to it)
    Larger images use the central size x size window at original resolution.
    """
    if isinstance(image, str):
        image = Image.open(image)
    img = _ensure_min_size(image.convert("RGB"), size)
    ela, grey = _channels(img, quality)
    h, w = grey.shape
    y, x = (h - size) // 2, (w - size) // 2
    return np.stack([ela[y:y + size, x:x + size], grey[y:y + size, x:x + size]])


def to_float(arr_uint8):
    """Convert the uint8 input into the float array the CNN expects."""
    x = arr_uint8.astype(np.float32)
    x[0] = np.clip(x[0] / ELA_NORM, 0.0, 1.0)
    x[1] = x[1] / 255.0
    return x


def _starts(length, size=IMG_SIZE, stride=TILE_STRIDE):
    starts = list(range(0, length - size + 1, stride))
    if starts[-1] != length - size:
        starts.append(length - size)
    return starts


def tile_inputs(image, quality=ELA_QUALITY, size=IMG_SIZE):
    """Inference input: overlapping size x size tiles at original resolution.

    Returns (tiles float32 (N, 2, size, size), positions [(x, y)], (W, H)).
    """
    if isinstance(image, str):
        image = Image.open(image)
    img = _ensure_min_size(image.convert("RGB"), size)
    ela, grey = _channels(img, quality)
    h, w = grey.shape
    tiles, pos = [], []
    for y in _starts(h, size):
        for x in _starts(w, size):
            tiles.append(to_float(np.stack([ela[y:y + size, x:x + size],
                                            grey[y:y + size, x:x + size]])))
            pos.append((x, y))
    return np.stack(tiles), pos, (w, h)


def ela_tensor_input(image, quality=ELA_QUALITY, size=IMG_SIZE):
    """CNN input for one image (central window): float32 (2, size, size)."""
    return to_float(model_input_uint8(image, quality, size))
