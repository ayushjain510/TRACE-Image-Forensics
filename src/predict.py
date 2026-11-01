"""Run the full analysis on one image (used by the CLI, the web app and
evaluate_system.py).

Classification branch: the image is split into overlapping 256 x 256 tiles at
original resolution; the CNN scores every tile and the image score is the
highest tile score (one suspicious region is enough).
Localisation branch: SIFT copy-move detection.
Final verdict: Tampered if the CNN score >= threshold OR copy-move is detected.

Usage:
    python src/predict.py path/to/image.jpg [--save]
"""
import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image

import config
from copy_move import detect_copy_move
from ela import ela_visual, tile_inputs
from model import load_model

THRESHOLD = 0.5
_model = None


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(config.MODEL_PATH):
            raise FileNotFoundError(
                "No trained model found. Run: python src/train.py --data <dataset>")
        _model = load_model(config.MODEL_PATH)
    return _model


@torch.no_grad()
def tile_scores(pil_img, batch=16):
    tiles, pos, size = tile_inputs(pil_img)
    model = get_model()
    probs = []
    for s in range(0, len(tiles), batch):
        logits = model(torch.from_numpy(tiles[s:s + batch]))
        probs.append(torch.softmax(logits, dim=1)[:, 1].numpy())
    return np.concatenate(probs), pos, size


def suspicion_map(probs, pos, size, tile=config.IMG_SIZE):
    """Average the tile probabilities over each pixel -> (H, W) map 0..1."""
    w, h = size
    acc = np.zeros((h, w), np.float32)
    cnt = np.zeros((h, w), np.float32)
    for p, (x, y) in zip(probs, pos):
        acc[y:y + tile, x:x + tile] += p
        cnt[y:y + tile, x:x + tile] += 1
    return acc / np.maximum(cnt, 1)


def heatmap_overlay(pil_img, heat):
    """Colour the 0..1 suspicion map (blue = low, red = high) over the image."""
    rgb = np.asarray(pil_img.convert("RGB"))
    heat = cv2.resize(heat, (rgb.shape[1], rgb.shape[0]))
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=max(rgb.shape[:2]) / 40)
    colour = cv2.applyColorMap((np.clip(heat, 0, 1) * 255).astype(np.uint8),
                               cv2.COLORMAP_JET)
    colour = cv2.cvtColor(colour, cv2.COLOR_BGR2RGB)
    blended = (0.6 * rgb + 0.4 * colour).astype(np.uint8)
    return Image.fromarray(blended)


def analyse(path_or_image, with_visuals=True):
    pil = path_or_image if isinstance(path_or_image, Image.Image) \
        else Image.open(path_or_image)
    pil = pil.convert("RGB")

    probs, pos, size = tile_scores(pil)
    prob_tampered = float(probs.max())

    bgr = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    cm = detect_copy_move(bgr)

    cnn_flag = prob_tampered >= THRESHOLD
    tampered = cnn_flag or cm["suspicious"]
    if cnn_flag and cm["suspicious"]:
        reason = "CNN and copy-move detector agree"
    elif cnn_flag:
        reason = "CNN found compression / noise inconsistencies"
    elif cm["suspicious"]:
        reason = "duplicated region found by copy-move detector"
    else:
        reason = "no inconsistencies found"

    result = {
        "verdict": "Tampered" if tampered else "Authentic",
        "reason": reason,
        "prob_tampered": prob_tampered,
        "cnn_flag": cnn_flag,
        "confidence": prob_tampered if tampered else 1 - prob_tampered,
        "num_tiles": len(probs),
        "copy_move_suspicious": cm["suspicious"],
        "copy_move_pairs": len(cm["pairs"]),
    }
    if with_visuals:
        heat = suspicion_map(probs, pos, size)
        result.update({
            "ela_image": ela_visual(pil),
            "heatmap_image": heatmap_overlay(pil, heat),
            "copy_move_image": Image.fromarray(
                cv2.cvtColor(cm["annotated"], cv2.COLOR_BGR2RGB)),
        })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--save", action="store_true",
                        help="save ELA / heatmap / copy-move images to results/")
    args = parser.parse_args()

    r = analyse(args.image)
    print(f"Verdict      : {r['verdict']}  ({r['reason']})")
    print(f"P(tampered)  : {r['prob_tampered']:.4f}  (max over {r['num_tiles']} tiles)")
    print(f"Copy-move    : {'detected' if r['copy_move_suspicious'] else 'not detected'}"
          f" ({r['copy_move_pairs']} matched pairs)")
    if args.save:
        stem = os.path.splitext(os.path.basename(args.image))[0]
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        for key in ("ela_image", "heatmap_image", "copy_move_image"):
            out = os.path.join(config.RESULTS_DIR, f"{stem}_{key}.png")
            r[key].save(out)
            print(f"saved {out}")


if __name__ == "__main__":
    main()
