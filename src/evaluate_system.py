"""Evaluate the complete system (CNN + copy-move fusion) on the test split.

train.py reports the CNN alone. This script runs predict.analyse() on every
test image, exactly as the web app does, and reports:
  * CNN only          : Tampered if P(tampered) >= 0.5
  * Copy-move only    : Tampered if the SIFT detector fires
  * Fused (final)     : Tampered if either fires

Usage:
    python src/evaluate_system.py --data data/synthetic
"""
import argparse
import json
import os

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

import config
from dataset import scan_dataset, split_indices
from predict import analyse


def summary(y, pred):
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join(config.DATA_DIR, "CASIA2"))
    args = parser.parse_args()

    paths, labels = scan_dataset(args.data)
    _, _, test = split_indices(labels)
    y = np.asarray(labels)[test]

    cnn, cmv, kinds = [], [], []
    for k, i in enumerate(test):
        r = analyse(paths[i], with_visuals=False)
        cnn.append(int(r["cnn_flag"]))
        cmv.append(int(r["copy_move_suspicious"]))
        kinds.append(os.path.basename(paths[i]))
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(test)} images")
    cnn, cmv = np.array(cnn), np.array(cmv)
    fused = np.maximum(cnn, cmv)

    out = {"cnn_only": summary(y, cnn), "copy_move_only": summary(y, cmv),
           "fused": summary(y, fused)}

    # recall per forgery type (file names from make_synthetic_dataset.py)
    for kind in ("splice", "copymove"):
        mask = np.array([kind in n for n in kinds])
        if mask.any():
            out[f"recall_{kind}"] = {
                "cnn_only": float(cnn[mask].mean()),
                "copy_move_only": float(cmv[mask].mean()),
                "fused": float(fused[mask].mean()),
                "n": int(mask.sum()),
            }

    for name in ("cnn_only", "copy_move_only", "fused"):
        m = out[name]
        print(f"{name:15s} acc {m['accuracy']:.4f}  prec {m['precision']:.4f}  "
              f"rec {m['recall']:.4f}  f1 {m['f1']:.4f}  cm {m['confusion_matrix']}")
    for kind in ("splice", "copymove"):
        if f"recall_{kind}" in out:
            print(f"recall on {kind}: {out[f'recall_{kind}']}")

    with open(os.path.join(config.RESULTS_DIR, "system_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
