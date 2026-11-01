"""Train the ELA-CNN tampering classifier.

Usage:
    python src/train.py --data data/CASIA2 --epochs 15
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from torch.utils.data import DataLoader

import config
from dataset import ELADataset, build_features, scan_dataset, split_indices
from model import ELACNN, count_parameters


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            count += x.size(0)
    return total_loss / count, correct / count


@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        p = torch.softmax(model(x.to(device)), dim=1)[:, 1]
        probs.append(p.cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(probs)


def save_plots(history, y_true, y_prob, out_dir):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(epochs, history["train_loss"], label="Train")
    ax[0].plot(epochs, history["val_loss"], label="Validation")
    ax[0].set_title("Loss"); ax[0].set_xlabel("Epoch"); ax[0].legend()
    ax[1].plot(epochs, history["train_acc"], label="Train")
    ax[1].plot(epochs, history["val_acc"], label="Validation")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("Epoch"); ax[1].legend()
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)
    plt.close(fig)

    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(config.CLASS_NAMES); ax.set_yticklabels(config.CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (Test set)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, y_prob):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (Test set)"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join(config.DATA_DIR, "CASIA2"))
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    args = parser.parse_args()

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    print(f"Scanning {args.data} ...")
    paths, labels = scan_dataset(args.data)
    print(f"  {len(paths)} images | authentic={labels.count(0)} "
          f"tampered={labels.count(1)}")

    cache = os.path.join(config.MODEL_DIR,
                         f"ela_cache_{os.path.basename(os.path.normpath(args.data))}.npy")
    print("Computing ELA features ...")
    feats = build_features(paths, cache)

    tr, va, te = split_indices(labels)
    print(f"  split: train={len(tr)} val={len(va)} test={len(te)}")
    train_dl = DataLoader(ELADataset(feats, labels, tr, augment=True),
                          batch_size=args.batch, shuffle=True)
    val_dl = DataLoader(ELADataset(feats, labels, va), batch_size=args.batch)
    test_dl = DataLoader(ELADataset(feats, labels, te), batch_size=args.batch)

    model = ELACNN().to(device)
    print(f"Model parameters: {count_parameters(model):,}  device={device}")

    # class weights handle the authentic/tampered imbalance in CASIA v2
    counts = np.bincount(np.asarray(labels)[tr], minlength=2)
    weights = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val, patience_left = float("inf"), 5
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        tl, ta = run_epoch(model, train_dl, criterion, device, optimizer)
        vl, vacc = run_epoch(model, val_dl, criterion, device)
        scheduler.step(vl)
        for k, v in zip(history, (tl, ta, vl, vacc)):
            history[k].append(v)
        print(f"Epoch {epoch:2d}/{args.epochs} | train loss {tl:.4f} acc {ta:.4f}"
              f" | val loss {vl:.4f} acc {vacc:.4f}")
        if vl < best_val:
            best_val, patience_left = vl, 5
            torch.save({"model_state": model.state_dict(),
                        "epoch": epoch}, config.MODEL_PATH)
        else:
            patience_left -= 1
            if patience_left == 0:
                print("Early stopping.")
                break
    train_time = time.time() - start

    model.load_state_dict(torch.load(config.MODEL_PATH,
                                     map_location=device)["model_state"])
    y_true, y_prob = predict_loader(model, test_dl, device)
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "dataset": args.data,
        "num_images": len(paths),
        "train/val/test": [len(tr), len(va), len(te)],
        "model": "ELA-SRM CNN",
        "parameters": count_parameters(model),
        "lr": args.lr,
        "epochs_run": len(history["train_loss"]),
        "train_time_sec": round(train_time, 1),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "history": history,
    }
    print("\nTest results")
    print(classification_report(y_true, y_pred, target_names=config.CLASS_NAMES,
                                digits=4))
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")

    with open(os.path.join(config.RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    save_plots(history, y_true, y_prob, config.RESULTS_DIR)
    print(f"Saved model to {config.MODEL_PATH} and plots to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
