"""Figures for the report / presentation: ELA examples and analysis panels."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import config
from ela import ela_visual
from predict import analyse

SAMPLES = os.path.join(config.BASE_DIR, "sample_images")
OUT = os.path.join(config.RESULTS_DIR, "figures")
os.makedirs(OUT, exist_ok=True)


def ela_examples():
    files = ["authentic_coffee.jpg", "tampered_splice.jpg"]
    titles = ["Authentic", "Spliced (tampered)"]
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))
    for i, (f, t) in enumerate(zip(files, titles)):
        img = Image.open(os.path.join(SAMPLES, f)).convert("RGB")
        ax[i, 0].imshow(img); ax[i, 0].set_title(f"{t} - original")
        ax[i, 1].imshow(ela_visual(img)); ax[i, 1].set_title(f"{t} - ELA")
    for a in ax.ravel():
        a.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ela_examples.png"), dpi=150)
    plt.close(fig)


def analysis_panel(fname, out_name):
    path = os.path.join(SAMPLES, fname)
    r = analyse(path)
    img = Image.open(path).convert("RGB")
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    panels = [(img, "Original"), (r["ela_image"], "ELA"),
              (r["heatmap_image"], "Suspicion heatmap"),
              (r["copy_move_image"], "Copy-move matches")]
    for a, (im, t) in zip(ax, panels):
        a.imshow(im); a.set_title(t); a.axis("off")
    fig.suptitle(f"{fname}:  {r['verdict']}  (P(tampered) = {r['prob_tampered']:.3f}, "
                 f"copy-move pairs = {r['copy_move_pairs']})", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, out_name), dpi=150)
    plt.close(fig)
    return r


def single_images():
    """Copy-move result and a portrait 'hero' image for the title slide."""
    r = analyse(os.path.join(SAMPLES, "tampered_copymove_rocket.jpg"))
    r["copy_move_image"].save(os.path.join(OUT, "cm_rocket.png"))
    s = analyse(os.path.join(SAMPLES, "tampered_splice.jpg"))
    hm = s["heatmap_image"]
    w, h = hm.size
    hm.crop((int(w * 0.45), 0, int(w * 0.95), h)).save(os.path.join(OUT, "hero.png"))


if __name__ == "__main__":
    ela_examples()
    single_images()
    for f in sorted(os.listdir(SAMPLES)):
        if f.lower().endswith(".jpg"):
            r = analysis_panel(f, "panel_" + os.path.splitext(f)[0] + ".png")
            print(f"{f:32s} {r['verdict']:9s} P={r['prob_tampered']:.3f} "
                  f"copy-move={r['copy_move_suspicious']} ({r['copy_move_pairs']})")
