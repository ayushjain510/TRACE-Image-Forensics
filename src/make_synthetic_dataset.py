"""Generate a small labelled tampering dataset for quick testing / demos.

It mimics how CASIA v2 forgeries are made:
  * Authentic : a natural image saved once as JPEG.
  * Spliced   : a region from a *different* image (with its own JPEG history)
                pasted in, then the result saved as JPEG.
  * Copy-move : a region of the same image copied to another location.

Source photos come from scikit-image's built-in sample images, so no download
is needed. Use this to check the pipeline end to end; use CASIA v2 for the
real evaluation.

Usage:
    python src/make_synthetic_dataset.py --out data/synthetic --n 600
"""
import argparse
import io
import os
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from skimage import data as skdata

SOURCES = ["astronaut", "coffee", "chelsea", "rocket", "camera", "coins",
           "moon", "page", "text", "brick", "grass", "gravel", "colorwheel",
           "hubble_deep_field", "immunohistochemistry", "retina", "cat",
           "logo", "motorcycle_left", "motorcycle_right", "horse", "clock",
           "checkerboard", "shepp_logan_phantom"]

SIZE = 256


def load_sources():
    imgs = []
    for name in SOURCES:
        try:
            arr = getattr(skdata, name)()
        except Exception:
            continue
        if isinstance(arr, tuple):
            arr = arr[0]
        arr = np.asarray(arr)
        if arr.dtype == bool:
            arr = arr.astype(np.uint8) * 255
        if arr.dtype != np.uint8:
            arr = arr.astype(np.float64)
            arr = (255 * (arr - arr.min()) / (np.ptp(arr) or 1)).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, -1)
        arr = arr[..., :3]
        if min(arr.shape[:2]) >= 160:
            imgs.append(Image.fromarray(arr))
    return imgs


def jpeg(img, q):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def random_view(src):
    """Random crop / scale / flip / colour change so every sample differs."""
    w, h = src.size
    s = random.uniform(0.45, 1.0) * min(w, h)
    x, y = random.uniform(0, w - s), random.uniform(0, h - s)
    img = src.crop((int(x), int(y), int(x + s), int(y + s))).resize(
        (SIZE, SIZE), Image.BICUBIC)
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.75, 1.25))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
    img = ImageEnhance.Color(img).enhance(random.uniform(0.7, 1.3))
    return img


def random_mask(pw, ph):
    m = Image.new("L", (pw, ph), 0)
    from PIL import ImageDraw
    d = ImageDraw.Draw(m)
    if random.random() < 0.5:
        d.ellipse((0, 0, pw - 1, ph - 1), fill=255)
    else:
        d.rectangle((0, 0, pw - 1, ph - 1), fill=255)
    return m.filter(ImageFilter.GaussianBlur(random.uniform(0, 2)))


def make_tampered(sources, base_src):
    base = jpeg(random_view(base_src), random.randint(70, 95))
    pw, ph = random.randint(48, 120), random.randint(48, 120)
    dx, dy = random.randint(0, SIZE - pw), random.randint(0, SIZE - ph)

    if random.random() < 0.6:  # splicing from another image
        other = random.choice([s for s in sources if s is not base_src])
        donor = jpeg(random_view(other), random.randint(60, 100))
        sx, sy = random.randint(0, SIZE - pw), random.randint(0, SIZE - ph)
        patch = donor.crop((sx, sy, sx + pw, sy + ph))
        kind = "splice"
    else:  # copy-move within the same image
        sx, sy = random.randint(0, SIZE - pw), random.randint(0, SIZE - ph)
        patch = base.crop((sx, sy, sx + pw, sy + ph))
        if random.random() < 0.5:
            patch = ImageEnhance.Brightness(patch).enhance(random.uniform(0.9, 1.1))
        kind = "copymove"
    base.paste(patch, (dx, dy), random_mask(pw, ph))
    return base, kind


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/synthetic")
    p.add_argument("--n", type=int, default=600, help="images per class")
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed)

    sources = load_sources()
    print(f"Loaded {len(sources)} source photos")
    au, tp = os.path.join(a.out, "Au"), os.path.join(a.out, "Tp")
    os.makedirs(au, exist_ok=True); os.makedirs(tp, exist_ok=True)

    for i in range(a.n):
        src = random.choice(sources)
        random_view(src).save(os.path.join(au, f"Au_{i:04d}.jpg"), "JPEG",
                              quality=random.randint(70, 95))
        img, kind = make_tampered(sources, src)
        img.save(os.path.join(tp, f"Tp_{kind}_{i:04d}.jpg"), "JPEG",
                 quality=random.randint(70, 95))
    print(f"Wrote {a.n} authentic + {a.n} tampered images to {a.out}")


if __name__ == "__main__":
    main()
