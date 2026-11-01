# Image Tampering Detection

Detects whether a digital image has been edited (spliced or copy-moved) using
**Error Level Analysis (ELA) + SRM noise residuals + a compact CNN**, and
localises copy-move forgeries with **SIFT keypoint self-matching**. Includes a
Flask web app that shows the verdict, the ELA image, a suspicion heatmap and
copy-move matches.

Group 34 — 3rd Year Industry Track, School of Technology, Woxsen University

## Project structure

```
image-tampering-detection/
├── app.py                      Flask web app
├── requirements.txt
├── src/
│   ├── config.py               paths and hyper-parameters
│   ├── ela.py                  Error Level Analysis + model input
│   ├── model.py                SRM filter layer + ELA-SRM CNN
│   ├── dataset.py              dataset scan, input caching, train/val/test split
│   ├── train.py                training + test evaluation + plots
│   ├── predict.py              single-image analysis (CLI and app)
│   ├── copy_move.py            SIFT-based copy-move localisation
│   ├── make_synthetic_dataset.py  generated dataset for quick tests
│   └── make_figures.py         figures for the report / slides
├── templates/index.html
├── static/style.css
├── models/ela_srm_cnn.pt       trained weights
├── results/                    metrics.json, curves, confusion matrix, ROC, figures
└── sample_images/              demo images for the viva
```

## Setup (Windows)

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset

**CASIA v2.0** (7,491 authentic + 5,123 tampered images). Download from Kaggle
("CASIA 2.0 Image Tampering Detection Dataset") and arrange it as:

```
data/CASIA2/Au/   authentic images
data/CASIA2/Tp/   tampered images
```

To recreate the generated dataset used for the reported results:

```bat
python src\make_synthetic_dataset.py --out data\synthetic --n 1000
```

## Train

```bat


python train.py --data ..\data\synthetic --epochs 15
python train.py --data ..\data\CASIA2 --epochs 15
```

Model inputs are cached in `models/` after the first run (uint8, ~1.6 GB for
CASIA v2), so later runs start quickly. The best model (lowest validation loss)
is saved to `models/ela_srm_cnn.pt`, and test metrics/plots go to `results/`.
A GPU is used automatically if available.

## Predict one image

```bat
python src\predict.py sample_images\tampered_splice.jpg --save
```

## Web app

```bat
python app.py
```

Open http://127.0.0.1:5000 and upload an image.

## How it works

1. **ELA** – resize to 256×256, re-save as JPEG at quality 90 and take the
   absolute difference. Edited regions have a different compression history,
   so they show a different error level. Scaled by a fixed factor (÷16), not
   per image, so the absolute level is kept.
2. **SRM noise residuals** – three fixed high-pass filters applied to the
   greyscale image remove the content and keep the noise pattern; pasted
   regions carry foreign noise.
3. **CNN** – the 4 maps (ELA + 3 SRM) go through 4 conv blocks
   (16→32→64→128 filters, BatchNorm, ReLU, MaxPool), then global **max + average**
   pooling (max catches a small tampered patch), dense 64 + dropout, softmax.
   114,642 parameters.
4. **Copy-move localisation** – SIFT keypoints matched against the same image;
   pairs that pass Lowe's ratio test, are ≥40 px apart, and share a consistent
   shift with ≥4 other pairs are drawn as lines.
5. **Heatmap** – the ELA map is smoothed and overlaid in colour.
