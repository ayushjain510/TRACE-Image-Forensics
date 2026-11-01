"""Central configuration for the Image Tampering Detection project."""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODEL_PATH = os.path.join(MODEL_DIR, "ela_srm_cnn.pt")

# Error Level Analysis
ELA_QUALITY = 90      # JPEG quality used for the re-save step
ELA_SCALE = 15        # brightness multiplier when visualising ELA
ELA_NORM = 16.0       # ELA values are divided by this and clipped to 0..1 for the CNN

# Model input: images are resized to IMG_SIZE x IMG_SIZE (no further downsampling,
# so small local edits are not averaged away)
IMG_SIZE = 256

# Training
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42

# Folder names accepted as class labels (CASIA v2 uses Au / Tp)
AUTHENTIC_DIRS = {"au", "authentic", "real", "original"}
TAMPERED_DIRS = {"tp", "tampered", "fake", "forged"}

CLASS_NAMES = ["Authentic", "Tampered"]
VALID_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
