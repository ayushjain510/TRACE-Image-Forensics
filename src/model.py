"""ELA-SRM CNN: classifies an image as Authentic or Tampered.

Input (N, 2, 256, 256):
    channel 0 = Error Level Analysis map (compression inconsistencies)
    channel 1 = greyscale image, turned into 3 noise-residual maps by fixed
                SRM high-pass filters (Rao & Ni, 2016) inside the network

So the network sees 4 maps (ELA + 3 SRM residuals), none of which contain the
image content itself, only traces of editing.

Tampering is usually local, so after the convolution blocks we use global MAX
pooling (strongest local evidence) together with global AVERAGE pooling.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Three classic SRM high-pass kernels
_K1 = np.array([[0, 0, 0, 0, 0],
                [0, -1, 2, -1, 0],
                [0, 2, -4, 2, 0],
                [0, -1, 2, -1, 0],
                [0, 0, 0, 0, 0]], np.float32) / 4
_K2 = np.array([[-1, 2, -2, 2, -1],
                [2, -6, 8, -6, 2],
                [-2, 8, -12, 8, -2],
                [2, -6, 8, -6, 2],
                [-1, 2, -2, 2, -1]], np.float32) / 12
_K3 = np.array([[0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, -2, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]], np.float32) / 2


class SRMFilter(nn.Module):
    """Fixed (non-trainable) SRM noise-residual filters."""

    def __init__(self):
        super().__init__()
        kernels = torch.tensor(np.stack([_K1, _K2, _K3])[:, None])
        self.register_buffer("weight", kernels)

    def forward(self, grey):
        residual = F.conv2d(grey, self.weight, padding=2)
        return torch.clamp(residual * 10, -1, 1)


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class ELASRMCNN(nn.Module):
    def __init__(self, num_classes=2, dropout=0.5):
        super().__init__()
        self.srm = SRMFilter()
        self.features = nn.Sequential(
            conv_block(4, 16),     # 256 -> 128
            conv_block(16, 32),    # 128 -> 64
            conv_block(32, 64),    # 64 -> 32
            conv_block(64, 128),   # 32 -> 16
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        ela, grey = x[:, :1], x[:, 1:2]
        maps = torch.cat([ela, self.srm(grey)], dim=1)
        f = self.features(maps)
        pooled = torch.cat([f.amax(dim=(2, 3)), f.mean(dim=(2, 3))], dim=1)
        return self.classifier(pooled)


# Name kept for imports elsewhere
ELACNN = ELASRMCNN


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_model(path, device="cpu"):
    model = ELASRMCNN()
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state"] if "model_state" in state else state)
    model.to(device).eval()
    return model
