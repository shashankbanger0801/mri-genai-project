import torch
import torch.nn as nn
import torch.nn.functional as F

class L1Loss(nn.Module):
    def forward(self, pred, target):
        return torch.mean(torch.abs(pred - target))


class SSIM3D(nn.Module):
    """
    Placeholder — MONAI has full SSIM3D implemented,
    but this will allow switching to SSIM quickly later.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return 1 - torch.mean((pred * target) / (pred.abs() + target.abs() + 1e-8))
