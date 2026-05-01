import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_coefficient(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    probs = torch.sigmoid(logits).view(logits.size(0), -1)
    targets = targets.view(targets.size(0), -1).float()
    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    return ((2 * intersection + eps) / (union + eps)).mean()


def iou_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-7) -> torch.Tensor:
    preds = (torch.sigmoid(logits) > threshold).float().view(logits.size(0), -1)
    targets = targets.view(targets.size(0), -1).float()
    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection
    return ((intersection + eps) / (union + eps)).mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets.float())
        dice_loss = 1.0 - dice_coefficient(logits, targets)
        return self.bce_weight * bce + (1 - self.bce_weight) * dice_loss