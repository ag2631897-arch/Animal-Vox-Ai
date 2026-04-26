"""
AnimalVox AI — Behavioral Classifier
Multi-label classifier with intensity and urgency heads.
Fuses BEATs (256d) + CNN (256d) + Prosodic (10d) features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SpectrogramCNN(nn.Module):
    """EfficientNet-style CNN for mel-spectrogram analysis (Path B)."""

    def __init__(self, output_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.BatchNorm2d(512), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(512, output_dim), nn.LayerNorm(output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class FocalLoss(nn.Module):
    """Focal Loss for class imbalance (Lin et al., 2017)."""
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy(preds, targets, reduction='none')
        pt = torch.where(targets == 1, preds, 1 - preds)
        weight = (1 - pt) ** self.gamma
        if self.alpha is not None:
            at = torch.where(targets == 1, self.alpha.to(preds.device), 1 - self.alpha.to(preds.device))
            weight = weight * at
        return (weight * bce).mean()


class BehavioralClassifier(nn.Module):
    """
    Multi-label behavioral classifier.
    Input: [BEATs(256) | CNN(256) | Prosodic(10)] = 522-dim
    Output: 12 behavior labels + intensity + urgency
    """

    def __init__(self, input_dim: int = 522, hidden: int = 512, out: int = 256, n_cls: int = 12, drop: float = 0.2):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(hidden, out), nn.GELU(), nn.Dropout(drop * 0.5),
        )
        self.classifier = nn.Linear(out, n_cls)
        self.intensity_head = nn.Sequential(nn.Linear(out, 64), nn.GELU(), nn.Linear(64, 1))
        self.urgency_head = nn.Sequential(nn.Linear(out, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.fusion(x)
        labels = torch.sigmoid(self.classifier(h))
        intensity = torch.sigmoid(self.intensity_head(h))
        urgency = torch.sigmoid(self.urgency_head(h))
        return labels, intensity, urgency

    def get_fused(self, x: torch.Tensor) -> torch.Tensor:
        return self.fusion(x)
