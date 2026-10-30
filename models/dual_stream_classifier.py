"""
AnimalVox AI — Dual-Stream Biometric + Semantic Acoustic Classifier
====================================================================
Specialized dual-tower neural network for animal emotion & breed recognition.

Separates input (266 dims) into:
1. Semantic Acoustic Stream (BEATs 256-D): general vocal timbre & spectral representation
2. Biometric Morphological Stream (Prosodic 10-D): physical features (F0, HNR, duration, ZCR, etc.)
Cross-gating & bilinear modulation ensures morphological biometrics directly control
classification logits rather than being drowned out by high-dimensional acoustic features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with LayerNorm + GELU + Dropout."""
    def __init__(self, dim: int, dropout: float = 0.15):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout * 0.5),
        )

    def forward(self, x):
        return x + self.block(x)


class DualStreamBiometricClassifier(nn.Module):
    """
    Dual-stream architecture with biometric cross-gating:
    - Stream 1 (Semantic): 256-d BEATs acoustic embedding -> 128-d representation
    - Stream 2 (Biometric): 10-d Prosody (F0, HNR, duration, ZCR) -> 64-d representation
    - Bilinear cross-gating: Biometric features gate and modulate the acoustic representation
    """
    def __init__(
        self,
        semantic_dim: int = 256,
        biometric_dim: int = 10,
        hidden_dim: int = 160,
        num_classes: int = 10,
        num_residual_blocks: int = 2,
        dropout: float = 0.15,
        is_multilabel: bool = False,
    ):
        super().__init__()
        self.semantic_dim = semantic_dim
        self.biometric_dim = biometric_dim
        self.is_multilabel = is_multilabel

        # Stream 1: Semantic Acoustic Tower
        self.semantic_tower = nn.Sequential(
            nn.Linear(semantic_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Stream 2: Biometric Morphological Tower
        self.biometric_tower = nn.Sequential(
            nn.Linear(biometric_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # Bilinear Cross-Gating Fusion
        self.gate_proj = nn.Linear(64, 128)
        self.bio_proj = nn.Linear(64, 128)
        self.fusion_norm = nn.LayerNorm(128)

        self.fusion_proj = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Residual Tower
        self.residuals = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout=dropout * 0.75) for _ in range(num_residual_blocks)]
        )

        # Classification Head
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Split into semantic and biometric streams
        if x.shape[-1] >= self.semantic_dim + self.biometric_dim:
            sem = x[..., :self.semantic_dim]
            bio = x[..., self.semantic_dim:self.semantic_dim + self.biometric_dim]
        elif x.shape[-1] == self.semantic_dim:
            sem = x
            bio = torch.zeros(x.shape[0], self.biometric_dim, device=x.device, dtype=x.dtype)
        else:
            sem = F.pad(x, (0, max(0, self.semantic_dim - x.shape[-1])))[..., :self.semantic_dim]
            bio = torch.zeros(x.shape[0], self.biometric_dim, device=x.device, dtype=x.dtype)

        h_sem = self.semantic_tower(sem)
        h_bio = self.biometric_tower(bio)

        gate = torch.sigmoid(self.gate_proj(h_bio))
        fused = h_sem * gate + self.bio_proj(h_bio)
        fused = self.fusion_norm(fused)

        h = self.fusion_proj(fused)
        h = self.residuals(h)
        logits = self.head(h)

        if self.is_multilabel:
            return torch.sigmoid(logits)
        return logits


class EnhancedClassifier(nn.Module):
    """
    Deeper MLP head with residual connections, LayerNorm, and GELU.
    Matches the trained checkpoints in animal_detector.pt and health_monitor.pt.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int,
                 num_residual_blocks: int = 2, dropout: float = 0.2,
                 is_multilabel: bool = False):
        super().__init__()
        self.is_multilabel = is_multilabel

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Residual tower
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout=dropout * 0.75)
              for _ in range(num_residual_blocks)]
        )

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        h = self.input_proj(x)
        h = self.residual_blocks(h)
        logits = self.head(h)
        if self.is_multilabel:
            return torch.sigmoid(logits)
        return logits

