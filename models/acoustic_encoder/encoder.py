"""
AnimalVox AI — Acoustic Encoder (BEATs Fine-tuning)
=====================================================
Fine-tunes the BEATs audio encoder (Microsoft) separately
for each species group. Produces 768-dim audio embeddings
projected to 256-dim species-specific representations.

Part of Stage 2, Path A of the AnimalVox AI pipeline.

BEATs was pre-trained on AudioSet with 2M+ clips and produces
embeddings that encode texture, rhythm, frequency content,
and temporal patterns of sound.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class AnimalAcousticEncoder(nn.Module):
    """
    BEATs-based acoustic encoder with species-specific projection.

    Architecture:
        Raw Audio → BEATs Encoder → 768-dim embedding
        → Projection Head → 256-dim species-optimized embedding

    The projection head adapts the general-purpose BEATs
    embeddings to be specifically useful for animal vocalization
    analysis within each species group.
    """

    def __init__(
        self,
        species_group: str = "birds",
        embed_dim: int = 768,
        projection_dim: int = 256,
        pretrained: bool = True,
        freeze_encoder_layers: int = 0
    ):
        """
        Args:
            species_group: One of 'birds', 'dogs', 'cats', 'dolphins', 'whales', 'frogs'
            embed_dim: BEATs embedding dimension (768 default)
            projection_dim: Output projection dimension
            pretrained: Load pretrained BEATs weights
            freeze_encoder_layers: Number of early encoder layers to freeze
        """
        super().__init__()
        self.species_group = species_group
        self.embed_dim = embed_dim
        self.projection_dim = projection_dim

        # ── BEATs Encoder ────────────────────────────────────
        # In production, load from HuggingFace Hub:
        # self.encoder = AutoModel.from_pretrained("microsoft/BEATs")
        # For now, we use a wav2vec2-style encoder as a compatible backbone
        self.encoder = self._build_encoder(embed_dim, pretrained)

        # ── Species-Specific Projection Head ─────────────────
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, projection_dim),
            nn.LayerNorm(projection_dim)
        )

        # Optionally freeze early encoder layers for fine-tuning
        if freeze_encoder_layers > 0:
            self._freeze_layers(freeze_encoder_layers)

        logger.info(
            f"AnimalAcousticEncoder initialized for '{species_group}' "
            f"({embed_dim} → {projection_dim})"
        )

    def _build_encoder(self, embed_dim: int, pretrained: bool) -> nn.Module:
        """
        Build the audio encoder backbone.

        Uses a CNN + Transformer architecture similar to BEATs.
        In production, replace with actual BEATs weights from HuggingFace.
        """

        class AudioEncoderBackbone(nn.Module):
            """
            Simplified audio encoder backbone.
            In production, this is replaced by the full BEATs model.
            """
            def __init__(self, embed_dim):
                super().__init__()
                # CNN feature extractor (processes raw waveform)
                self.feature_extractor = nn.Sequential(
                    nn.Conv1d(1, 64, kernel_size=10, stride=5, padding=2),
                    nn.GELU(),
                    nn.BatchNorm1d(64),
                    nn.Conv1d(64, 128, kernel_size=8, stride=4, padding=2),
                    nn.GELU(),
                    nn.BatchNorm1d(128),
                    nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
                    nn.GELU(),
                    nn.BatchNorm1d(256),
                    nn.Conv1d(256, 512, kernel_size=4, stride=2, padding=1),
                    nn.GELU(),
                    nn.BatchNorm1d(512),
                    nn.Conv1d(512, embed_dim, kernel_size=4, stride=2, padding=1),
                    nn.GELU(),
                )

                # Transformer encoder layers
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=8,
                    dim_feedforward=2048,
                    dropout=0.1,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(
                    encoder_layer, num_layers=6
                )

                # Positional encoding
                self.pos_embed = nn.Parameter(
                    torch.randn(1, 500, embed_dim) * 0.02
                )

            def forward(self, x):
                # x shape: (B, 1, T) or (B, T)
                if x.dim() == 2:
                    x = x.unsqueeze(1)  # (B, 1, T)

                # CNN features
                features = self.feature_extractor(x)  # (B, embed_dim, T')
                features = features.transpose(1, 2)    # (B, T', embed_dim)

                # Add positional encoding
                T_prime = features.shape[1]
                if T_prime <= self.pos_embed.shape[1]:
                    features = features + self.pos_embed[:, :T_prime, :]
                else:
                    # Interpolate positional encoding
                    pos = self.pos_embed.transpose(1, 2)
                    pos = nn.functional.interpolate(
                        pos, size=T_prime, mode='linear'
                    )
                    features = features + pos.transpose(1, 2)

                # Transformer
                encoded = self.transformer(features)  # (B, T', embed_dim)
                return encoded

        return AudioEncoderBackbone(embed_dim)

    def _freeze_layers(self, num_layers: int):
        """Freeze the first N transformer layers for fine-tuning."""
        if hasattr(self.encoder, 'transformer'):
            for i, layer in enumerate(self.encoder.transformer.layers):
                if i < num_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
            logger.info(f"Froze first {num_layers} encoder layers")

    def forward(self, audio_input: torch.Tensor) -> torch.Tensor:
        """
        Process raw audio through encoder + projection.

        Args:
            audio_input: Raw audio tensor of shape (B, T)
                         where T = num_samples (e.g., 80000 for 5s @ 16kHz)

        Returns:
            Projected embedding of shape (B, projection_dim)
        """
        # Encode audio
        encoded = self.encoder(audio_input)  # (B, T', embed_dim)

        # Mean pool over time dimension
        embeddings = encoded.mean(dim=1)  # (B, embed_dim)

        # Project to species-specific space
        projected = self.projection(embeddings)  # (B, projection_dim)

        return projected

    def get_full_embeddings(self, audio_input: torch.Tensor) -> torch.Tensor:
        """Get raw encoder embeddings without projection (for analysis)."""
        encoded = self.encoder(audio_input)
        return encoded.mean(dim=1)
