"""
AnimalVox AI — Call Sequence Transformer (Stage 6)
Processes sliding window of last 5 calls to capture
escalation, de-escalation, and conversational patterns.
"""

import torch
import torch.nn as nn
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class CallSequenceTransformer(nn.Module):
    """
    Lightweight 4-layer Transformer for call sequence context.
    
    Input per call: [12 behavior labels + 1 intensity + 1 urgency 
                     + 256 audio embed + 6 species one-hot] = 276-dim
    
    Captures temporal patterns:
    - Escalation (calls getting more urgent)
    - De-escalation (calming down)
    - Call-and-response between individuals
    - Behavioral narrative of a session
    """

    def __init__(self, embed_dim=276, n_heads=4, n_layers=4, seq_len=5, ff_dim=512, dropout=0.1, ctx_dim=128):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len

        # Positional encoding for temporal order
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim) * 0.02)

        # Transformer encoder
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=ff_dim, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

        # Context output head
        self.context_head = nn.Sequential(
            nn.Linear(embed_dim, ctx_dim),
            nn.LayerNorm(ctx_dim),
            nn.GELU()
        )

        # Escalation detector
        self.escalation_head = nn.Sequential(
            nn.Linear(ctx_dim, 32), nn.GELU(),
            nn.Linear(32, 3)  # [de-escalation, stable, escalation]
        )

    def forward(self, call_sequence: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            call_sequence: (B, seq_len, embed_dim) — last N calls
            mask: Optional padding mask for shorter sequences
        Returns:
            context: (B, ctx_dim) — context embedding for LLM
            escalation: (B, 3) — escalation pattern logits
        """
        B, S, D = call_sequence.shape
        x = call_sequence + self.pos_embed[:, :S, :]
        encoded = self.transformer(x, src_key_padding_mask=mask)

        # Use last call's context-aware embedding
        last_ctx = self.context_head(encoded[:, -1, :])
        escalation = self.escalation_head(last_ctx)

        return last_ctx, escalation

    def build_call_embedding(self, behavior_labels, intensity, urgency, audio_embed, species_onehot):
        """Helper to build a single call's embedding vector."""
        return torch.cat([behavior_labels, intensity, urgency, audio_embed, species_onehot], dim=-1)
