"""
Structure encoder that consumes precomputed MACE features.

Input: (batch_size, 128) tensor of absorber-pooled MACE-MP-0 features
Output: (batch_size, 256) L2-normalized embedding for contrastive alignment

The MACE backbone is entirely frozen and applied offline (see scripts/preprocess_mace_features.py).
Only the projection head trains.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MACEStructureEncoder(nn.Module):
    def __init__(self, in_dim: int = 128, hidden_dim: int = 512, out_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, mace_features):
        x = self.projection(mace_features)
        return F.normalize(x, dim=1)