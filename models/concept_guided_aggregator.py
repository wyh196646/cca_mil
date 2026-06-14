import torch
import torch.nn as nn


class ConceptGuidedAggregator(nn.Module):
    def __init__(self, feature_dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, concept_features, patch_features, concept_weights):
        query = concept_features.unsqueeze(0)
        key_value = patch_features.unsqueeze(0)
        attended, attn_weights = self.attn(query, key_value, key_value, need_weights=True)
        attended = self.norm(attended.squeeze(0) + concept_features)

        weights = concept_weights.to(attended.device).float().unsqueeze(-1)
        pooled = (attended * weights).sum(dim=0) / weights.sum().clamp_min(1e-6)

        return pooled, attended, attn_weights.squeeze(0)
