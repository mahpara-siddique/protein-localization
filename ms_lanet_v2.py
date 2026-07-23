"""
MS-LANet v2: Modular architecture with swappable pooling mechanisms
for comprehensive ablation studies.

Pooling Strategies:
  - 'mean_only'       : Global Mean Pooling
  - 'max_only'        : Global Max Pooling
  - 'mean_max'        : Mean + Max concatenated
  - 'conv_pool_only'  : Convolutional Weighted Pooling (our original v1 mechanism)
  - 'conv_pool_mean'  : Conv Weighted Pooling + Mean Pooling (original MS-LANet v1)
  - 'self_attn_only'  : TRUE Multi-Head Self-Attention Pooling
  - 'self_attn_mean'  : Self-Attention + Mean Pooling (PROPOSED NOVEL MODEL v2)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# POOLING MODULE 1: Convolutional Weighted Pooling
# (Our original mechanism — correctly named, NOT "attention")
# ============================================================
class ConvWeightedPooling(nn.Module):
    """
    Applies 1D convolutions to compute position-wise scores,
    then softmax-normalizes them to create weighted pooling.
    This is NOT attention — there is no query-key interaction.
    """
    def __init__(self, embed_dim, num_heads=8, kernel_size=9):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv_keys = nn.Conv1d(embed_dim, num_heads, kernel_size, padding=padding)
        self.conv_values = nn.Conv1d(embed_dim, embed_dim, kernel_size, padding=padding)

    def forward(self, x, mask=None):
        # x: [B, L, D]
        x_t = x.transpose(1, 2)  # [B, D, L]
        keys = self.conv_keys(x_t)      # [B, num_heads, L]
        values = self.conv_values(x_t)   # [B, D, L]

        if mask is not None:
            mask_exp = mask.unsqueeze(1)  # [B, 1, L]
            keys = keys.masked_fill(~mask_exp, -1e9)

        weights = F.softmax(keys, dim=-1)              # [B, num_heads, L]
        weights = weights.mean(dim=1, keepdim=True)    # [B, 1, L]
        out = (values * weights).sum(dim=-1)           # [B, D]
        return out


# ============================================================
# POOLING MODULE 2: TRUE Multi-Head Self-Attention Pooling
# (Genuine attention with learnable query-key interaction)
# ============================================================
class SelfAttentionPooling(nn.Module):
    """
    TRUE attention mechanism:
    - A learnable query vector attends over all sequence positions
    - Attention weights = softmax(Q @ K^T / sqrt(d_k))
    - Each position's contribution depends on its RELATIONSHIP to the query
    - This IS genuine attention with query-key interaction
    """
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, dropout=0.1
        )
        # Learnable query vector (like BERT's [CLS] token)
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        nn.init.xavier_uniform_(self.query)

    def forward(self, x, mask=None):
        # x: [B, L, D]
        B = x.size(0)
        query = self.query.expand(B, -1, -1)  # [B, 1, D]

        # MultiheadAttention expects key_padding_mask: True = IGNORE
        key_padding_mask = ~mask if mask is not None else None

        attn_out, _ = self.attention(
            query, x, x, key_padding_mask=key_padding_mask
        )  # [B, 1, D]
        return attn_out.squeeze(1)  # [B, D]


# ============================================================
# MAIN MODEL: MS-LANet v2 (Modular)
# ============================================================
class MSLANetV2(nn.Module):
    """
    Multi-Scale Light Attention Network v2.
    Supports multiple pooling strategies for ablation studies.
    """

    # Map strategy names to the number of pooling branches
    STRATEGY_BRANCHES = {
        'mean_only': 1,
        'max_only': 1,
        'mean_max': 2,
        'conv_pool_only': 1,
        'conv_pool_mean': 2,
        'self_attn_only': 1,
        'self_attn_mean': 2,
    }

    def __init__(
        self,
        embedding_dim=1280,
        num_heads=8,
        kernel_size=9,
        hidden_dim=512,
        num_classes=10,
        dropout=0.4,
        pooling_strategy='self_attn_mean',
        use_layernorm=True
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.pooling_strategy = pooling_strategy
        self.use_layernorm = use_layernorm

        # --- Initialize pooling modules based on strategy ---
        if 'conv_pool' in pooling_strategy:
            self.conv_pool = ConvWeightedPooling(embedding_dim, num_heads, kernel_size)

        if 'self_attn' in pooling_strategy:
            self.self_attn_pool = SelfAttentionPooling(embedding_dim, num_heads)

        # --- Calculate concatenated dimension ---
        num_branches = self.STRATEGY_BRANCHES[pooling_strategy]
        concat_dim = num_branches * embedding_dim

        # --- Classifier Head ---
        if use_layernorm:
            self.layer_norm = nn.LayerNorm(concat_dim)

        self.fc1 = nn.Linear(concat_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def _get_mean_pooled(self, x, mask):
        """Global Mean Pooling (masked)."""
        if mask is not None:
            x_masked = x * mask.unsqueeze(-1).float()
            lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
            return x_masked.sum(dim=1) / lengths
        return x.mean(dim=1)

    def _get_max_pooled(self, x, mask):
        """Global Max Pooling (masked)."""
        if mask is not None:
            x_masked = x.masked_fill(~mask.unsqueeze(-1), -1e9)
            return x_masked.max(dim=1)[0]
        return x.max(dim=1)[0]

    def forward(self, x, mask=None):
        """
        Args:
            x: [B, L, D] residue-level embeddings
            mask: [B, L] boolean mask (True = real token, False = padding)
        Returns:
            logits: [B, num_classes]
        """
        vectors = []

        # --- Select pooling based on strategy ---
        if self.pooling_strategy == 'mean_only':
            vectors.append(self._get_mean_pooled(x, mask))

        elif self.pooling_strategy == 'max_only':
            vectors.append(self._get_max_pooled(x, mask))

        elif self.pooling_strategy == 'mean_max':
            vectors.append(self._get_mean_pooled(x, mask))
            vectors.append(self._get_max_pooled(x, mask))

        elif self.pooling_strategy == 'conv_pool_only':
            vectors.append(self.conv_pool(x, mask))

        elif self.pooling_strategy == 'conv_pool_mean':
            vectors.append(self.conv_pool(x, mask))
            vectors.append(self._get_mean_pooled(x, mask))

        elif self.pooling_strategy == 'self_attn_only':
            vectors.append(self.self_attn_pool(x, mask))

        elif self.pooling_strategy == 'self_attn_mean':
            vectors.append(self.self_attn_pool(x, mask))
            vectors.append(self._get_mean_pooled(x, mask))

        # --- Concatenate all branch outputs ---
        fused = torch.cat(vectors, dim=-1)  # [B, concat_dim]

        # --- Classifier Head ---
        if self.use_layernorm:
            fused = self.layer_norm(fused)

        out = F.gelu(self.fc1(fused))
        out = self.dropout(out)
        logits = self.fc2(out)

        return logits


# ============================================================
# QUICK SANITY TEST
# ============================================================
if __name__ == "__main__":
    print("Testing all 7 pooling strategies...\n")

    test_input = torch.randn(2, 50, 1280)
    test_mask = torch.zeros(2, 50, dtype=torch.bool)
    test_mask[0, :40] = True
    test_mask[1, :30] = True

    for strategy in MSLANetV2.STRATEGY_BRANCHES.keys():
        model = MSLANetV2(pooling_strategy=strategy)
        output = model(test_input, test_mask)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  {strategy:20s} | Output: {output.shape} | Params: {num_params:,}")

    print("\n✅ All strategies passed!")