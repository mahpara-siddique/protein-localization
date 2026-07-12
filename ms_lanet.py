import torch
import torch.nn as nn
import torch.nn.functional as F

class MSLANet(nn.Module):
    def __init__(self, embedding_dim=1280, num_heads=8, kernel_size=9, hidden_dim=512, num_classes=10, dropout=0.4):
        super(MSLANet, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        # --- BRANCH A: Light Attention Pooling ---
        padding = (kernel_size - 1) // 2
        self.conv_keys = nn.Conv1d(
            in_channels=embedding_dim, 
            out_channels=num_heads, 
            kernel_size=kernel_size, 
            padding=padding
        )
        
        self.conv_values = nn.Conv1d(
            in_channels=embedding_dim, 
            out_channels=embedding_dim, 
            kernel_size=kernel_size, 
            padding=padding
        )
        
        # --- CLASSIFIER HEAD ---
        # Input dimension is 2 * embedding_dim because we concatenate:
        # [Light Attention Vector, Global Mean-Pooled] (Max-pooling dropped to reduce noise/overfitting)
        concat_dim = 2 * embedding_dim
        
        self.layer_norm = nn.LayerNorm(concat_dim)
        self.fc1 = nn.Linear(concat_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.size()
        x_transposed = x.transpose(1, 2)
        
        # --- BRANCH A: Light Attention ---
        keys = self.conv_keys(x_transposed)
        values = self.conv_values(x_transposed)
        
        if mask is not None:
            mask_expanded = mask.unsqueeze(1)
            keys = keys.masked_fill(~mask_expanded, -1e9)
        
        attn_weights = F.softmax(keys, dim=-1)
        attn_weights = attn_weights.mean(dim=1, keepdim=True)
        light_attention_out = (values * attn_weights).sum(dim=-1)
        
        # --- BRANCH B: Global Mean Pooling ---
        if mask is not None:
            x_masked = x * mask.unsqueeze(-1).float()
            seq_lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
            mean_pooled = x_masked.sum(dim=1) / seq_lengths
        else:
            mean_pooled = x.mean(dim=1)
            
        # --- CONCATENATION ---
        # Shape: [batch_size, 2 * embedding_dim]
        fused_features = torch.cat([light_attention_out, mean_pooled], dim=-1)
        
        # --- CLASSIFIER HEAD ---
        out = self.layer_norm(fused_features)
        out = F.gelu(self.fc1(out))
        out = self.dropout(out)
        logits = self.fc2(out)
        
        return logits

if __name__ == "__main__":
    test_input = torch.randn(2, 50, 1280)
    test_mask = torch.zeros(2, 50, dtype=torch.bool)
    test_mask[0, :40] = True
    test_mask[1, :30] = True
    
    model = MSLANet()
    output = model(test_input, test_mask)
    print("Test output shape:", output.shape)  # Expected: [2, 10]