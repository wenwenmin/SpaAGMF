import torch.nn as nn
from models.gate_attention import GateAttention

class AttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = GateAttention(embed_dim, num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(embed_dim)

    def forward(self, q, k, v):
        attn_out, _ = self.attn(q, k, v)
        x = q + self.dropout(attn_out)

        return x


class SelfAttention(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, dropout=0.2):
        super().__init__()
        self.attention_layer = nn.ModuleList([
            AttentionLayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.attention_layer:
            x = layer(x, x, x)

        return x