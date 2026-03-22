import math
import torch
import torch.nn as nn

class GateAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        """
        Gated multi-head attention module

        Args:
            embed_dim: feature dimension
            num_heads: number of attention heads
            dropout: dropout probability
        """
        super(GateAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.head_dim = int(embed_dim // num_heads)

        self.q_proj = nn.Linear(embed_dim, embed_dim * 2, bias=True)  # derive the gate score
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)

    def forward(self, q, k, v):
        """
        Forward pass

        Args:
            q: query tensor (B, L_q, D)
            k: key tensor   (B, L_k, D)
            v: value tensor (B, L_k, D)
        """
        # 1. Project x to q, k ,v
        query = self.q_proj(q)  # (batch_size, seq_len, embed_dim * 2)
        key = self.k_proj(k)  # (batch_size, seq_len, embed_dim)
        value = self.v_proj(v)  # (batch_size, seq_len, embed_dim)

        # 2. Split query and gate features
        batch_size, seq_len_k, embed_dim = key.shape
        seq_len_q = query.shape[1]
        query = query.reshape(batch_size, seq_len_q, self.num_heads, self.head_dim * 2)
        query, gate_score = torch.split(query,[self.head_dim, self.head_dim], dim=-1)  # (batch_size, seq_len, num_heads, head_dim)

        # 3. Reshape and Normalization
        query = query.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        key = key.reshape(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.reshape(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)

        query = self.q_norm(query)
        key = self.k_norm(key)

        # 4. Scaled dot-product attention
        attn_weights = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.head_dim)  # (batch_size, num_heads, seq_len, seq_len)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)  # (batch_size, num_heads, seq_len, seq_len)
        attn_weights = self.dropout(attn_weights)

        # 5. Attention-weighted sum
        attn_output = torch.matmul(attn_weights, value)  # (batch_size, num_heads, seq_len, head_dim)

        # 6. Gate modulation
        attn_output = attn_output.transpose(1, 2).contiguous()  # (batch_size, seq_len, num_heads, head_dim)
        attn_output = attn_output * torch.sigmoid(gate_score)  # (batch_size, seq_len, num_heads, head_dim)

        # 7. Head concatenation and output projection
        attn_output = attn_output.reshape(batch_size, seq_len_q, -1)  # (batch_size, seq_len, num_heads * head_dim == embed_dim)
        attn_output = self.o_proj(attn_output)  # (batch_size, seq_len, embed_dim)

        return attn_output, attn_weights







