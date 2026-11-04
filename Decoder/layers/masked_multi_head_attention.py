import torch
import torch.nn as nn

from layers.attention import Head

class MaskedMultiHeadAttention(nn.Module):
    """ "Masked Multi-Head Attention" box in figure 1 (paper section 3.2.2)

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
    where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
    h heads of size d_model/h run in parallel, so the total cost is similar to one full-size head,
    but each head can learn to attend to a different kind of relationship.
    "Masked": every Head applies the causal mask, so position t only sees positions <= t.
    """

    def __init__(self, n_embd, num_heads, head_size, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd) # W^O
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1) # Concat: (B,T,hs*num_heads) = (B,T,C)
        out = self.dropout(self.proj(out)) # (B,T,C)
        return out
