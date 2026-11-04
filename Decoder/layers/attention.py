import torch
import torch.nn as nn
from torch.nn import functional as F

def scaled_dot_product_attention(q, k, v, mask=None, dropout=None):
    """ Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V    (paper section 3.2.1, equation 1) """
    d_k = k.shape[-1]
    # MatMul + Scale: compute attention scores ("affinities")
    wei = q @ k.transpose(-2,-1) * d_k**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
    # Mask (opt.): a position may not look at future positions
    if mask is not None:
        wei = wei.masked_fill(mask == 0, float('-inf')) # (B, T, T)
    # SoftMax: each row becomes a probability distribution over the positions it attends to
    wei = F.softmax(wei, dim=-1) # (B, T, T)
    if dropout is not None:
        wei = dropout(wei)
    # MatMul: perform the weighted aggregation of the values
    return wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)

class Head(nn.Module):
    """ one head of masked self-attention

    Q, K and V all come from the same input x (self-attention). Each is a linear projection
    of x down to head_size = d_k = d_v = d_model / h dimensions.
    """

    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)   # W^K
        self.query = nn.Linear(n_embd, head_size, bias=False) # W^Q
        self.value = nn.Linear(n_embd, head_size, bias=False) # W^V
        # lower-triangular matrix: row t has ones in columns 0..t, i.e. the causal mask
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        v = self.value(x) # (B,T,hs)
        return scaled_dot_product_attention(q, k, v, mask=self.tril[:T, :T], dropout=self.dropout)
