import torch.nn as nn

from layers.masked_multi_head_attention import MaskedMultiHeadAttention
from layers.add_and_norm import AddNorm
from layers.feed_forward import FeedForward

class DecoderBlock(nn.Module):
    """ one decoder layer, the "Nx" box on the right of figure 1 (paper section 3.1)

    Masked Multi-Head Attention -> Add & Norm -> Feed Forward -> Add & Norm
    ("communication" between tokens, then "computation" on each token)

    The middle "Multi-Head Attention" + "Add & Norm" of figure 1 (cross-attention over the
    encoder output) is absent: there is no encoder here (decoder-only, like GPT).
    """

    def __init__(self, n_embd, n_head, block_size, dropout):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.masked_multi_head_attention = MaskedMultiHeadAttention(n_embd, n_head, head_size, block_size, dropout)
        self.add_norm1 = AddNorm(n_embd)
        self.feed_forward = FeedForward(n_embd, dropout)
        self.add_norm2 = AddNorm(n_embd)

    def forward(self, x):
        x = self.add_norm1(x, self.masked_multi_head_attention) # (B,T,C)
        x = self.add_norm2(x, self.feed_forward) # (B,T,C)
        return x
