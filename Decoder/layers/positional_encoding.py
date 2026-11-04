import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """ "Positional Encoding" box in figure 1 (paper section 3.5)

    Attention has no notion of order, so a vector that depends only on the position is added
    to every token embedding:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    Fixed, not learned: computed once and stored as a buffer.
    """

    def __init__(self, block_size, n_embd):
        super().__init__()
        pe = torch.zeros(block_size, n_embd)
        pos = torch.arange(block_size, dtype=torch.float).unsqueeze(1) # (block_size, 1)
        # 1 / 10000^(2i/d_model), computed in log space for numerical stability
        div_term = torch.exp(torch.arange(0, n_embd, 2, dtype=torch.float) * (-math.log(10000.0) / n_embd)) # (C/2,)
        pe[:, 0::2] = torch.sin(pos * div_term) # even dimensions
        pe[:, 1::2] = torch.cos(pos * div_term) # odd dimensions
        self.register_buffer('pe', pe)

    def forward(self, T):
        return self.pe[:T] # (T,C)

class LearnedPositionalEncoding(nn.Module):
    """ learned alternative to the sinusoid (GPT style, used by the original gpt.py)

    The paper (section 3.5, table 3 row E) reports learned position embeddings give nearly
    identical results. Each position 0..block_size-1 gets its own trainable vector.
    """

    def __init__(self, block_size, n_embd):
        super().__init__()
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

    def forward(self, T):
        pos = torch.arange(T, device=self.position_embedding_table.weight.device)
        return self.position_embedding_table(pos) # (T,C)
