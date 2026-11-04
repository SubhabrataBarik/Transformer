import torch.nn as nn

class AddNorm(nn.Module):
    """ "Add & Norm" box in figure 1 (paper section 3.1)

    "Add" is the residual connection (the arrow that skips around each sub-layer),
    "Norm" is LayerNorm.
    Paper (post-norm): LayerNorm(x + Sublayer(x))
    Here  (pre-norm):  x + Sublayer(LayerNorm(x))
    Pre-norm is what GPT-2 onwards use; it keeps a clean residual path and trains more stably.
    """

    def __init__(self, n_embd):
        super().__init__()
        self.ln = nn.LayerNorm(n_embd)

    def forward(self, x, sublayer):
        return x + sublayer(self.ln(x)) # (B,T,C)
