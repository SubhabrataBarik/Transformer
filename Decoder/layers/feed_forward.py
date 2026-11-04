import torch.nn as nn

class FeedForward(nn.Module):
    """ "Feed Forward" box in figure 1 (paper section 3.3)

    FFN(x) = max(0, x W_1 + b_1) W_2 + b_2
    Applied to each position separately and identically. The inner layer is 4x wider
    (paper: d_model = 512, d_ff = 2048).
    """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # W_1, b_1
            nn.ReLU(),                     # max(0, .)
            nn.Linear(4 * n_embd, n_embd), # W_2, b_2
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x) # (B,T,C)
