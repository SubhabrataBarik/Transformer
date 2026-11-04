import torch.nn as nn

class OutputEmbedding(nn.Module):
    """ "Output Embedding" box in figure 1 (paper section 3.4)

    Each token id is looked up in a learned table and turned into a vector of size d_model.
    The paper also multiplies the result by sqrt(d_model); GPT-style models skip that and
    rely on weight init instead, so it is left out here.
    """

    def __init__(self, vocab_size, n_embd):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

    def forward(self, idx):
        # idx: (B,T) tensor of token ids
        return self.token_embedding_table(idx) # (B,T,C)
