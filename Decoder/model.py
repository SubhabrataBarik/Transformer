import torch
import torch.nn as nn
from torch.nn import functional as F

from layers.output_embedding import OutputEmbedding
from layers.positional_encoding import PositionalEncoding, LearnedPositionalEncoding
from decoder_block import DecoderBlock

class GPTLanguageModel(nn.Module):
    """ the decoder side of figure 1, read bottom to top:

    Outputs (shifted right)               (data.py: y is x shifted by one)
      -> Output Embedding                 (layers/output_embedding.py)
      +  Positional Encoding              (layers/positional_encoding.py)
      -> Nx DecoderBlock                  (decoder_block.py)
           -> Masked Multi-Head Attention (layers/masked_multi_head_attention.py -> layers/attention.py)
           -> Add & Norm                  (layers/add_and_norm.py)
           -> Feed Forward                (layers/feed_forward.py)
           -> Add & Norm                  (layers/add_and_norm.py)
      -> LayerNorm (final, needed by pre-norm; not in figure 1)
      -> Linear
      -> Softmax -> Output Probabilities
    """

    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout, positional_encoding='learned'):
        super().__init__()
        self.block_size = block_size
        self.output_embedding = OutputEmbedding(vocab_size, n_embd)
        if positional_encoding == 'sinusoidal':
            self.positional_encoding = PositionalEncoding(block_size, n_embd)
        else:
            self.positional_encoding = LearnedPositionalEncoding(block_size, n_embd)
        self.blocks = nn.Sequential(*[DecoderBlock(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.linear = nn.Linear(n_embd, vocab_size) # "Linear" box: d_model -> one score (logit) per token

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.output_embedding(idx) # (B,T,C)
        pos_emb = self.positional_encoding(T) # (T,C)
        x = tok_emb + pos_emb # (B,T,C), the (+) in figure 1; pos_emb broadcasts over the batch
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.linear(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) # applies the Softmax internally

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # "Softmax" box: turn logits into "Output Probabilities"
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
