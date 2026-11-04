import os
import torch

from config import (batch_size, block_size, max_iters, eval_interval, learning_rate, device,
                    eval_iters, n_embd, n_head, n_layer, dropout, positional_encoding, checkpoint_path)
from data import chars, vocab_size, decode, get_batch
from model import GPTLanguageModel

torch.manual_seed(1337)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def save_checkpoint(iter, val_loss):
    # everything generate.py needs to rebuild the model without input.txt
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'model_args': model_args,
        'chars': chars, # vocabulary: token id i <-> chars[i]
        'iter': iter,
        'val_loss': val_loss,
    }, checkpoint_path)

model_args = dict(vocab_size=vocab_size, n_embd=n_embd, n_head=n_head, n_layer=n_layer,
                  block_size=block_size, dropout=dropout, positional_encoding=positional_encoding)
model = GPTLanguageModel(**model_args)
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

best_val_loss = float('inf')
for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        # keep only the best model: past a point the train loss keeps falling but val loss rises (overfitting)
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val'].item()
            save_checkpoint(iter, best_val_loss)
            print(f"  saved checkpoint (val loss {best_val_loss:.4f}) to {checkpoint_path}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
