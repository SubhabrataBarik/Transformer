import argparse
import sys
import torch

from config import device, checkpoint_path
from model import GPTLanguageModel

parser = argparse.ArgumentParser(description='Continue a text prompt with the trained model.')
parser.add_argument('prompt', nargs='?', help='starting text; omit it to type prompts interactively')
parser.add_argument('--max_new_tokens', type=int, default=500, help='number of characters to generate')
parser.add_argument('--checkpoint', default=checkpoint_path, help='path to the .pt file saved by train.py')
parser.add_argument('--seed', type=int, default=None, help='fix the random sampling for repeatable output')
args = parser.parse_args()

# rebuild the model exactly as it was trained, then load the learned weights
checkpoint = torch.load(args.checkpoint, map_location=device)
model = GPTLanguageModel(**checkpoint['model_args'])
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval() # disables dropout
print(f"loaded {args.checkpoint} (step {checkpoint['iter']}, val loss {checkpoint['val_loss']:.4f})")

# same character vocabulary as training, so the token ids mean the same thing
chars = checkpoint['chars']
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
decode = lambda l: ''.join([itos[i] for i in l])

def encode(s):
    unknown = sorted(set(s) - set(stoi))
    if unknown:
        raise ValueError(f"characters not in the training vocabulary: {unknown!r}")
    return [stoi[c] for c in s]

def complete(prompt):
    prompt = prompt.replace('\\n', '\n') # let "ROMEO:\nI" typed on the command line mean a real line break
    # an empty prompt starts from token 0 ('\n'), like train.py does
    idx = encode(prompt) if prompt else [0]
    context = torch.tensor([idx], dtype=torch.long, device=device) # (1, T)
    with torch.no_grad():
        out = model.generate(context, max_new_tokens=args.max_new_tokens)
    return decode(out[0].tolist())

if args.seed is not None:
    torch.manual_seed(args.seed)

if args.prompt is not None:
    try:
        print(complete(args.prompt))
    except ValueError as e:
        sys.exit(str(e))
else:
    print("type a prompt and press Enter (empty line = no prompt, Ctrl+C to quit)")
    while True:
        try:
            prompt = input('> ')
        except (EOFError, KeyboardInterrupt):
            break
        try:
            print(complete(prompt))
        except ValueError as e:
            print(e)
        print('-' * 40)
