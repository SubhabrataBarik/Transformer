import torch

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384 # d_model in the paper
n_head = 6 # h in the paper
n_layer = 6 # N in the paper (number of stacked decoder blocks)
dropout = 0.2 # P_drop in the paper (section 5.4)
positional_encoding = 'learned' # 'learned' (GPT style) or 'sinusoidal' (paper section 3.5)
# ------------
