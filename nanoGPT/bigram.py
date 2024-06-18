import torch
import torch.nn as nn 
from torch.nn import functional as F 

# hyperparameters
batch_size = 4  # how many independent sequences will be processed in parallel?
block_size = 8  # what is the maximum context length for prediction?
max_iters = 5000
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
# ---------------

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as file:
  text = file.read()

# Unqiue characters in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Create mapping for characters
itos = {i:s for i, s in enumerate(chars)}
stoi = {s:i for i, s in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # take a string output a list of integer
decode = lambda l: ''.join([itos[i] for i in l]) # take a list of integer output a string

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
  # generate small batch of data of inputs x and target y
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(train_data) - block_size, (batch_size, ))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1: i+block_size+1] for i in ix])
  return x, y

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

class Head(nn.Module):
    """one head of self attention"""

    def __init__(self, head_size) -> None:
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)     # (B, T, C)
        q = self.query(x)   # (B, T, C)
        # compute attention scores ('affinities)
        wei = q @ k.transpose(-2, -1) * C**-0.5     # (B, T, C) @ (B, C, T) --> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))    # (B, T, T)
        wei = F.softmax(wei, dim=-1)    # (B, T, T)
        # perform weighted aggregation of values
        v = self.value(x)
        out = wei @ v   # (B, T, T) @ (B, T, C) --> (B, T, C)
        return out

class BigramLanguageModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.poition_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_head = Head(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensors of integers
        token_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.poition_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = token_emb + pos_emb # (B, T, C)
        x = self.sa_head(x)
        logits = self.lm_head(x) 

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to last block size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus on last time step
            logits = logits[:, -1, :]   # becomes (B, T)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=1)
            # sample from probs
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
model = BigramLanguageModel()
m = model.to(device)

# create a pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate loss on train and val split
    if iter % eval.interval == 0:
        losses = estimate_loss()
        print(f'Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}')

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))