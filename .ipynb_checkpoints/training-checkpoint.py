import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle
import argparse

parser = argparse.ArgumentParser(description = 'This is a demonstration program')

# here we add an argument to the parse, specifying the expected type, a help message, etc.
parser.add_argument('-bs', type = str, required = True, help = 'Please provide a batch_size')

args = parser.parse_args()

# Now we can use the argument value in our program
print(f'batch size: {args.bs}')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

block_size = 32
batch_size = 128
batch_size = args.bs
max_iters = 200
learning_rate = 3e-4
eval_iters = 100
# number of embedding dimensions
n_embd = 384
# number of decoder blocks
n_layer = 1
# number of heads
n_head = 1
# makes a certain percentage of neurons drop out to prevent overfitting
dropout = 0.2

print(device)

chars = ""
with open('vocab.txt','r',encoding='utf-8') as f:
    text = f.read()
    chars = sorted(list(set(text)))
    
vocab_size = len(chars)

string_to_int = { ch:i for i, ch in enumerate(chars)}
int_to_string = { i:ch for i, ch in enumerate(chars)}
encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

# memory map for using small snippets of text from a single file of any size
def get_random_chunk(split):
    filename = "train_split.txt" if split == 'train' else "val_split.txt"
    with open(filename, 'rb') as f:
        with mmap.mmap(f.fileno(), 0, access = mmap.ACCESS_READ) as mm:
            # Determine the file size and a random position to start reading
            file_size = len(mm)
            start_pos = random.randint(0, (file_size) - block_size * batch_size)

            # Seek to the random position and read the block of text
            mm.seek(start_pos)
            block = mm.read(block_size * batch_size - 1)

            # Decode the block to a string, ignoring any invalid byte sequences
            decoded_block = block.decode('utf-8', errors = 'ignore').replace('\r','')

            # Train and test splits
            data = torch.tensor(encode(decoded_block), dtype = torch.long)
    return data

# returns a batch of input-target sequences from either the training or validation set
def get_batch(split):
    data = get_random_chunk(split)
    rand_indices_of_batch = torch.randint(len(data) - block_size, (batch_size,))
    # print(rand_indices_of_batch)
    input_seq = torch.stack([data[i:i+block_size] for i in rand_indices_of_batch])
    target_seq = torch.stack([data[i+1:i+block_size+1] for i in rand_indices_of_batch])
    input_seq, target_seq = input_seq.to(device), target_seq.to(device)
    return input_seq, target_seq

@torch.no_grad()
def estimate_loss():
    out = {}
    # Puts the model into evaluation mode, which changes how certain layers are treated- turns off dropout (need the whole network)
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            train_data, val_data = get_batch(split)
            logits, loss = model(train_data, val_data)
            losses[k] = loss.item()
        out[split] = losses.mean()
    # Puts the model into training mode, which changes how certain layers are treated- allows the use of dropout
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        # registers the masking in the model's state
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_tensor):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        Batch, Time, Channels = input_tensor.shape
        k = self.key(input_tensor) # (B, T, hs)
        q = self.query(input_tensor) # (B, T, hs)
        # compute attention scores ("affinities") wie = weight
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T) - attention matrix with each token
        wei = wei.masked_fill(self.tril[:Time, :Time] == 0, float('-inf')) # (B, T, T) - exposes one piece of data for each time step
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(input_tensor) # B, T, hs
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out
        

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        # runs our heads in parallel
        self.heads = nn.ModuleList([Head(head_size) for _ in range (num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_tensor):
        out = torch.cat([h(input_tensor) for h in self.heads], dim=-1) # Concatonates along the feature dim (contains the features of each head)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, input_tensor):
        return self.net(input_tensor)

class Block(nn.Module):
    """ Transformer Block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: number of heads we'd like
        super().__init__()
        # number of features that each head will be capturing in our multi head attention
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, input_tensor):
        self_attention = self.sa(input_tensor)
        input_tensor = self.ln1(input_tensor + self_attention)
        feedforward = self.ffwd(input_tensor)
        input_tensor = self.ln2(input_tensor + feedforward)
        return input_tensor

class GPTLanguageModel(nn.Module):
    # creating an embedding table to predict the next character (aa, ab, ac, ad)
    def __init__(self, vocab_size):
        super().__init__()
        # Turns each token (word/character ID) into a learned vector of size n_embd
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # Adds information about position in the sequence (e.g. first word vs. tenth word).
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # Will make 4 Decoder Blocks sequentially
        self.blocks = nn.Sequential(*[Block(n_embd, n_head = n_head) for _ in range (n_layer)])
        # final layer normalization
        self.ln_f = nn.LayerNorm(n_embd)
        # transforms into something that softmax can use
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # initializes our weights around certain standard deviations
        self.apply(self._init_weights)

    # used in practice to ensure our weights are initialized correctly and our training converges better
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
        
    # maps input token IDs to their corresponding learned embeddings (logits)
    def forward(self, index, targets=None):
        Batch, Time = index.shape
        
        # idx and targets are both (Batch, Time) tensor of integers
        tok_emb = self.token_embedding_table(index) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(Time, device = device)) # (T,C)
        hidden_representation = tok_emb + pos_emb # (B,T,C)- Broadcasting semantics explain how + works for different tensors
        hidden_representation = self.blocks(hidden_representation) # (B,T,C)
        hidden_representation = self.ln_f(hidden_representation) # (B,T,C)
        logits = self.lm_head(hidden_representation) # (B,T,vocab_size)
        
        if targets is None:
            loss = None
        else:
            Batch, Time, Channels = logits.shape
            logits = logits.view(Batch * Time, Channels)
            targets = targets.view(Batch * Time)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    # generates new tokens
    def generate(self, index, max_new_tokens):
        # index is (Batch, Time) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the prediction
            logits, loss = self.forward(index)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (Batch, Channels)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim = -1) # (Batch, Channels)
            # sample from the distribution
            index_next = torch.multinomial(probs, num_samples=1) # (Batch, 1)
            # append sampled index to the running sequence
            index = torch.cat((index, index_next), dim=1) # (Batch, Time + 1)
        return index

# creating the model
model = GPTLanguageModel(vocab_size)
#print('loading model parameters...')
#model.load_state_dict(torch.load('model-01.pth', map_location='cpu'))
model.to(device)

#print("Loading model on CPU...")
#model = torch.load("model-01.pkl", map_location='cpu', weights_only=False)  # Load safely
#model = model.to(device)  # Now move to GPU if needed
#print("Model loaded successfully.")

cuda_model = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

# iter - iteration
for iter in range(max_iters):
    if iter % eval_iters == 0:
        # evaluates the loss at every eval iter
        losses = estimate_loss()
        print(f"step: {iter}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}")
    # sample a batch of data
    input_batch, target_batch = get_batch('train')

    # evaluate the loss
    logits, loss = model.forward(input_batch, target_batch)
    # ensures they do not add over time- previous gradients do not affect the current one
    optimizer.zero_grad(set_to_none = True) # most efficient
    loss.backward()
    optimizer.step()
print(loss.item())

#with open('model-01.pkl', 'wb') as f:
#    pickle.dump(model, f)
#print('model saved')
torch.save(model.state_dict(), 'model-01.pth')