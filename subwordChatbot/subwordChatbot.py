# run with python subwordChatbot.py

import torch
import torch.nn as nn
from torch.nn import functional as F
import mmap
import random
import pickle
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'

block_size = 32
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

tokenizer_data = torch.load(
    "subword-tokenizer.pth",
    map_location="cpu"
)

encoding_array = tokenizer_data["encoding_array"]
vocab_size = tokenizer_data["vocab_size"]

def merge_pair(encoded_chunk, pair_to_merge, new_token):
    new_encoded_chunk = []
    number = 0

    while number < len(encoded_chunk):
        if number < len(encoded_chunk) - 1:
            currentPair = (encoded_chunk[number], encoded_chunk[number + 1])
            if currentPair == pair_to_merge:
                new_encoded_chunk.append(new_token)
                number+=2
                continue
        new_encoded_chunk.append(encoded_chunk[number])
        number+=1
    return new_encoded_chunk

def encode_bytes(text):
    # Encode the text into a list of integers (byte values)
    return list(text.encode('utf-8'))

def encode(text):
    encoded_text = encode_bytes(text)

    for encoding in encoding_array:
        pair_to_merge = encoding[0]
        tokenID = encoding[1]
        encoded_text = merge_pair(encoded_text, pair_to_merge, tokenID)

    return encoded_text

def decode(token_ids):

    def expand_token(token):
        # Raw byte: nothing more to expand
        if token < 256:
            return [token]

        # Find the pair that created this BPE token
        for encoding in encoding_array:
            pair = encoding[0]
            tokenID = encoding[1]

            if token == tokenID:
                left = pair[0]
                right = pair[1]

                return expand_token(left) + expand_token(right)

    decoded_bytes = []

    for token in token_ids:
        decoded_bytes.extend(expand_token(token))

    return bytes(decoded_bytes).decode('utf-8', errors="replace")

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
            # crop idx to the last block_size tokens
            index_cond = index[:, -block_size:]
            # get the predictions
            logits, loss = self.forward(index_cond)
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
print('loading model parameters...')
model.load_state_dict(torch.load("subword-model.pth", map_location='cpu'))
model.to(device)
cuda_model = model.to(device)

while True:
    prompt = input("Prompt:\n")
    context = torch.tensor(encode(prompt), dtype = torch.long, device = device)
    generated_tokens = decode(cuda_model.generate(context.unsqueeze(0), max_new_tokens = 150)[0].tolist())
    print(f'Completion:\n{generated_tokens}')