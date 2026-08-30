# run with python subwordTraining.py -bs 32 (batch size)

import mmap
import random

import torch
import torch.nn as nn
from torch.nn import functional as F

block_size = 32
batch_size = 128
max_iters = 500
learning_rate = 3e-4
eval_iters = 50
# number of embedding dimensions
n_embd = 384
# number of decoder blocks
n_layer = 1
# number of heads
n_head = 1
# makes a certain percentage of neurons drop out to prevent overfitting
dropout = 0.2
# number of characters from the text to train the model on
chars_from_text = 100_000
# number of total encodings, including the 256 utf-8 encodings
vocab_size = 1024
# number of raw bytes to read at a time from the file (different for subword encoding from character-level encoding)
chunk_size = 4096
# contains the array of the byte pairs and what tokenID it is encoded as in the form [[(pair1,pair2),tokenID],...]
encoding_array = []

device = "cuda" if torch.cuda.is_available() else "cpu"


# encodes the text into a utf-8 encoding before byte pair encoding
def encode_bytes(text):
    return list(text.encode("utf-8"))


# finds the most common pair from an encoded chunk of text
def create_pairs(encoded_chunk):
    pairs = {}
    for number in range(len(encoded_chunk) - 1):
        current_pair = (encoded_chunk[number], encoded_chunk[number + 1])
        if current_pair in pairs:
            pairs[current_pair] += 1
        else:
            pairs[current_pair] = 1
    if len(pairs) == 0:
        return None
    most_common_pair = max(pairs, key=pairs.get)
    return most_common_pair


# merges two pairs in an encoded chunk into a new token
def merge_pair(encoded_chunk, pair_to_merge, new_token):
    new_encoded_chunk = []
    number = 0
    while number < len(encoded_chunk):
        if number < len(encoded_chunk) - 1:
            currentPair = (encoded_chunk[number], encoded_chunk[number + 1])
            if currentPair == pair_to_merge:
                new_encoded_chunk.append(new_token)
                number += 2
                continue
        new_encoded_chunk.append(encoded_chunk[number])
        number += 1
    return new_encoded_chunk


# learns the most common byte pairs present in the text
def train_tokenizer(chunk):
    tokenID = 256
    encoded_chunk = encode_bytes(chunk)
    while tokenID < vocab_size:
        most_common_pair = create_pairs(encoded_chunk)
        if most_common_pair is None:
            break
        encoding_array.append([most_common_pair, tokenID])
        encoded_chunk = merge_pair(encoded_chunk, most_common_pair, tokenID)
        tokenID += 1
    return encoded_chunk


with open("train_split.txt", "r", encoding="utf-8") as f:
    tokenizer_training_text = f.read(chars_from_text)

# ensures encoding array is empty
encoding_array = []

print("Training tokenizer...")
train_tokenizer(tokenizer_training_text)
print(f"Learned {len(encoding_array)} BPE merges")


# encodes the text using the most common byte pairs (Byte Pair Encoding)
def encode(text):
    encoded_text = encode_bytes(text)
    for encoding in encoding_array:
        pair_to_merge = encoding[0]
        tokenID = encoding[1]
        encoded_text = merge_pair(encoded_text, pair_to_merge, tokenID)
    return encoded_text


# decodes the token Ids using the most common byte pairs
def decode(token_ids):
    def expand_token(token):
        # raw byte: nothing more to expand
        if token < 256:
            return [token]
        # find the pair that created this BPE token
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
    return bytes(decoded_bytes).decode("utf-8", errors="replace")


# memory map for using small snippets of text from a single file of any size
def get_random_chunk(split):
    filename = "train_split.txt" if split == "train" else "val_split.txt"
    with open(filename, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # Determine the file size and a random position to start reading
            file_size = len(mm)
            start_pos = random.randint(0, (file_size) - chunk_size)

            # Seek to the random position and read the block of text
            mm.seek(start_pos)
            block = mm.read(chunk_size)

            # Decode the block to a string, ignoring any invalid byte sequences
            decoded_block = block.decode("utf-8", errors="ignore").replace("\r", "")

            # Train and test splits
            data = torch.tensor(encode(decoded_block), dtype=torch.long)
    return data


# returns a batch of input-target sequences from either the training or validation set
def get_batch(split):
    data = get_random_chunk(split)
    rand_indices_of_batch = torch.randint(len(data) - block_size, (batch_size,))
    # print(rand_indices_of_batch)
    input_seq = torch.stack([data[i : i + block_size] for i in rand_indices_of_batch])
    target_seq = torch.stack(
        [data[i + 1 : i + block_size + 1] for i in rand_indices_of_batch]
    )
    input_seq, target_seq = input_seq.to(device), target_seq.to(device)
    return input_seq, target_seq


@torch.no_grad()
def estimate_loss():
    out = {}
    # Puts the model into evaluation mode, which changes how certain layers are treated- turns off dropout (need the whole network)
    model.eval()
    for split in ["train", "val"]:
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
    """one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # registers the masking in the model's state
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_tensor):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        Batch, Time, Channels = input_tensor.shape
        k = self.key(input_tensor)  # (B, T, hs)
        q = self.query(input_tensor)  # (B, T, hs)
        # compute attention scores ("affinities") wie = weight
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T) - attention matrix with each token
        wei = wei.masked_fill(
            self.tril[:Time, :Time] == 0, float("-inf")
        )  # (B, T, T) - exposes one piece of data for each time step
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(input_tensor)  # B, T, hs
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        # runs our heads in parallel
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_tensor):
        out = torch.cat(
            [h(input_tensor) for h in self.heads], dim=-1
        )  # Concatonates along the feature dim (contains the features of each head)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

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
    """Transformer Block: communication followed by computation"""

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
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        # final layer normalization
        self.ln_f = nn.LayerNorm(n_embd)
        # transforms into something that softmax can use
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # initializes our weights around certain standard deviations
        self.apply(self._init_weights)

    # used in practice to ensure our weights are initialized correctly and our training converges better
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # maps input token IDs to their corresponding learned embeddings (logits)
    def forward(self, index, targets=None):
        Batch, Time = index.shape

        # idx and targets are both (Batch, Time) tensor of integers
        tok_emb = self.token_embedding_table(index)  # (B,T,C)
        pos_emb = self.position_embedding_table(
            torch.arange(Time, device=device)
        )  # (T,C)
        hidden_representation = (
            tok_emb + pos_emb
        )  # (B,T,C)- Broadcasting semantics explain how + works for different tensors
        hidden_representation = self.blocks(hidden_representation)  # (B,T,C)
        hidden_representation = self.ln_f(hidden_representation)  # (B,T,C)
        logits = self.lm_head(hidden_representation)  # (B,T,vocab_size)

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
            logits = logits[:, -1, :]  # becomes (Batch, Channels)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (Batch, Channels)
            # sample from the distribution
            index_next = torch.multinomial(probs, num_samples=1)  # (Batch, 1)
            # append sampled index to the running sequence
            index = torch.cat((index, index_next), dim=1)  # (Batch, Time + 1)
        return index


# creating the model
model = GPTLanguageModel(vocab_size)
model.to(device)
cuda_model = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# iter - iteration
for iter in range(max_iters):
    if iter % eval_iters == 0:
        # evaluates the loss at every eval iter
        losses = estimate_loss()
        print(
            f"step: {iter}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}"
        )
    # sample a batch of data
    input_batch, target_batch = get_batch("train")

    # evaluate the loss
    logits, loss = model.forward(input_batch, target_batch)
    # ensures they do not add over time- previous gradients do not affect the current one
    optimizer.zero_grad(set_to_none=True)  # most efficient
    loss.backward()
    optimizer.step()
print(loss.item())

torch.save(model.state_dict(), "subword-model.pth")

torch.save(
    {"encoding_array": encoding_array, "vocab_size": vocab_size},
    "subword-tokenizer.pth",
)

print("Model and tokenizer saved.")
