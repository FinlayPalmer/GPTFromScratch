# GPT From Scratch

A GPT-style language model implemented from scratch with PyTorch.

The initial model was built by following this [GPT from Scratch tutorial](https://www.youtube.com/watch?v=UU1WVnMk4E8) and was then extended with a **Byte Pair Encoding (BPE) subword tokenizer** to explore a more realistic tokenization approach.

## Components

### GPT Model

The core model implements the main components of a GPT-style transformer:

* Token and positional embeddings
* Self-attention
* Multi-head attention
* Feed-forward networks
* Transformer blocks
* Next-token prediction
* Autoregressive text generation

### Tokenization

Two tokenization approaches are explored:

**Character-level tokenizer**
The original implementation represents text as individual characters. This keeps the vocabulary small and makes the tokenization process easy to understand.

**BPE subword tokenizer**
The extended implementation uses **Byte Pair Encoding (BPE)** to learn common subword units from the training data. This produces more meaningful tokens and shorter input sequences than character-level tokenization.

### Repository Structure

```text
GPTFromScratch/
├── chatbot/          # Original GPT implementation
├── subwordChatbot/   # GPT using BPE subword tokenization
└── prototypes/       # Experiments and prototype implementations
```

## Purpose

This repository is an educational project for understanding the components behind GPT-style language models, particularly **transformer architecture, attention, tokenization, training, and text generation**.

## Credits

The original implementation was based on the following tutorial:

https://www.youtube.com/watch?v=UU1WVnMk4E8

The project was then independently extended to include a **BPE subword tokenizer and subword-based GPT implementation**.
