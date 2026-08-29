from datasets import load_dataset

print("Loading OpenWebText...")

dataset = load_dataset(
    "Skylion007/openwebtext",
    split="train",
    streaming=True
)

documents = []

for i, example in enumerate(dataset):
    documents.append(example["text"])

    if i + 1 >= 10000:
        break

text = "\n\n".join(documents)

split_index = int(len(text) * 0.9)

train_text = text[:split_index]
val_text = text[split_index:]

with open("train_split.txt", "w", encoding="utf-8") as f:
    f.write(train_text)

with open("val_split.txt", "w", encoding="utf-8") as f:
    f.write(val_text)

chars = sorted(set(text))

with open("vocab.txt", "w", encoding="utf-8") as f:
    f.write("".join(chars))

print("Done!")
print(f"Characters: {len(text):,}")
print(f"Vocabulary size: {len(chars):,}")