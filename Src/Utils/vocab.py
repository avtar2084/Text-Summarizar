import nltk
from collections import Counter
from nltk.tokenize import word_tokenize
import json
import re

nltk.download("punkt_tab")

PAD = "<PAD>"
UNK = "<UNK>"
SOS = "<SOS>"
EOS = "<EOS>"

# PAD_IDX = 0
# UNK_IDX = 3
# SOS_IDX = 1
# EOS_IDX = 2



def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


def build_vocab(texts, max_vocab_size=10000):
    all_text = " ".join(texts)
    tokens = tokenize(all_text)
    counter = Counter(tokens)
    most_common = counter.most_common(max_vocab_size - 4)

    vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
    for idx, (word, _) in enumerate(most_common, start=4):
        vocab[word] = idx

    return vocab

#
def text_to_indices(text, vocab, max_len=500):
    tokens = word_tokenize(text.lower())
    indices = [vocab.get(w, vocab[UNK]) for w in tokens]
    return indices[:max_len]

def save_vocab(vocab, file_path):
    """Save vocab dictionary to JSON file"""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)
    print(f"✅ Vocab saved to {file_path}")



def load_vocab(file_path):
    """Load vocab dictionary from JSON file"""
    with open(file_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    print(f"✅ Vocab loaded from {file_path}")
    return vocab
