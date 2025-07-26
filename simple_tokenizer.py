import re
from collections import Counter

class SimpleTokenizer:
    def __init__(self, lower: bool = True):
        self.lower = lower
        self.vocab = None
        self.word2idx = None
        self.idx2word = None

    def fit(self, texts):
        """Build vocabulary from a list of texts."""
        words = []
        for text in texts:
            if self.lower:
                text = text.lower()
            # Split on whitespace and punctuation
            words.extend(re.findall(r"\b\w+\b", text))
        self.vocab = sorted(set(words))
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}

    def encode(self, text):
        if self.lower:
            text = text.lower()
        words = re.findall(r"\b\w+\b", text)
        return [self.word2idx[w] for w in words if w in self.word2idx]

    def decode(self, indices):
        return ' '.join(self.idx2word[i] for i in indices)

if __name__ == "__main__":
    # Example usage: fit on wikitext-2/train.txt
    with open("wikitext-2/train.txt", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    tokenizer = SimpleTokenizer()
    tokenizer.fit(lines)
    print(f"Vocab size: {len(tokenizer.vocab)}")
    # Encode and decode a sample
    sample = lines[0]
    encoded = tokenizer.encode(sample)
    print(f"Sample: {sample}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {tokenizer.decode(encoded)}")
