# tinygpt_train.py
import math
import json
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ----------------------------
# Config (tweak freely)
# ----------------------------
EMBED_DIM   = 64
NUM_HEADS   = 4
NUM_LAYERS  = 4
SEQ_LEN     = 64
BATCH_SIZE  = 32
EPOCHS      = 4
LR          = 3e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP   = 1.0
DROPOUT     = 0.1
LIMIT_LINES = None  # how many non-empty lines from wikitext-2/train.txt to load
DATA_PATH   = "wikitext-2/train.txt"
MODEL_OUT   = "tiny_gpt.pth"
TOKENIZER_OUT = "tiny_gpt_tokenizer.json"

DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


# ----------------------------
# Simple tokenizer (word-level)
# ----------------------------
class SimpleTokenizer:
    def __init__(self, lower=True, unk_token="<unk>", eol_token="<eol>"):
        self.lower = lower
        self.unk_token = unk_token
        self.eol_token = eol_token
        self.vocab = {}
        self.inv_vocab = {}

    def _normalize(self, text):
        return text.lower() if self.lower else text

    def fit(self, lines):
        # Build vocabulary from lines (space-split)
        vocab = {self.unk_token: 0, self.eol_token: 1}
        idx = 2
        for line in lines:
            line = self._normalize(line)
            for tok in line.split():
                if tok not in vocab:
                    vocab[tok] = idx
                    idx += 1
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}

    def encode(self, text):
        text = self._normalize(text)
        return [self.vocab.get(tok, self.vocab[self.unk_token]) for tok in text.split()]

    def decode(self, ids):
        toks = [self.inv_vocab.get(i, self.unk_token) for i in ids]
        # Join with spaces; replace eol token with newline for nicer samples
        return " ".join(toks).replace(f" {self.eol_token} ", "\n").replace(self.eol_token, "\n")

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "lower": self.lower,
                    "unk_token": self.unk_token,
                    "eol_token": self.eol_token,
                    "vocab": self.vocab,
                },
                f,
                ensure_ascii=False,
            )

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(lower=data["lower"], unk_token=data["unk_token"], eol_token=data["eol_token"])
        tok.vocab = data["vocab"]
        tok.inv_vocab = {v: k for k, v in tok.vocab.items()}
        return tok


# ----------------------------
# Dataset
# ----------------------------
class TextDataset(Dataset):
    def __init__(self, token_ids, seq_len):
        self.token_ids = token_ids
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.token_ids) - self.seq_len)

    def __getitem__(self, idx):
        x = self.token_ids[idx : idx + self.seq_len]
        y = self.token_ids[idx + 1 : idx + self.seq_len + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


# ----------------------------
# Model (causal Transformer)
# ----------------------------
class TinyGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, seq_len, p_drop=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb   = nn.Embedding(seq_len, embed_dim)
        self.drop      = nn.Dropout(p_drop)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True,
            dim_feedforward=4 * embed_dim,
            dropout=p_drop,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(embed_dim)
        self.fc   = nn.Linear(embed_dim, vocab_size, bias=False)  # will tie weights
        self.seq_len = seq_len

        # causal mask buffer (True => masked)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask, persistent=False)

        # weight tying
        self.fc.weight = self.token_emb.weight

    def forward(self, x):
        B, T = x.shape
        if T > self.seq_len:
            raise ValueError(f"Sequence length {T} exceeds model max length {self.seq_len}.")
        pos = torch.arange(0, T, device=x.device).unsqueeze(0)  # [1, T]
        h = self.token_emb(x) + self.pos_emb(pos)
        h = self.drop(h)
        h = self.transformer(h, mask=self.causal_mask[:T, :T])
        h = self.ln_f(h)
        logits = self.fc(h)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=64, top_k=50, temperature=1.0):
        self.eval()
        for _ in range(max_new_tokens):
            x_cond = idx[:, -self.seq_len :]
            logits = self.forward(x_cond)[:, -1, :]
            logits = logits / max(temperature, 1e-6)
            if top_k is not None and top_k > 0:
                k = min(top_k, logits.size(-1))
                top_vals, _ = torch.topk(logits, k=k)
                thresh = top_vals[:, -1].unsqueeze(-1)
                logits[logits < thresh] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


# ----------------------------
# Training / Eval
# ----------------------------
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total_loss += loss.item()
            n += 1
    avg = total_loss / max(n, 1)
    ppl = math.exp(avg) if avg < 20 else float("inf")
    return avg, ppl


def main():
    # ---------- Load data ----------
    path = Path(DATA_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"Couldn't find {DATA_PATH}. Download wikitext-2 and place train.txt there."
        )

    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if LIMIT_LINES is not None:
        lines = lines[:LIMIT_LINES]

    # ---------- Tokenize ----------
    tokenizer = SimpleTokenizer()
    tokenizer.fit(lines)

    EOL = tokenizer.vocab[tokenizer.eol_token]
    all_tokens = []
    for line in lines:
        ids = tokenizer.encode(line)
        all_tokens.extend(ids + [EOL])  # add end-of-line token

    print(f"Total tokens: {len(all_tokens)} | Vocab size: {len(tokenizer.vocab)}")

    # ---------- Split ----------
    usable = len(all_tokens) - (SEQ_LEN + 1)
    if usable <= 0:
        raise ValueError("Not enough tokens for the chosen SEQ_LEN.")
    split_idx = int(0.95 * (len(all_tokens) - SEQ_LEN))
    train_ds = TextDataset(all_tokens[:split_idx], SEQ_LEN)
    val_ds   = TextDataset(all_tokens[split_idx:], SEQ_LEN)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # ---------- Model ----------
    model = TinyGPT(
        vocab_size=len(tokenizer.vocab),
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        seq_len=SEQ_LEN,
        p_drop=DROPOUT,
    ).to(DEVICE)

    optimizer = optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95)
    )
    criterion = nn.CrossEntropyLoss()

    # ---------- Train ----------
    start = time.time()
    for epoch in range(EPOCHS):
        model.train()
        total_loss, n_batches = 0.0, 0

        for i, (x, y) in enumerate(train_dl):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if (i + 1) % 100 == 0:
                print(f"Epoch {epoch+1} | Batch {i+1}/{len(train_dl)} | Loss {loss.item():.4f}")

        train_loss = total_loss / max(n_batches, 1)
        val_loss, val_ppl = evaluate(model, val_dl, criterion)
        print(f"Epoch {epoch+1}/{EPOCHS} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | val_ppl {val_ppl:.2f}")

        # quick sample
        model.eval()
        prompt = "The history of machine learning"
        with torch.no_grad():
            prompt_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=DEVICE)
            out = model.generate(prompt_ids, max_new_tokens=80, top_k=40, temperature=0.9)[0].tolist()
        print("=== SAMPLE ===")
        print(tokenizer.decode(out))
        print("==============")

    dur_min = (time.time() - start) / 60
    print(f"Training complete in {dur_min:.2f} minutes")

    # ---------- Save ----------
    torch.save(model.state_dict(), MODEL_OUT)
    tokenizer.save(TOKENIZER_OUT)
    print(f"Saved model -> {MODEL_OUT}")
    print(f"Saved tokenizer -> {TOKENIZER_OUT}")


if __name__ == "__main__":
    main()
