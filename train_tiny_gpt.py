"""
Train a small causal Transformer ("TinyGPT") on Wikitext-2 (or any plain-text files).

Notes:
- This repo includes a `.venv/` with torch; on macOS prefer running with `.venv/bin/python`.
- By default, this script will use `wikitext-2/train.txt` and (if present) `wikitext-2/validation.txt`.
- The saved checkpoint includes both weights and the model config so inference doesn't have to guess.
"""

import argparse
import math
import json
import time
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


# ----------------------------
# Config (tweak freely)
# ----------------------------
EMBED_DIM   = 128
NUM_HEADS   = 4
NUM_LAYERS  = 6
SEQ_LEN     = 128
BATCH_SIZE  = 32
EPOCHS      = 1
LR          = 5e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP   = 1.0
DROPOUT     = 0.1
LIMIT_LINES = None  # how many non-empty lines from wikitext-2/train.txt to load
DATA_PATH   = "wikitext-2/train.txt"
VAL_PATH    = "wikitext-2/validation.txt"
TEST_PATH   = "wikitext-2/test.txt"
MODEL_OUT   = "tiny_gpt.pth"
TOKENIZER_OUT = "tiny_gpt_tokenizer.json"

SEED = 42
MAX_VOCAB = 50000  # includes special tokens; set None to disable cap
MIN_FREQ = 2       # drop tokens that appear fewer than this many times
WARMUP_RATIO = 0.05
LOG_INTERVAL = 100


def pick_device(choice: str) -> torch.device:
    if choice == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if choice == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if choice == "cpu":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int):
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _try_set_matmul_precision():
    # Speeds up attention-heavy models on many backends without changing code.
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


# ----------------------------
# Tokenizers
# ----------------------------
class SimpleTokenizer:
    def __init__(
        self,
        lower: bool = True,
        unk_token: str = "<unk>",
        eol_token: str = "<eol>",
        max_vocab: Optional[int] = MAX_VOCAB,
        min_freq: int = MIN_FREQ,
    ):
        self.lower = lower
        self.unk_token = unk_token
        self.eol_token = eol_token
        self.max_vocab = max_vocab
        self.min_freq = min_freq
        self.vocab = {}
        self.inv_vocab = {}

    def _normalize(self, text):
        return text.lower() if self.lower else text

    def fit(self, lines):
        # Build vocabulary from lines (whitespace-split), with optional min-freq and max-vocab.
        counts: Dict[str, int] = {}
        for line in lines:
            line = self._normalize(line)
            for tok in line.split():
                counts[tok] = counts.get(tok, 0) + 1

        # Always reserve special tokens at fixed ids so older checkpoints stay predictable.
        vocab: Dict[str, int] = {self.unk_token: 0, self.eol_token: 1}
        items = [(t, c) for t, c in counts.items() if c >= self.min_freq]
        items.sort(key=lambda x: (-x[1], x[0]))  # freq desc, token asc

        cap = None if self.max_vocab is None else max(self.max_vocab - len(vocab), 0)
        for tok, _ in items[:cap]:
            if tok in vocab:
                continue
            vocab[tok] = len(vocab)

        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

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
                    "type": "word",
                    "lower": self.lower,
                    "unk_token": self.unk_token,
                    "eol_token": self.eol_token,
                    "max_vocab": self.max_vocab,
                    "min_freq": self.min_freq,
                    "vocab": self.vocab,
                },
                f,
                ensure_ascii=False,
            )

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls.from_dict(data)
        return tok

    @classmethod
    def from_dict(cls, data: dict):
        tok = cls(
            lower=bool(data.get("lower", True)),
            unk_token=str(data.get("unk_token", "<unk>")),
            eol_token=str(data.get("eol_token", "<eol>")),
            max_vocab=data.get("max_vocab", None),
            min_freq=int(data.get("min_freq", 1)),
        )
        vocab = data.get("vocab", None) or {}
        tok.vocab = {str(k): int(v) for k, v in vocab.items()}
        tok.inv_vocab = {v: k for k, v in tok.vocab.items()}
        return tok


class ByteTokenizer:
    """
    Byte-level tokenizer with reserved special tokens.

    This removes OOV issues and preserves punctuation/spacing exactly.
    """

    TYPE = "byte"

    def __init__(
        self,
        lower: bool = False,
        unk_token: str = "<unk>",
        eol_token: str = "<eol>",
    ):
        self.lower = lower
        self.unk_token = unk_token
        self.eol_token = eol_token

        self.vocab: Dict[str, int] = {
            self.unk_token: 0,
            self.eol_token: 1,
        }
        for b in range(256):
            self.vocab[self._byte_token(b)] = 2 + b

        self._unk_id = self.vocab[self.unk_token]
        self._eol_id = self.vocab[self.eol_token]
        self._special_re = re.compile(f"({re.escape(self.unk_token)}|{re.escape(self.eol_token)})")

    def _normalize(self, text: str) -> str:
        return text.lower() if self.lower else text

    @staticmethod
    def _byte_token(b: int) -> str:
        return f"<0x{b:02x}>"

    def fit(self, lines: List[str]):
        # no-op (deterministic vocab)
        return

    def encode(self, text: str) -> List[int]:
        text = self._normalize(text)
        ids: List[int] = []
        parts = self._special_re.split(text)
        for part in parts:
            if not part:
                continue
            if part == self.unk_token:
                ids.append(self._unk_id)
                continue
            if part == self.eol_token:
                ids.append(self._eol_id)
                continue
            bs = part.encode("utf-8", errors="replace")
            ids.extend([2 + b for b in bs])
        return ids

    def decode(self, ids: List[int]) -> str:
        out: List[str] = []
        buf = bytearray()
        for i in ids:
            if i == self._eol_id:
                if buf:
                    out.append(buf.decode("utf-8", errors="replace"))
                    buf.clear()
                out.append("\n")
                continue
            if i == self._unk_id:
                if buf:
                    out.append(buf.decode("utf-8", errors="replace"))
                    buf.clear()
                out.append(self.unk_token)
                continue
            b = i - 2
            if 0 <= b <= 255:
                buf.append(b)
            else:
                if buf:
                    out.append(buf.decode("utf-8", errors="replace"))
                    buf.clear()
                out.append(self.unk_token)
        if buf:
            out.append(buf.decode("utf-8", errors="replace"))
        return "".join(out)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "type": self.TYPE,
                    "lower": self.lower,
                    "unk_token": self.unk_token,
                    "eol_token": self.eol_token,
                },
                f,
                ensure_ascii=False,
            )

    @classmethod
    def from_dict(cls, data: dict) -> "ByteTokenizer":
        return cls(
            lower=bool(data.get("lower", False)),
            unk_token=str(data.get("unk_token", "<unk>")),
            eol_token=str(data.get("eol_token", "<eol>")),
        )

    @classmethod
    def load(cls, path: str) -> "ByteTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def load_tokenizer(path: str):
    """
    Backward-compatible tokenizer loader.

    - Old files (no `type`) are treated as `SimpleTokenizer`.
    - New files include `type` ("word" or "byte").
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tok_type = str(data.get("type", "word")).lower()
    if tok_type == "byte":
        return ByteTokenizer.from_dict(data)

    # default: word-level
    return SimpleTokenizer.from_dict(data)


def default_suppress_ids(tokenizer, allow_nonprintable_bytes: bool = False) -> List[int]:
    """
    Keep early byte-level generations readable.

    A weak byte model can sample arbitrary UTF-8 continuation/control bytes, which
    decode as replacement characters. For English smoke tests, suppress those by
    default while still allowing normal printable ASCII and line breaks.
    """
    if allow_nonprintable_bytes or not isinstance(tokenizer, ByteTokenizer):
        ids: List[int] = []
        unk = getattr(tokenizer, "unk_token", None)
        vocab = getattr(tokenizer, "vocab", {})
        if unk is not None and unk in vocab:
            ids.append(int(vocab[unk]))
        return ids
    allowed = {9, 10, 13, *range(32, 127)}
    return [2 + b for b in range(256) if b not in allowed]


# ----------------------------
# Dataset
# ----------------------------
class TextDataset(Dataset):
    def __init__(self, token_ids, seq_len, stride: int = 1):
        self.token_ids = token_ids
        self.seq_len = seq_len
        self.stride = max(int(stride), 1)

    def __len__(self):
        n = len(self.token_ids) - self.seq_len  # matches the old (stride=1) behavior
        if n <= 0:
            return 0
        return ((n - 1) // self.stride) + 1

    def __getitem__(self, idx):
        start = idx * self.stride
        x = self.token_ids[start : start + self.seq_len]
        y = self.token_ids[start + 1 : start + self.seq_len + 1]
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
        # Use an additive mask (0 for allowed, -inf for disallowed) for better backend compatibility.
        m = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        m = m.masked_fill(m == 1, float("-inf")).masked_fill(m == 0, 0.0)
        self.register_buffer("causal_mask", m, persistent=False)

        self.apply(self._init_weights)

        # weight tying
        self.fc.weight = self.token_emb.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

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
    def generate(
        self,
        idx,
        max_new_tokens: int = 64,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = None,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        greedy: bool = False,
        suppress_ids: Optional[List[int]] = None,
    ):
        self.eval()
        suppress_tensor = None
        if suppress_ids:
            suppress_tensor = torch.tensor(suppress_ids, dtype=torch.long, device=idx.device)
        for _ in range(max_new_tokens):
            x_cond = idx[:, -self.seq_len :]
            logits = self.forward(x_cond)[:, -1, :]
            logits = logits / max(temperature, 1e-6)

            if suppress_tensor is not None:
                valid = suppress_tensor[suppress_tensor < logits.size(-1)]
                if valid.numel() > 0:
                    logits[:, valid] = -float("inf")

            if repetition_penalty and repetition_penalty != 1.0:
                # Penalize tokens already present in the context window.
                for b in range(idx.size(0)):
                    prev = idx[b, -self.seq_len :].unique()
                    logits[b, prev] /= repetition_penalty

            if top_k is not None and top_k > 0:
                k = min(top_k, logits.size(-1))
                top_vals, _ = torch.topk(logits, k=k)
                thresh = top_vals[:, -1].unsqueeze(-1)
                logits[logits < thresh] = -float("inf")

            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                sorted_probs = torch.softmax(sorted_logits, dim=-1)
                cumprobs = torch.cumsum(sorted_probs, dim=-1)
                mask = cumprobs > top_p
                mask[..., 0] = False  # always keep at least one token
                sorted_logits[mask] = -float("inf")
                logits = torch.full_like(logits, -float("inf"))
                logits.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)

            if greedy:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


# ----------------------------
# Training / Eval
# ----------------------------
@torch.no_grad()
def evaluate(model, dataloader, criterion, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        n_tok = int(y.numel())
        total_loss += loss.item() * n_tok
        total_tokens += n_tok
    avg = total_loss / max(total_tokens, 1)
    ppl = math.exp(avg) if avg < 20 else float("inf")
    return avg, ppl


def load_text_lines(path: Path, limit_lines: Optional[int] = None) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Couldn't find {path}.")
    lines: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            lines.append(s)
            if limit_lines is not None and len(lines) >= limit_lines:
                break
    return lines


def load_text_lines_many(paths: List[str], limit_lines: Optional[int] = None) -> List[str]:
    lines: List[str] = []
    for path in paths:
        p = Path(path)
        extra = load_text_lines(p, limit_lines=None)
        lines.extend(extra)
        if limit_lines is not None and len(lines) >= limit_lines:
            return lines[:limit_lines]
    return lines


def build_token_stream(tokenizer, lines: List[str]) -> List[int]:
    eol_id = tokenizer.vocab[tokenizer.eol_token]
    all_tokens: List[int] = []
    for line in lines:
        ids = tokenizer.encode(line)
        all_tokens.extend(ids)
        all_tokens.append(eol_id)
    return all_tokens


def load_checkpoint_for_state_dict(obj) -> Dict[str, torch.Tensor]:
    # Backward compatible: accept a raw state_dict or a richer checkpoint dict.
    if isinstance(obj, dict) and "model_state_dict" in obj:
        return obj["model_state_dict"]
    return obj


def get_checkpoint_config(obj) -> Optional[dict]:
    if isinstance(obj, dict) and "config" in obj:
        return obj["config"]
    return None


def main(argv: Optional[List[str]] = None):
    ap = argparse.ArgumentParser(description="Train TinyGPT on text.")
    ap.add_argument("--train-file", default=DATA_PATH, help="Training text file (one line = one document line)")
    ap.add_argument("--val-file", default=VAL_PATH, help="Validation file (optional; falls back to split if missing)")
    ap.add_argument("--test-file", default=TEST_PATH, help="Optional test file to report final perplexity")
    ap.add_argument("--limit-lines", type=int, default=LIMIT_LINES)
    ap.add_argument("--limit-val-lines", type=int, default=None)
    ap.add_argument("--limit-test-lines", type=int, default=None)

    ap.add_argument("--tokenizer-type", choices=["byte", "word"], default="byte")
    ap.add_argument(
        "--tokenizer-extra-file",
        action="append",
        default=[],
        help="Additional text file(s) used only to build the tokenizer vocabulary.",
    )
    ap.add_argument("--embed-dim", type=int, default=EMBED_DIM)
    ap.add_argument("--num-heads", type=int, default=NUM_HEADS)
    ap.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    ap.add_argument("--seq-len", type=int, default=SEQ_LEN)
    ap.add_argument("--dropout", type=float, default=DROPOUT)

    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    ap.add_argument("--grad-clip", type=float, default=GRAD_CLIP)
    ap.add_argument("--warmup-ratio", type=float, default=WARMUP_RATIO)
    ap.add_argument("--log-interval", type=int, default=LOG_INTERVAL)
    ap.add_argument("--stride", type=int, default=0, help="Token stride between sequences (0 => use seq-len)")

    ap.add_argument("--max-vocab", type=int, default=MAX_VOCAB)
    ap.add_argument("--min-freq", type=int, default=MIN_FREQ)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")

    ap.add_argument("--model-out", default=MODEL_OUT)
    ap.add_argument("--tokenizer-out", default=TOKENIZER_OUT)
    ap.add_argument("--save-best-only", action="store_true", help="Only keep best checkpoint by val loss")

    args = ap.parse_args(argv)

    device = pick_device(args.device)
    print(f"Using device: {device}")
    set_seed(args.seed)
    _try_set_matmul_precision()

    # ---------- Load data ----------
    train_path = Path(args.train_file)
    val_path = Path(args.val_file) if args.val_file else None
    test_path = Path(args.test_file) if args.test_file else None

    train_lines = load_text_lines(train_path, limit_lines=args.limit_lines)
    if not train_lines:
        raise SystemExit("No training lines found.")

    # ---------- Tokenize ----------
    if args.tokenizer_type == "word":
        tokenizer = SimpleTokenizer(max_vocab=args.max_vocab, min_freq=args.min_freq)
        tokenizer_lines = list(train_lines)
        if args.tokenizer_extra_file:
            tokenizer_lines.extend(load_text_lines_many(args.tokenizer_extra_file))
        tokenizer.fit(tokenizer_lines)
    else:
        tokenizer = ByteTokenizer()
    train_tokens = build_token_stream(tokenizer, train_lines)
    print(f"Train tokens: {len(train_tokens)} | Vocab size: {len(tokenizer.vocab)}")

    # ---------- Validation tokens ----------
    val_tokens: Optional[List[int]] = None
    if val_path and val_path.exists():
        val_lines = load_text_lines(val_path, limit_lines=args.limit_val_lines)
        val_tokens = build_token_stream(tokenizer, val_lines)
        print(f"Val tokens: {len(val_tokens)} (from {val_path})")

    # ---------- Split / datasets ----------
    seq_len = args.seq_len
    if len(train_tokens) <= seq_len + 1:
        raise ValueError("Not enough tokens for the chosen --seq-len.")

    stride = args.seq_len if (args.stride is None or int(args.stride) <= 0) else int(args.stride)
    if val_tokens is None:
        # fallback: last 5% of the token stream as a quick sanity check
        split_idx = int(0.95 * (len(train_tokens) - seq_len))
        train_ds = TextDataset(train_tokens[:split_idx], seq_len, stride=stride)
        val_ds = TextDataset(train_tokens[split_idx:], seq_len, stride=stride)
        print("[note] validation.txt missing; using a small holdout from train stream.")
    else:
        train_ds = TextDataset(train_tokens, seq_len, stride=stride)
        val_ds = TextDataset(val_tokens, seq_len, stride=stride)

    if len(train_ds) == 0:
        raise ValueError("Training dataset is empty; use more data or a smaller --seq-len.")
    if len(val_ds) == 0:
        raise ValueError("Validation dataset is empty; use more validation data or a smaller --seq-len.")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # ---------- Model ----------
    model = TinyGPT(
        vocab_size=len(tokenizer.vocab),
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        seq_len=args.seq_len,
        p_drop=args.dropout,
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95)
    )
    criterion = nn.CrossEntropyLoss(reduction="mean")

    # LR schedule: warmup + cosine decay (helps stability for small models)
    total_steps = max(args.epochs * len(train_dl), 1)
    warmup_steps = int(args.warmup_ratio * total_steps)

    def lr_mult(step: int):
        if step < warmup_steps:
            return float(step + 1) / max(warmup_steps, 1)
        progress = float(step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_mult)

    # ---------- Train ----------
    start = time.time()
    best_val = float("inf")
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0

        for i, (x, y) in enumerate(train_dl):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1

            total_loss += loss.item()
            n_batches += 1

            if args.log_interval and (i + 1) % args.log_interval == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch} | Batch {i+1}/{len(train_dl)} | "
                    f"loss {loss.item():.4f} | lr {lr:.2e}"
                )

        train_loss = total_loss / max(n_batches, 1)
        val_loss, val_ppl = evaluate(model, val_dl, criterion, device)
        print(
            f"Epoch {epoch}/{args.epochs} | train_loss {train_loss:.4f} | "
            f"val_loss {val_loss:.4f} | val_ppl {val_ppl:.2f}"
        )

        # quick sample (truncate prompt to model context)
        model.eval()
        prompt = "The history of machine learning"
        with torch.no_grad():
            seed = tokenizer.encode(prompt)[-args.seq_len :]
            prompt_ids = torch.tensor([seed], dtype=torch.long, device=device)
            out = model.generate(
                prompt_ids,
                max_new_tokens=80,
                top_k=40,
                top_p=0.95,
                temperature=0.9,
                repetition_penalty=1.1,
                suppress_ids=default_suppress_ids(tokenizer),
            )[0].tolist()
        print("=== SAMPLE ===")
        print(tokenizer.decode(out))
        print("==============")

        # save checkpoint
        ckpt = {
            "model_state_dict": model.state_dict(),
            "config": {
                "embed_dim": args.embed_dim,
                "num_heads": args.num_heads,
                "num_layers": args.num_layers,
                "seq_len": args.seq_len,
                "dropout": args.dropout,
                "vocab_size": len(tokenizer.vocab),
                "tokenizer_type": args.tokenizer_type,
            },
        }
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
        if (not args.save_best_only) or is_best:
            torch.save(ckpt, args.model_out)
            tokenizer.save(args.tokenizer_out)
            if is_best:
                print(f"Saved best -> {args.model_out} (val_loss {best_val:.4f})")

    dur_min = (time.time() - start) / 60
    print(f"Training complete in {dur_min:.2f} minutes")

    # ---------- Optional test perplexity ----------
    if test_path and test_path.exists():
        test_lines = load_text_lines(test_path, limit_lines=args.limit_test_lines)
        test_tokens = build_token_stream(tokenizer, test_lines)
        test_dl = DataLoader(
            TextDataset(test_tokens, args.seq_len, stride=stride),
            batch_size=args.batch_size,
            shuffle=False,
        )
        test_loss, test_ppl = evaluate(model, test_dl, criterion, device)
        print(f"Test | loss {test_loss:.4f} | ppl {test_ppl:.2f} (from {test_path})")


if __name__ == "__main__":
    main()
