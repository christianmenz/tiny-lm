# tinygpt_finetune.py
"""
Fine-tune TinyGPT on role-tagged OpenAssistant text.

Input file format (case-insensitive):
question
<user text line(s)>
answer
<assistant text line(s)>

(optional blank line between pairs)

Usage examples:
  # basic finetune
  python tinygpt_finetune.py --data openassist_pairs_en.txt --epochs 2 --lr 5e-5 \
    --sample-prompt "question <eol> how do i install python on macos <eol> answer <eol>"

  # freeze transformer, tune embeddings + output head (tied)
  python tinygpt_finetune.py --data openassist_pairs_en.txt --head-only

  # freeze embeddings (and output head via tying), tune transformer only
  python tinygpt_finetune.py --data openassist_pairs_en.txt --freeze-embeddings
"""

import argparse
import math
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# --- reuse from your training script (keeps things DRY) ---
from train_tiny_gpt import (
    TinyGPT,
    load_tokenizer,
    TextDataset,
    get_checkpoint_config,
    load_checkpoint_for_state_dict,
    EMBED_DIM as TRAIN_EMBED_DIM,
    NUM_HEADS as TRAIN_NUM_HEADS,
    NUM_LAYERS as TRAIN_NUM_LAYERS,
    SEQ_LEN as TRAIN_SEQ_LEN,
    DROPOUT as TRAIN_DROPOUT,
)


# ----------------------------
# Helpers
# ----------------------------
def pick_device(choice: str):
    if choice == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if choice == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if choice == "cpu":
        return torch.device("cpu")
    # auto
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, n = 0.0, 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total_loss += loss.item()
        n += 1
    avg = total_loss / max(n, 1)
    ppl = math.exp(avg) if avg < 20 else float("inf")
    return avg, ppl


def load_lines(paths, limit=None) -> List[str]:
    if isinstance(paths, (list, tuple)):
        files = [Path(p) for p in paths]
    else:
        files = [Path(paths)]
    lines: List[str] = []
    for p in files:
        if not p.exists():
            raise FileNotFoundError(f"Data file not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                lines.append(line.rstrip("\n"))
                if limit and len(lines) >= limit:
                    return lines
    return lines


def make_token_stream(
    tokenizer,
    lines: List[str],
    user_tag: str = "question",
    assistant_tag: str = "answer",
) -> Tuple[List[int], int, int]:
    """
    Encode role-tagged blocks into a flat token stream.

    For every line, we append <eol>. Additionally, we append an extra <eol>
    when we finish an assistant block to separate turns.

    Supports single- or multi-line content for each role. A new role tag or a
    blank line ends the current content block.
    """
    eol_id = tokenizer.vocab[tokenizer.eol_token]
    unk_id = tokenizer.vocab[tokenizer.unk_token]
    UT = user_tag.lower().strip()
    AT = assistant_tag.lower().strip()

    tokens: List[int] = []
    unk_count = 0
    total = 0

    current_role = None  # None | "user" | "assistant"

    def encode_and_append(text: str):
        nonlocal unk_count, total, tokens
        ids = tokenizer.encode(text)
        unk_count += sum(1 for t in ids if t == unk_id)
        total += len(ids)
        tokens.extend(ids)
        tokens.append(eol_id)

    i = 0
    N = len(lines)
    while i < N:
        raw = lines[i].strip()
        i += 1
        if not raw:
            # blank line ends any current block
            if current_role == "assistant":
                tokens.append(eol_id)  # extra separator
            current_role = None
            continue

        s_lower = raw.lower()

        if s_lower == UT:
            current_role = "user"
            encode_and_append(UT)  # include the tag token itself
            continue

        if s_lower == AT:
            # finishing any previous role; if it was assistant already, separate
            if current_role == "assistant":
                tokens.append(eol_id)
            current_role = "assistant"
            encode_and_append(AT)
            continue

        # content line for whichever role is active
        encode_and_append(raw)

        # If next line starts a new role or we hit end/blank, we'll detect in the next loop
        # and add the extra <eol> after assistant block ends.
        # Handle end-of-file assistant block: add separator if file ends while in assistant role
        if i == N:
            if current_role == "assistant":
                tokens.append(eol_id)

    return tokens, unk_count, total


def warn_if_oov_tags(tokenizer, user_tag: str, assistant_tag: str):
    unk_id = tokenizer.vocab[tokenizer.unk_token]
    ut_ids = tokenizer.encode(user_tag.lower())
    at_ids = tokenizer.encode(assistant_tag.lower())
    ut_oov = any(t == unk_id for t in ut_ids)
    at_oov = any(t == unk_id for t in at_ids)
    if ut_oov or at_oov:
        print(
            f"[warning] some role-tag tokens are <unk> in the base vocab. "
            f"Consider using common words (e.g., 'question'/'answer'). "
            f"(user_tag_oov={ut_oov}, assistant_tag_oov={at_oov})"
        )


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Fine-tune TinyGPT on role-tagged text (imports from tinygpt_train.py)")
    ap.add_argument("--data", required=True, nargs="+", help="Path(s) to finetune text file(s)")
    ap.add_argument("--model", default="tiny_gpt.pth", help="Pretrained .pth to start from")
    ap.add_argument("--tokenizer", default="tiny_gpt_tokenizer.json", help="Tokenizer json to reuse")
    ap.add_argument("--output", default="finetuned_tiny_gpt.pth", help="Where to save finetuned weights")

    # reuse training defaults but allow override
    ap.add_argument("--embed-dim", type=int, default=TRAIN_EMBED_DIM)
    ap.add_argument("--num-heads", type=int, default=TRAIN_NUM_HEADS)
    ap.add_argument("--num-layers", type=int, default=TRAIN_NUM_LAYERS)
    ap.add_argument("--seq-len", type=int, default=TRAIN_SEQ_LEN)
    ap.add_argument("--dropout", type=float, default=TRAIN_DROPOUT)

    # finetune hyperparams
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--limit-lines", type=int, default=None, help="Cap total number of lines read across all files")
    ap.add_argument("--train-split", type=float, default=0.95)

    # role tags (must match how your file is stored)
    ap.add_argument("--user-tag", default="question")
    ap.add_argument("--assistant-tag", default="answer")

    # freezing options
    ap.add_argument("--freeze-embeddings", action="store_true",
                    help="Freeze token + position embeddings (also freezes output head due to weight tying).")
    ap.add_argument("--head-only", action="store_true",
                    help="Freeze transformer; update embeddings/output head only.")

    # sampling during training
    ap.add_argument("--sample-prompt", type=str, default=None,
                    help="If set, generate a sample after each epoch. Use role/eol pattern, e.g. "
                         "'question <eol> how do i install python <eol> answer <eol>'")
    ap.add_argument("--sample-tokens", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=40)

    ap.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")

    args = ap.parse_args()
    device = pick_device(args.device)
    print(f"Using device: {device}")

    # --- load pretrained checkpoint early so seq_len/model dims match everywhere ---
    if not Path(args.model).exists():
        raise FileNotFoundError(f"Pretrained model not found: {args.model}")
    ckpt = torch.load(args.model, map_location=device)
    ckpt_cfg = get_checkpoint_config(ckpt)
    if ckpt_cfg:
        for k, v in {
            "embed_dim": args.embed_dim,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "seq_len": args.seq_len,
        }.items():
            if int(ckpt_cfg[k]) != int(v):
                print(f"[note] overriding --{k.replace('_','-')}={v} with checkpoint {k}={ckpt_cfg[k]}")
        args.embed_dim = int(ckpt_cfg["embed_dim"])
        args.num_heads = int(ckpt_cfg["num_heads"])
        args.num_layers = int(ckpt_cfg["num_layers"])
        args.seq_len = int(ckpt_cfg["seq_len"])
        args.dropout = float(ckpt_cfg["dropout"])

    # --- load tokenizer (reused; do NOT refit) ---
    tok = load_tokenizer(args.tokenizer)
    warn_if_oov_tags(tok, args.user_tag, args.assistant_tag)

    # --- data ---
    raw_lines = load_lines(args.data, limit=args.limit_lines)
    if not raw_lines:
        raise SystemExit("No lines read from --data files.")

    all_tokens, unk_count, total = make_token_stream(
        tokenizer=tok,
        lines=raw_lines,
        user_tag=args.user_tag,
        assistant_tag=args.assistant_tag,
    )
    unk_pct = 100.0 * (unk_count / max(total, 1))
    print(f"Finetune tokens: {len(all_tokens)} | OOV mapped to <unk>: {unk_count}/{total} ({unk_pct:.2f}%)")

    # --- split ---
    if len(all_tokens) <= args.seq_len + 1:
        raise ValueError("Not enough tokens for the chosen --seq-len.")
    split_idx = int(args.train_split * (len(all_tokens) - args.seq_len))
    train_ds = TextDataset(all_tokens[:split_idx], args.seq_len)
    val_ds   = TextDataset(all_tokens[split_idx:], args.seq_len)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dl   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # --- model ---
    model = TinyGPT(
        vocab_size=len(tok.vocab),
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        seq_len=args.seq_len,
        p_drop=args.dropout,
    ).to(device)

    # load pretrained weights (checkpoint dict or raw state_dict)
    state = load_checkpoint_for_state_dict(ckpt)
    model.load_state_dict(state, strict=True)

    # optional freezing
    if args.freeze_embeddings:
        for p in model.token_emb.parameters():
            p.requires_grad = False
        for p in model.pos_emb.parameters():
            p.requires_grad = False
        # fc.weight is tied to token_emb.weight, so it's effectively frozen too.

    if args.head_only:
        for p in model.transformer.parameters():
            p.requires_grad = False
        # embeddings (and thus output head) stay trainable

    # optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise SystemExit("All parameters are frozen; nothing to train. Remove freezing flags.")
    optimizer = optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    criterion = nn.CrossEntropyLoss()

    # --- train ---
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n_batches = 0.0, 0
        for i, (x, y) in enumerate(train_dl, start=1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                nn.utils.clip_grad_norm_(params, args.grad_clip)
            optimizer.step()

            running += loss.item()
            n_batches += 1

            if i % 100 == 0:
                print(f"Epoch {epoch} | Batch {i}/{len(train_dl)} | loss {loss.item():.4f}")

        train_loss = running / max(n_batches, 1)
        val_loss, val_ppl = evaluate(model, val_dl, criterion, device)
        print(f"Epoch {epoch}/{args.epochs} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | val_ppl {val_ppl:.2f}")

        # sample
        if args.sample_prompt:
            model.eval()
            with torch.no_grad():
                seed = torch.tensor([tok.encode(args.sample_prompt)], dtype=torch.long, device=device)
                out = model.generate(
                    seed,
                    max_new_tokens=args.sample_tokens,
                    top_k=args.top_k,
                    temperature=args.temperature,
                )[0].tolist()
            print("=== SAMPLE ===")
            print(tok.decode(out))
            print("==============")

    dur_min = (time.time() - start) / 60
    print(f"Fine-tuning complete in {dur_min:.2f} minutes")

    # --- save ---
    ckpt_out = {
        "model_state_dict": model.state_dict(),
        "config": {
            "embed_dim": args.embed_dim,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "seq_len": args.seq_len,
            "dropout": args.dropout,
            "vocab_size": len(tok.vocab),
        },
    }
    torch.save(ckpt_out, args.output)
    print(f"Saved finetuned weights -> {args.output}")
    print("Note: reuse the SAME tokenizer JSON you trained with.")
    print("Prompt pattern at inference:")
    print(f"  \"{args.user_tag} <eol> your question here <eol> {args.assistant_tag} <eol>\"")

if __name__ == "__main__":
    main()
