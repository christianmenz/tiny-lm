"""
Generate text from a fine-tuned TinyGPT checkpoint.

This is the same interface as `generate_tiny_gpt.py`, but defaults to the fine-tuned weights.
"""

import argparse
import re
from pathlib import Path
import torch

from train_tiny_gpt import (
    TinyGPT,
    load_tokenizer,
    get_checkpoint_config,
    load_checkpoint_for_state_dict,
    EMBED_DIM as TRAIN_EMBED_DIM,
    NUM_HEADS as TRAIN_NUM_HEADS,
    NUM_LAYERS as TRAIN_NUM_LAYERS,
    SEQ_LEN as TRAIN_SEQ_LEN,
    DROPOUT as TRAIN_DROPOUT,
)


def pick_device(choice: str):
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

def infer_config_from_state_dict(state_dict: dict) -> dict:
    vocab_size, embed_dim = state_dict["token_emb.weight"].shape
    seq_len = state_dict["pos_emb.weight"].shape[0]

    layer_ids = set()
    for k in state_dict.keys():
        m = re.match(r"transformer\.layers\.(\d+)\.", k)
        if m:
            layer_ids.add(int(m.group(1)))
    num_layers = (max(layer_ids) + 1) if layer_ids else 0

    return {
        "vocab_size": int(vocab_size),
        "embed_dim": int(embed_dim),
        "seq_len": int(seq_len),
        "num_layers": int(num_layers),
    }


def pick_num_heads(embed_dim: int, requested: int) -> int:
    # Heads don't affect parameter shapes, but must divide embed_dim.
    if requested and embed_dim % int(requested) == 0:
        return int(requested)
    for cand in (8, 4, 2, 1):
        if embed_dim % cand == 0:
            return cand
    return 1


def load_model_and_tokenizer(
    model_path: str,
    tokenizer_path: str,
    embed_dim: int,
    num_heads: int,
    num_layers: int,
    seq_len: int,
    dropout: float,
    device: torch.device,
):
    tok = load_tokenizer(tokenizer_path)
    ckpt = torch.load(model_path, map_location=device)
    state = load_checkpoint_for_state_dict(ckpt)

    cfg = get_checkpoint_config(ckpt)
    if cfg:
        embed_dim = int(cfg.get("embed_dim", embed_dim))
        num_heads = int(cfg.get("num_heads", num_heads))
        num_layers = int(cfg.get("num_layers", num_layers))
        seq_len = int(cfg.get("seq_len", seq_len))
        dropout = float(cfg.get("dropout", dropout))
    else:
        inferred = infer_config_from_state_dict(state)
        embed_dim = inferred["embed_dim"]
        seq_len = inferred["seq_len"]
        num_layers = inferred["num_layers"]
        num_heads = pick_num_heads(embed_dim, num_heads)

    model = TinyGPT(
        vocab_size=len(tok.vocab),
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        seq_len=seq_len,
        p_drop=dropout,
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, tok


def generate_once(model, tok, device, prompt, max_new_tokens, temperature, top_k, top_p, repetition_penalty):
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
    )[0].tolist()
    return tok.decode(out)

def build_chat_prompt(question: str, user_tag: str, assistant_tag: str, eol_token: str) -> str:
    # Training expects a role-tagged pattern with <eol> tokens.
    q = question.strip()
    return f"{user_tag} {eol_token} {q} {eol_token} {assistant_tag} {eol_token}"


def extract_answer(decoded_text: str, assistant_tag: str, user_tag: str) -> str:
    """
    Best-effort extraction of the assistant's answer from decoded text.
    `tok.decode()` converts <eol> to newlines, so tags typically appear on their own lines.
    """
    text = decoded_text.strip()
    a_tag = assistant_tag.strip().lower()
    u_tag = user_tag.strip().lower()

    # Find last "assistant tag" line and return everything after it, stopping at the next user tag if it appears.
    needle = f"\n{a_tag}\n"
    if text.lower().startswith(a_tag + "\n"):
        start = len(a_tag) + 1
    else:
        pos = text.lower().rfind(needle)
        start = pos + len(needle) if pos != -1 else 0

    answer = text[start:].lstrip()
    stop_needle = f"\n{u_tag}\n"
    stop = answer.lower().find(stop_needle)
    if stop != -1:
        answer = answer[:stop]
    return answer.strip()


def main(argv=None):
    p = argparse.ArgumentParser(description="TinyGPT fine-tuned inference")
    p.add_argument("--model", default="finetuned_tiny_gpt.pth", help="Path to fine-tuned .pth")
    p.add_argument("--tokenizer", default="tiny_gpt_tokenizer.json", help="Path to tokenizer json (same as base)")

    # Defaults mirror training constants, but can be overridden.
    # If the finetune checkpoint includes config, it will be used automatically.
    p.add_argument("--embed-dim", type=int, default=TRAIN_EMBED_DIM)
    p.add_argument("--num-heads", type=int, default=TRAIN_NUM_HEADS)
    p.add_argument("--num-layers", type=int, default=TRAIN_NUM_LAYERS)
    p.add_argument("--seq-len", type=int, default=TRAIN_SEQ_LEN)
    p.add_argument("--dropout", type=float, default=TRAIN_DROPOUT)

    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    p.add_argument("--prompt", type=str, help="Prompt (or plain question when --chat is set)")
    p.add_argument("--chat", action="store_true", help="Treat --prompt / input as a plain question (wrap with role tags)")
    p.add_argument("--user-tag", default="question", help="Role tag for user (must match finetune data)")
    p.add_argument("--assistant-tag", default="answer", help="Role tag for assistant (must match finetune data)")
    p.add_argument("--answer-only", action="store_true", help="When --chat is set, print only the extracted answer")
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--interactive", action="store_true")
    args = p.parse_args(argv)

    device = pick_device(args.device)
    print(f"Using device: {device}")

    if not Path(args.model).exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not Path(args.tokenizer).exists():
        raise FileNotFoundError(f"Tokenizer not found: {args.tokenizer}")

    model, tok = load_model_and_tokenizer(
        model_path=args.model,
        tokenizer_path=args.tokenizer,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        seq_len=args.seq_len,
        dropout=args.dropout,
        device=device,
    )

    if args.interactive:
        print("TinyGPT fine-tuned REPL (Ctrl+C or empty line to exit)")
        if args.chat:
            print(f"[chat] using pattern: {args.user_tag} <eol> ... <eol> {args.assistant_tag} <eol>")
        try:
            while True:
                prompt = input("\nPrompt> ").strip()
                if not prompt:
                    break
                print("\nGenerating...\n")
                full_prompt = (
                    build_chat_prompt(prompt, args.user_tag, args.assistant_tag, tok.eol_token)
                    if args.chat
                    else prompt
                )
                decoded = generate_once(
                    model,
                    tok,
                    device,
                    full_prompt,
                    args.max_new_tokens,
                    args.temperature,
                    args.top_k,
                    args.top_p,
                    args.repetition_penalty,
                )
                if args.chat and args.answer_only:
                    print(extract_answer(decoded, args.assistant_tag, args.user_tag))
                else:
                    print(decoded)
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
        return

    if not args.prompt:
        raise SystemExit("Provide --prompt or use --interactive.")

    full_prompt = (
        build_chat_prompt(args.prompt, args.user_tag, args.assistant_tag, tok.eol_token)
        if args.chat
        else args.prompt
    )
    decoded = generate_once(
        model,
        tok,
        device,
        full_prompt,
        args.max_new_tokens,
        args.temperature,
        args.top_k,
        args.top_p,
        args.repetition_penalty,
    )
    if args.chat and args.answer_only:
        print(extract_answer(decoded, args.assistant_tag, args.user_tag))
    else:
        print(decoded)


if __name__ == "__main__":
    main()
