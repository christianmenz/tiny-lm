# tinygpt_infer.py
"""
TinyGPT inference that REUSES code from tinygpt_train.py (no duplication).

Examples:
  python tinygpt_infer.py --prompt "The history of machine learning"
  python tinygpt_infer.py --interactive
  # if you trained with different files:
  python tinygpt_infer.py --model tiny_gpt.pth --tokenizer tiny_gpt_tokenizer.json
"""

import argparse
from pathlib import Path
import torch

# --- reuse everything from the training script ---
from train_tiny_gpt import (
    TinyGPT,
    default_suppress_ids,
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
    # auto
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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
    ckpt_cfg = get_checkpoint_config(ckpt)
    if ckpt_cfg:
        # Prefer the saved config to avoid silent mismatch.
        embed_dim = int(ckpt_cfg["embed_dim"])
        num_heads = int(ckpt_cfg["num_heads"])
        num_layers = int(ckpt_cfg["num_layers"])
        seq_len = int(ckpt_cfg["seq_len"])
        dropout = float(ckpt_cfg["dropout"])

    model = TinyGPT(
        vocab_size=len(tok.vocab),
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        seq_len=seq_len,
        p_drop=dropout,
    ).to(device)

    state = load_checkpoint_for_state_dict(ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, tok


def generate_once(
    model,
    tok,
    device,
    prompt,
    max_new_tokens,
    temperature,
    top_k,
    top_p,
    repetition_penalty,
    greedy,
    allow_nonprintable_bytes,
):
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    if ids.size(1) == 0:
        raise ValueError("Prompt encoded to zero tokens; provide a non-empty prompt known to the tokenizer.")
    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        greedy=greedy,
        suppress_ids=default_suppress_ids(tok, allow_nonprintable_bytes=allow_nonprintable_bytes),
    )[0].tolist()
    return tok.decode(out)


def repl(model, tok, device, args):
    print("TinyGPT REPL (Ctrl+C or empty line to exit)")
    try:
        while True:
            prompt = input("\nPrompt> ").strip()
            if not prompt:
                break
            print("\nGenerating...\n")
            text = generate_once(
                model, tok, device, prompt,
                args.max_new_tokens,
                args.temperature,
                args.top_k,
                args.top_p,
                args.repetition_penalty,
                args.greedy,
                args.allow_nonprintable_bytes,
            )
            print(text)
    except (KeyboardInterrupt, EOFError):
        print("\nBye!")


def main(argv=None):
    p = argparse.ArgumentParser(description="TinyGPT inference (imports from tinygpt_train.py)")
    p.add_argument("--model", default="tiny_gpt.pth", help="Path to .pth weights")
    p.add_argument("--tokenizer", default="tiny_gpt_tokenizer.json", help="Path to tokenizer json")

    # Defaults mirror your training script constants, but can be overridden
    p.add_argument("--embed-dim", type=int, default=TRAIN_EMBED_DIM)
    p.add_argument("--num-heads", type=int, default=TRAIN_NUM_HEADS)
    p.add_argument("--num-layers", type=int, default=TRAIN_NUM_LAYERS)
    p.add_argument("--seq-len", type=int, default=TRAIN_SEQ_LEN)
    p.add_argument("--dropout", type=float, default=TRAIN_DROPOUT)

    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    p.add_argument("--prompt", type=str, help="Prompt to complete (omit for --interactive)")
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=40, help="<=0 disables top-k")
    p.add_argument("--top-p", type=float, default=0.95, help="(0,1) enables nucleus sampling; else disabled")
    p.add_argument("--repetition-penalty", type=float, default=1.1, help="1.0 disables")
    p.add_argument("--greedy", action="store_true", help="Greedy decoding (argmax), disables sampling randomness")
    p.add_argument(
        "--allow-nonprintable-bytes",
        action="store_true",
        help="For byte tokenizers, allow arbitrary byte values in generated text.",
    )
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
        repl(model, tok, device, args)
        return

    if not args.prompt:
        raise SystemExit("Provide --prompt or use --interactive.")

    text = generate_once(
        model, tok, device, args.prompt,
        args.max_new_tokens,
        args.temperature,
        args.top_k,
        args.top_p,
        args.repetition_penalty,
        args.greedy,
        args.allow_nonprintable_bytes,
    )
    print(text)


if __name__ == "__main__":
    main()
