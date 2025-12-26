"""
Quick sanity checks for the training / fine-tuning data files in this repo.

Usage:
  .venv/bin/python validate_data.py
  .venv/bin/python validate_data.py --openassist openassist_pairs_en.txt
  .venv/bin/python validate_data.py --wikitext-dir wikitext-2
"""

from __future__ import annotations

import argparse
from pathlib import Path


def check_wikitext(wikitext_dir: Path):
    print(f"[wikitext] dir: {wikitext_dir}")
    for name in ["train.txt", "validation.txt", "test.txt"]:
        p = wikitext_dir / name
        if not p.exists():
            print(f"  - missing: {p}")
            continue
        n_lines = 0
        n_nonempty = 0
        n_chars = 0
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                n_lines += 1
                s = line.strip()
                if s:
                    n_nonempty += 1
                    n_chars += len(s)
        print(f"  - {name}: lines={n_lines} nonempty={n_nonempty} chars(nonempty)={n_chars}")


def check_openassist(path: Path, user_tag: str = "question", assistant_tag: str = "answer"):
    print(f"[openassist] file: {path}")
    if not path.exists():
        print(f"  - missing: {path}")
        return

    ut = user_tag.strip().lower()
    at = assistant_tag.strip().lower()

    pairs = 0
    bad = 0
    state = "expect_tag"  # expect_tag | in_user | in_assistant
    saw_user = False
    saw_assistant = False

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                # blank line ends a pair
                if saw_user or saw_assistant:
                    if saw_user and saw_assistant:
                        pairs += 1
                    else:
                        bad += 1
                state = "expect_tag"
                saw_user = False
                saw_assistant = False
                continue

            s = line.lower()
            if s == ut:
                if saw_user and not saw_assistant:
                    bad += 1  # repeated user tag without answer
                state = "in_user"
                saw_user = True
                continue
            if s == at:
                if not saw_user:
                    bad += 1  # answer without question
                state = "in_assistant"
                saw_assistant = True
                continue

            # content line: just track that it exists for the current state
            if state == "expect_tag":
                bad += 1  # content without a tag

    # file might not end with a blank line
    if saw_user or saw_assistant:
        if saw_user and saw_assistant:
            pairs += 1
        else:
            bad += 1

    print(f"  - pairs={pairs} bad_blocks={bad} tags=({ut},{at})")


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description="Sanity checks for TinyGPT data files.")
    ap.add_argument("--wikitext-dir", default="wikitext-2", help="Directory containing Wikitext-2 files")
    ap.add_argument("--openassist", default="openassist_pairs_en.txt", help="Role-tagged OpenAssistant pairs file")
    ap.add_argument("--user-tag", default="question")
    ap.add_argument("--assistant-tag", default="answer")
    args = ap.parse_args(argv)

    check_wikitext(Path(args.wikitext_dir))
    check_openassist(Path(args.openassist), user_tag=args.user_tag, assistant_tag=args.assistant_tag)


if __name__ == "__main__":
    main()

