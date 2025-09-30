# prepare_oasst_pairs.py
"""
Create role-prefixed QA pairs from OpenAssistant/oasst1.

It writes lines in this pattern (lowercased):
question
<user text>
answer
<assistant text>

Blank line between pairs for readability. Your finetuner will read non-empty lines
and insert <eol> automatically.

Usage:
  python prepare_oasst_pairs.py --out openassist_pairs_en.txt --lang en
  # choose tags that exist in your vocab, e.g. question/answer (default)
  python prepare_oasst_pairs.py --out openassist_pairs_en.txt --user-tag question --assistant-tag answer
"""

import argparse
from collections import defaultdict
from datasets import load_dataset

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="openassist_pairs_en.txt")
    ap.add_argument("--lang", default="en", help="Filter by language code (set '' to keep all)")
    ap.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    ap.add_argument("--limit", type=int, default=None, help="Cap number of pairs")
    ap.add_argument("--user-tag", default="question", help="Role tag for user (plain token)")
    ap.add_argument("--assistant-tag", default="answer", help="Role tag for assistant (plain token)")
    args = ap.parse_args()

    ds = load_dataset("OpenAssistant/oasst1")

    # Collect rows across selected splits
    rows = []
    for sp in args.splits:
        if sp not in ds:
            continue
        for r in ds[sp]:
            if args.lang and str(r.get("lang", "")).lower() != args.lang.lower():
                continue
            # normalize keys across possible variants
            rid = r.get("message_id") or r.get("id")
            pid = r.get("parent_id")
            role = r.get("role")  # expected 'prompter' or 'assistant'
            text = r.get("text")
            if not (rid and role and isinstance(text, str) and text.strip()):
                continue
            rows.append({
                "id": rid,
                "parent_id": pid,
                "role": role,
                "text": text.strip().replace("\r", " "),
            })

    # Build index
    by_id = {r["id"]: r for r in rows}

    # Create simple QA pairs: parent = prompter, child = assistant
    pairs = []
    children = defaultdict(list)
    for r in rows:
        if r["parent_id"]:
            children[r["parent_id"]].append(r["id"])

    for cid, child in by_id.items():
        if child["role"] != "assistant":
            continue
        pid = child["parent_id"]
        if not pid:
            continue
        parent = by_id.get(pid)
        if not parent or parent["role"] != "prompter":
            continue
        q = parent["text"].strip().lower().replace("\n", " ")
        a = child["text"].strip().lower().replace("\n", " ")
        if q and a:
            pairs.append((q, a))
        if args.limit and len(pairs) >= args.limit:
            break

    print(f"Built {len(pairs)} QA pairs (lang='{args.lang or 'ALL'}'). Writing {args.out} ...")
    with open(args.out, "w", encoding="utf-8") as f:
        for q, a in pairs:
            f.write(f"{args.user_tag}\n")
            f.write(q + "\n")
            f.write(f"{args.assistant_tag}\n")
            f.write(a + "\n\n")  # blank line between pairs

    print("Done.")

if __name__ == "__main__":
    main()
