import os
from datasets import load_dataset

SAVE_DIR = "wikitext-2"


def save_wikitext2():
    os.makedirs(SAVE_DIR, exist_ok=True)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    for split in ["train", "validation", "test"]:
        with open(os.path.join(SAVE_DIR, f"{split}.txt"), "w", encoding="utf-8") as f:
            for line in dataset[split]["text"]:
                f.write(line + "\n")
    print(f"Wikitext-2 splits saved to {SAVE_DIR}/")

if __name__ == "__main__":
    save_wikitext2()
