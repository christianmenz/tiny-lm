import torch
from simple_tokenizer import SimpleTokenizer
from train_tiny_gpt import TinyGPT, EMBED_DIM, NUM_HEADS, NUM_LAYERS, SEQ_LEN

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Use the same line limit as in training/fine-tuning
LIMIT = 10000
with open("wikitext-2/train.txt", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()][:LIMIT]
tokenizer = SimpleTokenizer()
tokenizer.fit(lines)

# Load fine-tuned model
model = TinyGPT(len(tokenizer.vocab), EMBED_DIM, NUM_HEADS, NUM_LAYERS, SEQ_LEN)
model.load_state_dict(torch.load("finetuned_tiny_gpt.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

def generate(prompt, max_new_tokens=64):
    tokens = tokenizer.encode(prompt)
    tokens = tokens[-SEQ_LEN:]  # ensure length
    for _ in range(max_new_tokens):
        x = torch.tensor([tokens[-SEQ_LEN:]], dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            logits = model(x)
            next_token_logits = logits[0, -1]
            next_token = torch.argmax(next_token_logits).item()
        tokens.append(next_token)
    return tokenizer.decode(tokens)

if __name__ == "__main__":
    prompt = input("Enter code prompt: ")
    output = generate(prompt)
    print("\nGenerated:\n", output)
