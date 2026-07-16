import torch
from tinygpt import TinyGPT, device

checkpoint_path = "tinygpt.pth"

def load_model():
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = TinyGPT(checkpoint["vocab_size"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()   # tells PyTorch we're generating, not training
    return model, checkpoint["stoi"], checkpoint["itos"]

def encode(text, stoi):
    return [stoi[c] for c in text if c in stoi]

def decode(ids, itos):
    return "".join(itos[i] for i in ids)

def main():
    model, stoi, itos = load_model()
    print("Model loaded. Type a prompt and press Enter.")
    print("Type 'quit' to exit.\n")
    while True:
        prompt = input("You: ")
        if prompt.lower() == "quit":
            break
        context = torch.tensor([encode(prompt, stoi)], dtype=torch.long, device=device)
        with torch.no_grad():
            generated = model.generate(context, max_new_tokens=200)
        full_text = decode(generated[0].tolist(), itos)
        print("\nModel:", full_text)
        print()

if __name__ == "__main__":
    main()