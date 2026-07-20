import pytest

try:
    import torch
except (OSError, ImportError) as e:
    pytest.skip(f"torch load failed: {e}", allow_module_level=True)

from tinygpt import TinyGPT, build_tokenizer, encode_text, decode_text, device


def test_model_forward():
    model = TinyGPT(vocab_size=50).to(device)
    x = torch.randint(0, 50, (2, 10), device=device)
    logits, loss = model(x, targets=x)
    assert logits.shape == (2, 10, 50)
    assert loss is not None
    assert loss.item() > 0


def test_model_forward_no_target():
    model = TinyGPT(vocab_size=50).to(device)
    x = torch.randint(0, 50, (2, 10), device=device)
    logits, loss = model(x)
    assert logits.shape == (2, 10, 50)
    assert loss is None


def test_generate():
    model = TinyGPT(vocab_size=50).to(device)
    x = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(x, max_new_tokens=20)
    assert out.shape == (1, 21)


def test_build_tokenizer():
    encode, decode, vocab_size, stoi, itos = build_tokenizer("abc")
    assert vocab_size == 3
    assert encode("abc") == [0, 1, 2]
    assert decode([0, 1, 2]) == "abc"


def test_encode_decode_roundtrip():
    text = "hello world"
    _, _, _, stoi, itos = build_tokenizer(text)
    encoded = encode_text(text, stoi)
    decoded = decode_text(encoded, itos)
    assert decoded == text
