Welcome! This is my git repo of me working to create an AI Lab Assistant. This assistant is meant to be ran on any device with a CPU with little requirements on how strong the components are.

## How to customize this AI for your own use

If you desire to train the model yourself then open `tinygpt.py` and change the `max_iters` to however high or low of a number you desire, I recommend that you change the `eval_iterval` to be a complete modulus of the `max_iter`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Training

```bash
python tinygpt.py training_data/
```

The cache is cleared automatically at the start of each training run so it re-processes the latest data.

## Inference

```bash
python run.py
```

Type a prompt and the model will generate up to 200 continuation characters. Type `quit` to exit.

## Development

Install dev dependencies:

```bash
pip install pytest ruff
```

Run tests:

```bash
pytest -v
```

Run linter:

```bash
ruff check .
```
