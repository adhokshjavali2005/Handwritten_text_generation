from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import random
import os
import traceback
from tensorflow.keras.models import load_model

# -----------------------------
# App setup
# -----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow Builder.io / any frontend
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------
# Load model
# -----------------------------
MODEL_PATH = os.path.join(BASE_DIR, "handwritten_text_rnn.h5")
TEXT_PATH = os.path.join(BASE_DIR, "text_corpus.txt")

model = load_model(MODEL_PATH)

# -----------------------------
# Load corpus safely
# -----------------------------
with open(TEXT_PATH, "r", encoding="utf-8") as f:
    text = f.read().strip()

if len(text) < 10:
    raise ValueError("text_corpus.txt is too small. Add more text.")

# -----------------------------
# Vocabulary
# -----------------------------
chars = sorted(list(set(text)))
vocab_size = len(chars)

char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

# -----------------------------
# Sequence length (SAFE)
# -----------------------------
MAX_SEQUENCE_LENGTH = 60
sequence_length = min(MAX_SEQUENCE_LENGTH, len(text) - 1)

if sequence_length < 5:
    raise ValueError(
        f"Dataset too small. Need at least 6 chars, got {len(text)}"
    )

# -----------------------------
# Sampling function
# -----------------------------
def sample(preds, temperature=0.4):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(preds), p=preds)

# -----------------------------
# Text generation (SAFE)
# -----------------------------
def generate_text(length=120):
    max_start = len(text) - sequence_length - 1

    if max_start <= 0:
        raise ValueError(
            f"Dataset too small for sequence length {sequence_length}"
        )

    start = random.randint(0, max_start)
    seed = text[start : start + sequence_length]
    generated = seed

    for _ in range(length):
        x = np.zeros((1, sequence_length, vocab_size))
        for t, char in enumerate(seed):
            x[0, t, char_to_idx[char]] = 1

        preds = model.predict(x, verbose=0)[0]
        next_char = idx_to_char[sample(preds)]

        generated += next_char
        seed = seed[1:] + next_char

    return generated

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def health():
    return {"status": "Handwritten Text Generator API running"}

@app.get("/generate")
def generate():
    try:
        text_out = generate_text(length=120)
        return {
            "success": True,
            "generated_text": text_out
        }
    except Exception as e:
        return {
            "success": False,
            "error": "Internal Server Error in /generate",
            "details": str(e),
            "trace": traceback.format_exc()
        }
