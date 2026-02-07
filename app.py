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
# Load corpus safely (NO CRASH)
# -----------------------------
try:
    with open(TEXT_PATH, "r", encoding="utf-8") as f:
        text = f.read().strip()
except Exception:
    text = ""

# Fallback corpus if file is missing or too small
if len(text) < 50:
    text = (
        "This is a fallback text corpus used for handwritten text generation. "
        "It ensures the API can run safely even when the original dataset "
        "is small or missing. The model will still generate character patterns."
    )

# -----------------------------
# Vocabulary
# -----------------------------
chars = sorted(list(set(text)))
vocab_size = len(chars)

char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

# -----------------------------
# Sequence length (SAFE, NO CRASH)
# -----------------------------
MAX_SEQUENCE_LENGTH = 60
sequence_length = min(MAX_SEQUENCE_LENGTH, max(10, len(text) - 1))

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
# Text generation (RUNTIME SAFE)
# -----------------------------
def generate_text(length=120):
    # Recompute safe sequence length at runtime
    effective_seq_len = min(sequence_length, len(text) - 1)

    if effective_seq_len < 5:
        return "Dataset too small to generate text safely."

    max_start = len(text) - effective_seq_len - 1
    if max_start <= 0:
        return "Dataset too small to select a valid starting point."

    start = random.randint(0, max_start)
    seed = text[start : start + effective_seq_len]
    generated = seed

    for _ in range(length):
        x = np.zeros((1, effective_seq_len, vocab_size))
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
