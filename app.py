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
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "handwritten_text_rnn.h5")
TEXT_PATH = os.path.join(BASE_DIR, "text_corpus.txt")

# -----------------------------
# Load model (DEBUG INCLUDED)
# -----------------------------
model = load_model(MODEL_PATH)
print("MODEL LOADED — VOCAB SIZE:", model.output_shape[-1])

# -----------------------------
# Load corpus (NO FALLBACK)
# -----------------------------
if not os.path.exists(TEXT_PATH):
    raise RuntimeError("text_corpus.txt missing. Must match training corpus.")

with open(TEXT_PATH, "r", encoding="utf-8") as f:
    text = f.read()

if len(text) < 100:
    raise RuntimeError("text_corpus.txt too small. Must be same as training.")

# -----------------------------
# Vocabulary (MUST MATCH TRAINING)
# -----------------------------
chars = sorted(set(text))
vocab_size = len(chars)

print("RUNTIME VOCAB SIZE:", vocab_size)

char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

# -----------------------------
# Sequence length (MATCH TRAINING)
# -----------------------------
SEQUENCE_LENGTH = 40  # same as training

# -----------------------------
# Sampling
# -----------------------------
def sample(preds, temperature=0.4):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(preds), p=preds)

# -----------------------------
# Text generation
# -----------------------------
def generate_text(length=120):
    max_start = len(text) - SEQUENCE_LENGTH - 1
    if max_start <= 0:
        raise RuntimeError("Corpus too small for generation.")

    start = random.randint(0, max_start)
    seed = text[start : start + SEQUENCE_LENGTH]
    generated = seed

    for _ in range(length):
        x = np.zeros((1, SEQUENCE_LENGTH, vocab_size))
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
        return {
            "success": True,
            "generated_text": generate_text()
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "trace": traceback.format_exc()
        }
