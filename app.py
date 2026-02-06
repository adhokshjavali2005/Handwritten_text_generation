from fastapi import FastAPI
import numpy as np
import random
from tensorflow.keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load model
model = load_model("handwritten_text_rnn.h5")

# Load corpus
with open("text_corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

sequence_length = 60

def sample(preds, temperature=0.4):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(preds), p=preds)

def generate_text(length=120):
    start = random.randint(0, len(text) - sequence_length - 1)
    seed = text[start:start + sequence_length]
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
    

@app.get("/")
def home():
    return {"status": "Handwritten Text Generator API running"}

@app.get("/generate")
def generate():
    return {"generated_text": generate_text()}



