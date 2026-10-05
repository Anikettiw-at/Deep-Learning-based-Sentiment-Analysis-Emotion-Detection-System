import pickle
import re
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from pydantic import BaseModel, ConfigDict, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Artifacts" / "BiGRU_Model.keras"
TOKENIZER_PATH = BASE_DIR / "Artifacts" / "tokenizer.pkl"
STATIC_DIR = BASE_DIR / "static"

MAX_LEN = 50  # same maxlen used in the notebook

# order matters - it's the label order from dair-ai/emotion
LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def clean_text(text: str) -> str:
    # the dataset is lowercase with no apostrophes ("im", "didnt"),
    # so we bring user input into the same shape before tokenizing
    text = text.lower()
    text = text.replace("'", "").replace("’", "")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class TextIn(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        json_schema_extra={"example": "I feel so happy and excited"},
    )


class Prediction(BaseModel):
    text: str
    predicted_emotion: str
    confidence: float
    all_probabilities: dict[str, float]


class Health(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool


# filled once at startup, emptied on shutdown
store = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("loading model + tokenizer...")
    # compile=False: we only need inference, not the saved optimizer state
    model = load_model(MODEL_PATH, compile=False)

    # wrapping the forward pass in tf.function makes a single prediction
    # ~9ms instead of ~500ms with plain model() / model.predict()
    @tf.function(input_signature=[tf.TensorSpec([None, MAX_LEN], tf.int32)])
    def infer(x):
        return model(x, training=False)

    infer(tf.zeros([1, MAX_LEN], tf.int32))  # warm-up so the first request isn't slow
    store["model"] = infer
    with open(TOKENIZER_PATH, "rb") as f:
        store["tokenizer"] = pickle.load(f)
    print("ready")
    yield
    store.clear()


app = FastAPI(title="Emotion Prediction API", lifespan=lifespan)

# credentials aren't used, and "*" + allow_credentials isn't valid anyway
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", response_model=Health)
def health():
    loaded = "model" in store and "tokenizer" in store
    return Health(status="ok", model_loaded=loaded)


@app.post("/predict", response_model=Prediction)
def predict(body: TextIn):
    model = store.get("model")
    tokenizer = store.get("tokenizer")
    if model is None or tokenizer is None:
        raise HTTPException(503, "Model is still loading, try again in a few seconds.")

    cleaned = clean_text(body.text)
    if not cleaned:
        raise HTTPException(422, "Couldn't find any words in that text. Try a full sentence.")

    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")

    probs = model(tf.constant(padded, tf.int32)).numpy()[0]
    best = int(np.argmax(probs))

    return Prediction(
        text=body.text,
        predicted_emotion=LABELS[best],
        confidence=float(probs[best]),
        all_probabilities={label: float(p) for label, p in zip(LABELS, probs)},
    )
