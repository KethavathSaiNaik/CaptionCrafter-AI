

import pickle
from pathlib import Path
from typing import Final, List

import numpy as np
from PIL import Image

from tensorflow.keras.models import load_model  # pyright: ignore[reportMissingImports]
from tensorflow.keras.preprocessing.image import img_to_array, load_img  # pyright: ignore[reportMissingImports]
from tensorflow.keras.preprocessing.sequence import pad_sequences # pyright: ignore[reportMissingImports]
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess  # pyright: ignore[reportMissingImports]

# ── Paths ──────────────────────────────────────────────────────────────────────
CAPTION_MODEL_PATH   = Path("model.keras")
FEATURE_EXTRACT_PATH = Path("feature_extractor.keras")
TOKENIZER_PATH       = Path("tokenizer.pkl")

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_LEN:   Final[int] = 34
START:     Final[str] = "startseq"
END:       Final[str] = "endseq"
IMG_SIZE:  Final[int] = 224  # EfficientNet default

# ── One-time loads (avoids reloading every request) ───────────────────────────
caption_model    = load_model(CAPTION_MODEL_PATH)
feature_extractor = load_model(FEATURE_EXTRACT_PATH)
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

# ── Helpers ───────────────────────────────────────────────────────────────────
def _extract_features(pil_img: Image.Image) -> np.ndarray:
    """Return EfficientNet feature vector for a PIL image."""
    img_arr = img_to_array(pil_img.resize((IMG_SIZE, IMG_SIZE)).convert("RGB"))
    img_arr = eff_preprocess(img_arr)
    img_arr = np.expand_dims(img_arr, axis=0)            # (1, 224, 224, 3)
    feats   = feature_extractor.predict(img_arr, verbose=0)
    return np.squeeze(feats)                             # (feature_dim,)

def _id_to_word(index: int) -> str | None:
    return tokenizer.index_word.get(index)

def generate_caption(pil_img: Image.Image) -> str:
    """Generate a caption for a PIL image using the trained models."""
    image_features = _extract_features(pil_img)

    in_text: List[str] = [START]
    for _ in range(MAX_LEN):
        # sequence → padded int array
        seq = tokenizer.texts_to_sequences([" ".join(in_text)])[0]
        seq = pad_sequences([seq], maxlen=MAX_LEN)

        # predict next word
        yhat = caption_model.predict(
            [np.expand_dims(image_features, axis=0), seq],
            verbose=0
        )
        next_id = int(np.argmax(yhat))
        word    = _id_to_word(next_id)

        if word is None:
            break
        in_text.append(word)
        if word == END:
            break

    # Strip startseq / endseq and neatly join
    caption_words = [w for w in in_text if w not in {START, END}]
    return " ".join(caption_words).capitalize()
