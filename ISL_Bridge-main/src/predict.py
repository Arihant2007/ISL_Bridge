import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F
from model import ISLModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "isl_model.pth")
LABELS_PATH = os.path.join(BASE_DIR, "..", "models", "labels_integers.pth")

model = None
integers_labels = {}
MODEL_ERROR = None

try:
    # Git LFS pointer text files are not usable PyTorch checkpoints.
    with open(MODEL_PATH, "rb") as checkpoint:
        if checkpoint.read(128).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise RuntimeError("Model checkpoint is a Git LFS pointer; download the actual model file.")
    with open(LABELS_PATH, "rb") as labels_file:
        if labels_file.read(128).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise RuntimeError("Label map is a Git LFS pointer; download the actual label file.")

    model = ISLModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()
    labels_integers = torch.load(LABELS_PATH, map_location="cpu", weights_only=True)
    integers_labels = {value: key for key, value in labels_integers.items()}
except Exception as exc:
    MODEL_ERROR = str(exc)


def predict_sign(d) -> str | None:
    label, _ = predict_sign_with_confidence(d)
    return label


def predict_sign_with_confidence(d) -> tuple[str | None, float]:
    """Return (label, confidence), or (None, 0) when model files are unavailable."""
    if model is None:
        return None, 0.0

    data = np.asarray(d, dtype=np.float32)
    if data.size != 126:
        return None, 0.0
    data_x = torch.from_numpy(data).unsqueeze(0)

    with torch.no_grad():
        probs = F.softmax(model(data_x), dim=1)
        confidence, prediction = probs.max(dim=1)
        label = integers_labels.get(prediction.item())
        if label is None:
            return None, 0.0
        return label, float(confidence.item())
