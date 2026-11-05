import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import numpy as np
from model import ISLModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "..", "models", "isl_model.pth")
labels_path = os.path.join(BASE_DIR, "..", "models", "labels_integers.pth")

model = ISLModel()
weights = torch.load(model_path, weights_only=True)
model.load_state_dict(weights)
model.eval()  # switching to evaluation mode - Dropout is disabled

# key - labels, value - integers
labels_integers = torch.load(labels_path, weights_only=True)

# key - integers, value - labels
integers_labels = {value: key for key, value in labels_integers.items()}


def predict_sign(d) -> str:
    """Return the predicted sign label (string)."""
    label, _ = predict_sign_with_confidence(d)
    return label


def predict_sign_with_confidence(d) -> tuple[str, float]:
    """Return (label, confidence) where confidence is in [0, 1]."""
    data = np.array(d)
    data_x = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # (1, 126)

    with torch.no_grad():
        output = model(data_x)
        probs = F.softmax(output, dim=1)          # (1, num_classes)
        confidence, prediction = probs.max(dim=1)  # both (1,)
        label = integers_labels[prediction[0].item()]
        conf_val = confidence[0].item()

    return label, conf_val


