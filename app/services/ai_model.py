import os
# Must be set before importing tensorflow to load older Keras 2 models correctly
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import cv2 as cv
import tensorflow as tf
import numpy as np
from pathlib import Path

# Ensure model is loaded relative to the project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_PATH = os.getenv("MODEL_PATH", str(PROJECT_ROOT / "model.h5"))
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}: {e}")
    model = None

def preproc(image):
    nparr = np.frombuffer(image, np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_COLOR)
    img = cv.resize(img, (224, 224))
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img

def processing(img):
    if model is None:
        raise RuntimeError("Model is not loaded.")
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    label = "FAKE" if prediction[0, 0] > 0.5 else "REAL"
    percentage = float(round((prediction[0, 0] * 100), 2))
    return label, percentage
