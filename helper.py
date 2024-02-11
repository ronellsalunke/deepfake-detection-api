import cv2 as cv
import tensorflow as tf
import numpy as np


model = tf.keras.models.load_model('model.h5')

def preproc(image):
    nparr = np.frombuffer(image, np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_COLOR)
    img = cv.resize(img, (224, 224))
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img


def processing(img):
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    label = "FAKE" if prediction[0, 0] > 0.5 else "REAL"
    percentage = round((prediction[0, 0] * 100), 2)
    return label, percentage
