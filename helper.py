import cv2 as cv
import tensorflow as tf
import numpy as np


model = tf.keras.models.load_model('model.h5')

def preproc(image):
    img = cv.imread(image)
    print(img.shape)
    img = cv.resize(img, (224, 224))
    print(img.shape)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img


def processing(img):
    img = preproc(img)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    label = "FAKE" if prediction[0, 0] > 0.33 else "REAL"
    return label
