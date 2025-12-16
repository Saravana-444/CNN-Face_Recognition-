import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import gdown
import os

# -----------------------------
# CONFIG
# -----------------------------
MODEL_URL = "https://drive.google.com/uc?id=1BQAWXyu7FcMq7rbKG6D5CJBuUCis7ukk"
MODEL_PATH = "face_recognition_augmented_3class.h5"
IMG_SIZE = 128

CLASS_NAMES = ["Gobinath", "Guru Nagajothi", "Saravana kumar"]  # change if needed

# -----------------------------
# DOWNLOAD MODEL FROM DRIVE
# -----------------------------
@st.cache_resource
def load_cnn_model():
    if not os.path.exists(MODEL_PATH):
        st.write("Downloading model from Google Drive...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    model = load_model(MODEL_PATH)
    return model

model = load_cnn_model()

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🧠 Face Recognition (CNN - 3 Classes)")
st.write("Upload a face image to identify the person")

uploaded_file = st.file_uploader(
    "Upload Face Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    img = image.load_img(uploaded_file, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    st.success(f"### Predicted Person: {CLASS_NAMES[class_index]}")
    st.info(f"Confidence: {confidence:.2f}%")
