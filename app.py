import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown

# Download model from Google Drive (direct link or use gdown)
MODEL_PATH = 'model.h5'
if not tf.io.gfile.exists(MODEL_PATH):
    gdown.download('https://drive.google.com/uc?id=1Mylf9TBYBwSDGdzeqZPEAPKq354W1I1-', MODEL_PATH, quiet=False)

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Class names
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Streamlit UI
st.title("Brain Tumor Detection")
st.write("Upload an MRI image to detect tumor type.")

uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((128, 128))
    st.image(image, caption='Uploaded Image', use_column_width=True)

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    pred_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.subheader(f"Prediction: **{pred_class.upper()}**")
    st.write(f"Confidence: {confidence * 100:.2f}%")
