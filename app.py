import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

MODEL_URL = 'https://drive.google.com/uc?id=1Mylf9TBYBwSDGdzeqZPEAPKq354W1I1-'
MODEL_PATH = 'model.h5'

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()
class_names = ['pituitary', 'glioma', 'notumor', 'meningioma']

st.title("🧠 Brain Tumor Detector")
st.write("Upload an MRI image to detect brain tumor.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    img = image.resize((128, 128))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions) * 100

    if class_names[predicted_class] == 'notumor':
        st.success(f"✅ No Tumor Detected! Confidence: {confidence:.2f}%")
    else:
        st.error(f"⚠️ Tumor Detected: {class_names[predicted_class]} (Confidence: {confidence:.2f}%)")

