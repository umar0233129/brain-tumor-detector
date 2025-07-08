import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import gdown
import os
from PIL import Image

# --- Model download ---
model_path = 'model.h5'
if not os.path.exists(model_path):
    url = 'https://drive.google.com/uc?id=1Mylf9TBYBwSDGdzeqZPEAPKq354W1I1-'
    gdown.download(url, model_path, quiet=False)

# --- Load model ---
model = load_model(model_path)
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

# --- Streamlit UI ---
st.title("🧠 Brain Tumor Detection System")
st.write("Upload an MRI image to detect brain tumor using VGG16 Deep Learning Model.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # --- Preprocess image ---
    image = img.resize((128, 128))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # --- Prediction ---
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0])) * 100

    if class_labels[predicted_class] == 'notumor':
        st.success(f"No Tumor Detected ✅ (Confidence: {confidence:.2f}%)")
    else:
        st.error(f"Tumor Detected: {class_labels[predicted_class].capitalize()} ❗ (Confidence: {confidence:.2f}%)")
