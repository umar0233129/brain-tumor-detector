import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# Load the trained model
model = load_model('model.h5')

# Class labels
class_names = ['pituitary', 'glioma', 'notumor', 'meningioma']

# Streamlit UI
st.title("🧠 Brain Tumor Detection")

uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    img = image.resize((128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    if predicted_class == 'notumor':
        st.success(f"✅ No Tumor Detected ({confidence:.2f}%)")
    else:
        st.error(f"❗ Tumor Detected: {predicted_class.upper()} ({confidence:.2f}%)")
