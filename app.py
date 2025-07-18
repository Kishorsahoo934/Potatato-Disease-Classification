import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load model
model = tf.keras.models.load_model("model.h5")

# Class names
class_names = ['Early Blight', 'Late Blight', 'Healthy']  # Update as per your model

# Preprocess function
def preprocess_image(img, target_size=(256, 256)):
    image = img.convert("RGB")
    image = image.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

# Predict function
def predict_image(model, image, class_names):
    img = preprocess_image(image)
    prediction = model.predict(img)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = np.max(prediction) * 100
    return predicted_class, confidence

# 🌿 UI Design
st.set_page_config(page_title="🌿 Plant Disease Classifier", layout="centered")

st.markdown("<h1 style='text-align: center; color: green;'>🌱 Plant Disease Detection</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid #90ee90;'>", unsafe_allow_html=True)
st.markdown("### 📤 Upload a leaf image to predict the disease")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    with st.spinner("🔍 Predicting..."):
        predicted_class, confidence = predict_image(model, image, class_names)

    st.success(f"🧪 **Prediction:** `{predicted_class}`")
    st.info(f"📊 **Confidence:** `{confidence:.2f}%`")
    st.markdown("---")
st.markdown("🌾 Built with ❤️ by **Kishor** | Powered by TensorFlow & Streamlit")
