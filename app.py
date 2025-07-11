import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('tile_defect_model.h5')

# Title
st.title("🧱 Tile Surface Defect Detection")
st.write("Upload a tile image and the model will predict whether it's Good or Bad.")

# Upload image
uploaded_file = st.file_uploader("Choose a tile image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Tile Image', use_column_width=True)

    # Preprocess image
    image = image.resize((128, 128))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(image_array)[0][0]
    label = "🟩 Good Tile" if prediction < 0.5 else "🟥 Bad Tile"
    
    st.markdown(f"### Prediction: {label}")
    st.markdown(f"**Confidence Score:** `{(1 - prediction) if prediction < 0.5 else prediction:.2f}`")
