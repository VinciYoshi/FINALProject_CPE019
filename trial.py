import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
@st.cache(allow_output_mutation=True)
def load_trained_model():
    model = load_model("best_model.h5")
    return model

model = load_trained_model()

# Class labels
class_labels = ["Cheetah", "Lion"]

# Streamlit app title and description
st.title("Animal Classification")
st.write("Upload an image of a Cheetah or Lion, and the model will predict the class.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Convert the image to the correct format
    image = np.array(image)
    if image.shape[2] == 4:  # if the image has an alpha channel, remove it
        image = image[:, :, :3]
    image = cv2.resize(image, (128, 128))
    image = image / 255.0
    image = np.reshape(image, (1, 128, 128, 3))

    # Make prediction
    predicted_probabilities = model.predict(image)
    predicted_class = np.argmax(predicted_probabilities)

    # Display the prediction
    st.write(f"Predicted class: **{class_labels[predicted_class]}**")

    # Display probabilities
    st.write(f"Confidence: {predicted_probabilities[0][predicted_class] * 100:.2f}%")

# To run this app, save it as streamlit_app.py and use the command `streamlit run streamlit_app.py`
