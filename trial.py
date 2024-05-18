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


st.page_icon="🦁",  # Lion emoji

# Streamlit app title
st.title("Animal Classification")
st.write("Upload an image of a Cheetah or Lion, and the model will predict the class.")

background_image_url = "https://i.ytimg.com/vi/4iJJXtxytqE/maxresdefault.jpg"

st.markdown(
    f"""
    <style>
    .stApp {{
        background: url({background_image_url});
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: black;
    }}
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, .stApp p, .stApp label, .stApp .caption {{
        color: black;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


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
    if image.shape[2] == 4:
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

