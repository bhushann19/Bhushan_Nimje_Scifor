import streamlit as st
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np

# Load the trained model
model = load_model("Pneumonia Detection.keras")

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = load_img(image, target_size=(224, 224))
    img = img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Function to make a prediction
def predict_image(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0][0]
    if prediction > 0.5:
        return "PNEUMONIA", prediction
    else:
        return "NORMAL", prediction

# Streamlit UI
st.title("Pneumonia Detection from Chest X-Rays")
st.write("Upload a chest X-ray image to detect if it shows signs of pneumonia.")

# File uploader for user to upload image
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")

    # Predict the uploaded image
    label, confidence = predict_image(uploaded_file)

    # Show progress bar
    progress_percentage = int(confidence * 100)
    st.progress(progress_percentage)

    # Show results
    st.write(f"Prediction: **{label}**")
    st.write(f"Model Confidence: **{confidence:.2f}**")

    # Display additional information about prediction
    if label == "PNEUMONIA":
        st.write("⚠️ **Warning:** The model detected signs of pneumonia in the X-ray image.")
    else:
        st.write("✅ **Good news:** The model did not detect pneumonia in the X-ray image.")