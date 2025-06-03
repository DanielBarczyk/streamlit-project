import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

file = st.file_uploader("Upload image:", type=["jpg", "jpeg", "png"])

if file is not None:
    image = Image.open(file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    image_np = np.array(image)
    fig, ax = plt.subplots(figsize=(8, 4))
    color = ('r', 'g', 'b')
    for i, col in enumerate(color):
        hist = cv2.calcHist([image_np], [i], None, [256], [0, 256])
        ax.plot(hist, color=col)
        ax.set_xlim([0, 256])
    ax.set_title("Color Histogram")
    ax.set_xlabel("Pixel Intensity")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    gray2 = np.empty_like(gray)
    cv2.fastNlMeansDenoising(gray, gray2, 4.0, 7, 21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray3 = clahe.apply(gray2)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", use_container_width=True)

    with col2:
        st.image(gray3, caption="Enhanced Image", use_container_width=True)

# x = st.slider("Select a value")
# st.write(x, "squared is", x * x)