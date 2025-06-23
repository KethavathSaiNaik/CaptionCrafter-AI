# frontend/app.py

import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import os

st.set_page_config(
    page_title="CaptionCrafter-AI",
    page_icon="🖼️",
    layout="centered",
)

BACKEND_URL = os.getenv("CAPTION_API", "http://localhost:8080")  # fallback for local

with st.sidebar:
    st.title("📘 How to Use")
    st.markdown("""
    1. **Upload** a `.jpg`, `.jpeg`, or `.png` image.  
    2. Click **Generate Caption**.  
    3. Read the AI-generated description! 🤖
    """)

st.markdown("<h1 style='text-align: center;'>🖼️ CaptionCrafter-AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Upload an image and get an AI-generated caption</p>", unsafe_allow_html=True)

uploaded = st.file_uploader(" Upload a JPG or PNG image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Preview", use_container_width=True)

    if st.button("✨ Generate Caption"):
        with st.spinner("Model is thinking..."):
            try:
                files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                response = requests.post(f"{BACKEND_URL}/caption", files=files, timeout=90)
                response.raise_for_status()
                caption = response.json().get("caption", "⚠️ No caption returned.")
                st.success(" Caption Generated!")
                st.markdown(f"<div style='text-align: center; font-size: 22px; font-weight: bold;'>{caption}</div>", unsafe_allow_html=True)
            except requests.RequestException as e:
                st.error(f"❌ Error: {e}")
