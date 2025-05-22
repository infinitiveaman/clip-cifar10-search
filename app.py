# app.py
import streamlit as st
import torch
import clip
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from utils import load_saved_data

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Load precomputed data
image_embeddings, image_labels, original_images = load_saved_data(device)

st.title("CLIP-based Image & Text Search on CIFAR-10")
choice = st.radio("Choose search type:", ["Image-to-Image", "Text-to-Image"])

if choice == "Image-to-Image":
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB").resize((224, 224))
        st.image(image, caption="Uploaded Image", use_column_width=True)

        image_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            uploaded_embedding = model.encode_image(image_tensor).float()
            uploaded_embedding /= uploaded_embedding.norm(dim=-1, keepdim=True)

        similarities = (uploaded_embedding @ image_embeddings.T).squeeze(0)
        top_indices = similarities.topk(5).indices.cpu().numpy()

        st.write("Top-5 similar CIFAR-10 images:")
        cols = st.columns(5)
        for i, idx in enumerate(top_indices):
            with cols[i]:
                st.image(original_images[idx], caption=class_names[image_labels[idx]])

elif choice == "Text-to-Image":
    query = st.text_input("Enter text query (e.g., 'a dog'):")
    if query:
        text_token = clip.tokenize([query]).to(device)
        with torch.no_grad():
            text_embedding = model.encode_text(text_token).float()
            text_embedding /= text_embedding.norm(dim=-1, keepdim=True)

        similarities = (text_embedding @ image_embeddings.T).squeeze(0)
        top_indices = similarities.topk(5).indices.cpu().numpy()

        st.write(f"Top-5 images matching: **{query}**")
        cols = st.columns(5)
        for i, idx in enumerate(top_indices):
            with cols[i]:
                st.image(original_images[idx], caption=class_names[image_labels[idx]])
