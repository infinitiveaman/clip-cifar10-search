# utils.py
import torch
import numpy as np
from PIL import Image

def load_saved_data(device):
    image_embeddings = torch.load("image_embeddings.pt").to(device)
    image_labels = np.load("image_labels.npy")
    raw_images = np.load("original_images.npy")
    original_images = [Image.fromarray(img) for img in raw_images]
    return image_embeddings, image_labels, original_images
