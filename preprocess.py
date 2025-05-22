# preprocess.py
import torch
import clip
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

dataset = CIFAR10(root="./data", train=False, download=True)
images = [preprocess(dataset[i][0]) for i in range(500)]
image_labels = [dataset[i][1] for i in range(500)]
original_images = [np.array(dataset[i][0]) for i in range(500)]

image_batch = torch.stack(images).to(device)

with torch.no_grad():
    image_embeddings = model.encode_image(image_batch).float()
    image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)

# Save to disk
torch.save(image_embeddings.cpu(), "image_embeddings.pt")
np.save("image_labels.npy", np.array(image_labels))
np.save("original_images.npy", np.array(original_images))
