# main.py
import torch
import cv2
import numpy as np
from mobilefacenet import MobileFaceNet

# Load image and preprocess
def preprocess(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = (img - 0.5) / 0.5  # Normalize
    img = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float).unsqueeze(0)
    return img

# Load model
model = MobileFaceNet()
model.eval()

# Inference
img = preprocess("F:\\clg\\internships\\clg_internship\\auto_vkyc\\sample_img.jpg")  # Replace with your image path
with torch.no_grad():
    embedding = model(img)
    print("Face embedding:", embedding.numpy())


