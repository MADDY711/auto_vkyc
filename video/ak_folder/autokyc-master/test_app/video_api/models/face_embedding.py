# # main.py
# import torch
# import cv2
# import numpy as np
# from mobilefacenet import MobileFaceNet

# # Load image and preprocess
# def preprocess(img_path):
#     img = cv2.imread(img_path)
#     img = cv2.resize(img, (112, 112))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = img / 255.0
#     img = (img - 0.5) / 0.5  # Normalize
#     img = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float).unsqueeze(0)
#     return img

# # Load model
# model = MobileFaceNet()
# model.eval()

# # Inference
# img = preprocess("F:\\clg\\internships\\clg_internship\\auto_vkyc\\sample_img.jpg")  # Replace with your image path
# with torch.no_grad():
#     embedding = model(img)
#     print("Face embedding:", embedding.numpy())


import torch
from PIL import Image
from torchvision import transforms
from .mobilefacenet import MobileFaceNet  # Ensure this import matches your project structure

# Load model
model = MobileFaceNet()
model = torch.jit.load("video_api/mobilefacenet_scripted.pt")  # Use relative path if possible
model.eval()

def get_face_embedding(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model(input_tensor)
    return embedding.squeeze(0)
