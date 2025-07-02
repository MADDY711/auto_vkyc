import cv2
import torch
import numpy as np
from torchvision import transforms
from mobilefacenet import MobileFaceNet  # model file
from PIL import Image

# --- 1. Load the model ---
model = MobileFaceNet()
model = torch.jit.load("F:\\clg\\internships\\clg_internship\\auto_vkyc\\video\\video_api\\mobilefacenet_scripted.pt")
model.eval()


# --- 2. Preprocess function ---
def preprocess(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return transform(img).unsqueeze(0)  # Add batch dimension

# --- 3. Capture selfie and save as file ---
def capture_selfie():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot access webcam")
        return None

    print(" Press 's' to take selfie, 'q' to quit")
    cv2.namedWindow("Live", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print(" Frame capture failed")
            break
        cv2.imshow("Live", frame)
        key = cv2.waitKey(1)
        if key == ord('s'):
            cv2.imwrite("selfie.jpg", frame)
            print(" Selfie saved as 'selfie.jpg'")
            break
        elif key == ord('q'):
            print(" Cancelled")
            break

    cap.release()
    cv2.destroyAllWindows()
    return "selfie.jpg"

# --- 4. Similarity function ---
def cosine_similarity(emb1, emb2):
    emb1 = emb1 / torch.norm(emb1)
    emb2 = emb2 / torch.norm(emb2)
    return torch.sum(emb1 * emb2).item()

# --- 5. Main flow ---
if __name__ == "__main__":
    # Get embedding of existing image
    img1 = preprocess("F:\\clg\\internships\\clg_internship\\auto_vkyc\\video\\static\\selfie.jpg")
    with torch.no_grad():
        emb1 = model(img1)

    # Get selfie and embedding
    selfie_path = capture_selfie()
    if selfie_path:
        img2 = preprocess(selfie_path)
        with torch.no_grad():
            emb2 = model(img2)

        # Compare
        similarity = cosine_similarity(emb1, emb2)
        print(f" Cosine Similarity: {similarity:.4f}")
        if similarity > 0.6:
            print(" Face Verified: Match")
        else:
            print("Face Mismatch: Verification Failed")






