import cv2
import numpy as np
from PIL import Image
import pytesseract

def preprocess_image(image_path, save_processed=False):
    # Load image
    image = cv2.imread(image_path)

    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Denoise the image
    denoised = cv2.fastNlMeansDenoising(gray, h=30)

    # Step 3: Thresholding (Binarization)
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 4: Morphological operations to remove small noise
    kernel = np.ones((1, 1), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Optional Step: Deskew image (auto-rotate)
    coords = np.column_stack(np.where(processed > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = processed.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    processed = cv2.warpAffine(processed, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Optionally save the preprocessed image
    if save_processed:
        cv2.imwrite("processed_image.png", processed)

    return processed

def extract_text_from_image(image):
    # If using OpenCV image, convert to PIL Image first
    pil_img = Image.fromarray(image)
    text = pytesseract.image_to_string(pil_img)
    return text

if __name__ == "__main__":
    image_path = "your_image_path_here.jpg"  # Replace with your document image path
    preprocessed_image = preprocess_image(image_path, save_processed=True)
    extracted_text = extract_text_from_image(preprocessed_image)
    
    print("Extracted Text:\n")
    print(extracted_text)
