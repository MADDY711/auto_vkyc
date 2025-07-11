import cv2
import numpy as np
from PIL import Image
import pytesseract

def preprocess_image(image_path, save_processed=False):
    # Load image
    image = cv2.imread(image_path)

    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Resize to enlarge small fonts
    scale_percent = 150  # percent of original size
    width = int(gray.shape[1] * scale_percent / 100)
    height = int(gray.shape[0] * scale_percent / 100)
    gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_LINEAR)

    # Step 3: Apply CLAHE for contrast improvement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)

    # Step 4: Remove noise
    denoised = cv2.fastNlMeansDenoising(contrast, h=30)

    # Step 5: Adaptive Thresholding
    adaptive = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)

    # Step 6: Morphological Closing
    kernel = np.ones((2, 2), np.uint8)
    processed = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)

    # Step 7: Deskew
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

    if save_processed:
        cv2.imwrite("processed_image_enhanced.png", processed)

    return processed

def extract_text_from_image(image):
    # Convert to PIL
    pil_img = Image.fromarray(image)

    # Optional: Use a custom config
   # custom_config = r'--oem 3 --psm 6'  # psm 6 = assume a uniform block of text

   # text = pytesseract.image_to_string(pil_img, config=custom_config)
    text = pytesseract.image_to_string(pil_img)

    return text

if __name__ == "__main__":
    image_path = r"F:\\clg\\internships\\clg_internship\\auto_vkyc\\OCR_doc_verifier\\Untitled.jpg"
    preprocessed_image = preprocess_image(image_path, save_processed=True)
    extracted_text = extract_text_from_image(preprocessed_image)

    print("Extracted Text:\n")
    print(extracted_text)
