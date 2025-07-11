import cv2

from resize_img import resize_image
from grayscal_img import convert_to_grayscale
from preprocessor import preprocess_image
from extract_text import extract_text_from_image
from load_img import load_and_fix_orientation



if __name__ == "__main__":
    # Step 0: Read original image
    image_path = r"F:\clg\internships\clg_internship\auto_vkyc\OCR_doc_verifier\Untitled.jpg"
    original_image = cv2.imread(image_path)

    # Step 1: Resize
    resized = resize_image(original_image)


    image = load_and_fix_orientation(image_path)
    resized = resize_image(original_image)


    # Step 2: Grayscale
    grayscale = convert_to_grayscale(resized)

    # Step 3: Preprocessing (contrast, denoise, threshold, morph, deskew)
    final_image = preprocess_image(grayscale)

    # Optional: Save final preprocessed image
    cv2.imwrite("final_preprocessed.png", final_image)

    # Step 4: OCR
    text = extract_text_from_image(final_image)

    print("Extracted Text:\n")
    print(text)
