from PIL import Image
import pytesseract

def extract_text_from_image(opencv_image):
    pil_img = Image.fromarray(opencv_image)
   # config = r'--oem 3 --psm 6'
  #  text = pytesseract.image_to_string(pil_img, config=config)
    text = pytesseract.image_to_string(pil_img)
    return text
