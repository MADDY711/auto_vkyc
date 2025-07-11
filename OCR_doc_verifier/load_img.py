from PIL import Image, ExifTags
import cv2
import numpy as np

def load_and_fix_orientation(image_path):
    img_pil = Image.open(image_path)

    # Fix orientation using EXIF (if present)
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(img_pil._getexif().items())

        if exif.get(orientation) == 3:
            img_pil = img_pil.rotate(180, expand=True)
        elif exif.get(orientation) == 6:
            img_pil = img_pil.rotate(270, expand=True)
        elif exif.get(orientation) == 8:
            img_pil = img_pil.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        # No EXIF data
        pass

    # Convert to OpenCV format
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img_cv
