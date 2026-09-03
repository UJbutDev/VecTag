import cv2
import numpy as np


def preprocess_plate(plate_img):
    """
    Takes a cropped plate region, returns a processed grayscale
    image tuned to improve OCR accuracy.
    """
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    # Resize up so OCR has more pixels to work with
    scale = 3
    resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(resized)

    # Adaptive threshold for varying lighting
    thresh = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 15
    )

    return thresh