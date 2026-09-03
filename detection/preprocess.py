import cv2
import numpy as np


def preprocess_plate(plate_img):
    """
    Takes a cropped plate region, returns a processed grayscale
    image tuned to improve OCR accuracy.
    """
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    scale = 4
    resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    blurred = cv2.GaussianBlur(resized, (3, 3), 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    # Adaptive threshold, not Otsu — Otsu applies one global cutoff, which
    # turns any stray border/shadow/frame noise into a solid black blob
    # indistinguishable from a character. Adaptive threshold judges each
    # region locally, which handles uneven lighting (shadows, glare, rain
    # streaks) more gracefully. Smaller block size (15) than before (31)
    # since the crop is now a small, tight plate-only region.
    thresh = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 8
    )

    # Morphological opening: erodes away small stray blobs (border bits,
    # dust, screw glare) that survived thresholding, without eating into
    # solid character strokes.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    return cleaned