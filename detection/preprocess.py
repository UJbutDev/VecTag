import cv2
import numpy as np


def preprocess_plate(plate_img):
    """
    Takes a cropped plate region, returns a processed grayscale
    image tuned to improve OCR accuracy.
    """
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

    # Resize up so OCR has more pixels to work with.
    # Bumped from 3x to 4x since crops are now tight, small plate-only regions
    # (previously the crop included the whole car front, so less zoom was needed).
    scale = 4
    resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Slight blur to reduce noise before thresholding — helps avoid
    # jagged/broken character strokes on the upscaled image.
    blurred = cv2.GaussianBlur(resized, (3, 3), 0)

    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    # Otsu's threshold instead of adaptive threshold — picks the split point
    # automatically based on the image's histogram. Plates are naturally
    # high-contrast (dark text on light background), and now that the crop
    # is tight and small, Otsu tends to give cleaner, less wavy character
    # edges than adaptive threshold, which was tuned for a much larger frame.
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh