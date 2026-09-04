import cv2
import numpy as np


def preprocess_plate_variants(plate_img):
    """
    Returns a list of differently-processed candidate images for the same
    plate crop. Different real-world plates (dotted/embossed vs flat-printed,
    uneven lighting vs clean/even lighting) respond better to different
    threshold strategies — no single fixed method has worked well across
    all real test cases so far. Caller should run OCR on each and keep
    whichever gives the best result, rather than committing to one variant.
    """
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    scale = 4
    resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(resized, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)

    variants = []

    # Variant A: adaptive threshold + closing — tuned for embossed/dotted
    # plates and uneven lighting (glare, shadow).
    adaptive = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 8
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    variants.append(cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=1))

    # Variant B: Otsu threshold, no morphology — tends to work better on
    # clean, high-contrast, evenly-lit flat-printed plates, where adaptive
    # threshold's local windowing can fragment otherwise-solid strokes.
    _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu)

    # Variant C: larger-block adaptive threshold, no morphology — a
    # gentler middle ground between A and B.
    adaptive_soft = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )
    variants.append(adaptive_soft)

    return variants


def preprocess_plate(plate_img):
    """
    Kept for backward compatibility — returns just the first (adaptive+
    closing) variant. Prefer preprocess_plate_variants() + best-of-N OCR
    in new code.
    """
    return preprocess_plate_variants(plate_img)[0]