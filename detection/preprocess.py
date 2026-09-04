import cv2
import numpy as np

from detection.super_res import maybe_super_resolve


def _deskew(gray_img):
    edges = cv2.Canny(gray_img, 50, 150)
    coords = np.column_stack(np.where(edges > 0))

    if len(coords) < 20:
        return gray_img

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]

    if angle < -45:
        angle = 90 + angle

    if abs(angle) > 20:
        return gray_img

    (h, w) = gray_img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        gray_img, M, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def _adaptive_enhance(gray_img):
    mean_brightness = gray_img.mean()

    if mean_brightness < 80:
        clip = 3.5
    elif mean_brightness < 150:
        clip = 2.0
    else:
        clip = 1.2

    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_img)

    if mean_brightness < 90:
        gamma = 1.5
    elif mean_brightness > 180:
        gamma = 0.8
    else:
        gamma = 1.0

    if gamma != 1.0:
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
        enhanced = cv2.LUT(enhanced, table)

    return enhanced


def _variants_from_crop(plate_img):
    """
    The original variant-generation pipeline, unchanged, factored out
    so it can run on either the raw crop or an SR'd copy of it.
    """
    padded = cv2.copyMakeBorder(
        plate_img, top=10, bottom=10, left=15, right=15,
        borderType=cv2.BORDER_REPLICATE
    )

    gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
    gray = _deskew(gray)

    h, w = gray.shape[:2]
    if w < 150:
        scale = 8
    elif w < 300:
        scale = 6
    else:
        scale = 4

    resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(resized, (3, 3), 0)
    enhanced = _adaptive_enhance(blurred)

    variants = [enhanced]

    adaptive = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 8
    )
    kernel_light = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    variants.append(cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel_light, iterations=1))

    _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu)

    adaptive_soft = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )
    variants.append(adaptive_soft)

    adaptive_inv = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 8
    )
    variants.append(cv2.morphologyEx(adaptive_inv, cv2.MORPH_CLOSE, kernel_light, iterations=1))

    _, otsu_inv = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    variants.append(otsu_inv)

    kernel_wide = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 3))
    strong_close = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel_wide, iterations=2)
    variants.append(strong_close)

    gaussian = cv2.GaussianBlur(enhanced, (0, 0), 3)
    sharpened = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
    _, sharp_otsu = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(sharp_otsu)

    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    _, denoised_otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(denoised_otsu)

    (h2, w2) = enhanced.shape[:2]
    center2 = (w2 // 2, h2 // 2)
    M_pos = cv2.getRotationMatrix2D(center2, 3, 1.0)
    nudged = cv2.warpAffine(enhanced, M_pos, (w2, h2), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    _, nudged_otsu = cv2.threshold(nudged, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(nudged_otsu)

    return variants


def preprocess_plate_variants(plate_img):
    variants = _variants_from_crop(plate_img)

    # Extra candidates only for small/low-res crops (e.g. PB01N0050-style
    # cases) — maybe_super_resolve() returns None instantly for crops
    # that are already large enough, or if the SR model isn't installed,
    # so this is a no-op for the common case and never blocks the
    # existing pipeline.
    sr_crop = maybe_super_resolve(plate_img)
    if sr_crop is not None:
        variants.extend(_variants_from_crop(sr_crop))

    return variants


def preprocess_plate(plate_img):
    return preprocess_plate_variants(plate_img)[0]