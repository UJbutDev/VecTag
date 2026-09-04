from ultralytics import YOLO
import cv2
import numpy as np
import os

# Plate-specific YOLOv8 model (yasirfaizahmed/license-plate-object-detection,
# fine-tuned on Keremberke's license-plate dataset). Falls back to OpenCV
# contour detection if no plate is found.
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "plate_detector.pt")
_model = YOLO(_MODEL_PATH)

# Shrinks the detected box inward by this fraction on each side before
# cropping. YOLO boxes tend to land right at (or slightly past) the plate
# edge, catching the frame/holder/screws.
_MARGIN_TRIM = 0.05  # 5% inward on each side

# Many HSRP plates have a blue IND chip/hologram badge on the far left,
# which is not part of the plate number. DISABLED for now — needs
# validation across more real samples before trusting it unconditionally.
_STRIP_IND_CHIP = False
_IND_CHIP_WIDTH_FRACTION = 0.10

# TEMP DIAGNOSTIC: lowered from the ultralytics default (0.25) to check
# whether low-confidence plate boxes are being found and silently
# discarded on hard cases (small/angled plates, glare, etc.), vs the
# model genuinely finding nothing at all. Remove/restore once diagnosed.
_DEBUG_CONF = 0.05


def detect_plate_region(image_path: str):
    """
    Returns the cropped plate region as a numpy array, or None if
    no plate-like region found.
    Fallback: if YOLO doesn't find a plate, use OpenCV contour
    detection as a naive rectangle finder.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    results = _model(img, verbose=False, conf=_DEBUG_CONF)
    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:
        for b in boxes:
            print(f"[debug] plate box conf={float(b.conf[0]):.3f}")

        best = boxes[boxes.conf.argmax()]
        x1, y1, x2, y2 = map(int, best.xyxy[0])
        x1, y1, x2, y2 = _trim_margin(x1, y1, x2, y2, img.shape)
        cropped = img[y1:y2, x1:x2]
        if _STRIP_IND_CHIP:
            cropped = _strip_ind_chip(cropped)
        return cropped

    print(f"[debug] literally zero boxes even at conf={_DEBUG_CONF}")
    return _contour_fallback(img)


def _trim_margin(x1, y1, x2, y2, img_shape):
    h_img, w_img = img_shape[:2]
    box_w = x2 - x1
    box_h = y2 - y1

    dx = int(box_w * _MARGIN_TRIM)
    dy = int(box_h * _MARGIN_TRIM)

    x1 = max(0, x1 + dx)
    y1 = max(0, y1 + dy)
    x2 = min(w_img, x2 - dx)
    y2 = min(h_img, y2 - dy)

    return x1, y1, x2, y2


def _strip_ind_chip(cropped):
    h, w = cropped.shape[:2]
    chip_width = int(w * _IND_CHIP_WIDTH_FRACTION)
    return cropped[:, chip_width:]


def _contour_fallback(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    edges = cv2.Canny(blur, 30, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    for c in contours:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 2 < aspect_ratio < 6:
                return img[y:y + h, x:x + w]

    return None