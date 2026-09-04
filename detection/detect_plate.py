from ultralytics import YOLO
import cv2
import numpy as np
import os

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "plate_detector.pt")
_model = YOLO(_MODEL_PATH)

_MARGIN_TRIM = 0.02
_PAD_PX = 8

_STRIP_IND_CHIP = False
_IND_CHIP_WIDTH_FRACTION = 0.10

# TODO: revisit once detection quality is stable — 0.05 risks accepting
# bad/spurious boxes on other images. Not yet re-tuned.
_DEBUG_CONF = 0.05

_LOW_CONF_WARN_THRESHOLD = 0.3


def detect_plate_region(image_path: str):
    """
    Returns a LIST of candidate cropped plate regions (usually 1, but 2
    when YOLO's best box is low-confidence). Confirmed need on
    PB01N0050 (box conf 0.129) — a weak YOLO box was being trusted
    outright even though it was clearly unreliable, and no downstream
    fix (preprocessing, voting, text-cleaning) can recover a plate from
    a crop that doesn't actually contain the full plate. Rather than
    guessing a new confidence cutoff (which risks breaking other
    working cases), a low-confidence YOLO box now gets a second,
    independent crop attempt via the contour fallback, and BOTH crops
    are handed downstream — cross-candidate OCR+DB-matching (already
    built) decides which one, if either, actually reads as a real
    plate. This is the same "let the database arbitrate" principle
    already used for OCR-variant and cleaning-interpretation selection,
    just applied one stage earlier.

    Returns [] if nothing usable was found at all.
    """
    img = cv2.imread(image_path)
    if img is None:
        return []

    results = _model(img, verbose=False, conf=_DEBUG_CONF)
    boxes = results[0].boxes

    crops = []

    if boxes is not None and len(boxes) > 0:
        for b in boxes:
            print(f"[debug] plate box conf={float(b.conf[0]):.3f}")

        best = boxes[boxes.conf.argmax()]
        best_conf = float(best.conf[0])

        x1, y1, x2, y2 = map(int, best.xyxy[0])
        x1, y1, x2, y2 = _trim_margin(x1, y1, x2, y2, img.shape)
        yolo_crop = img[y1:y2, x1:x2]

        if _STRIP_IND_CHIP:
            yolo_crop = _strip_ind_chip(yolo_crop)

        yolo_crop = _pad(yolo_crop)
        if yolo_crop is not None and yolo_crop.size > 0:
            crops.append(yolo_crop)

        if best_conf < _LOW_CONF_WARN_THRESHOLD:
            print(f"[detect_plate] WARNING: low-confidence detection "
                  f"({best_conf:.3f} < {_LOW_CONF_WARN_THRESHOLD}) on {os.path.basename(image_path)} "
                  f"— also trying contour fallback as a second candidate crop.")
            fallback_crop = _contour_fallback(img)
            if fallback_crop is not None:
                fallback_crop = _pad(fallback_crop)
                if fallback_crop is not None and fallback_crop.size > 0:
                    crops.append(fallback_crop)

        return crops

    print(f"[debug] literally zero boxes even at conf={_DEBUG_CONF}")
    fallback_crop = _contour_fallback(img)
    if fallback_crop is not None:
        fallback_crop = _pad(fallback_crop)
        if fallback_crop is not None and fallback_crop.size > 0:
            crops.append(fallback_crop)

    return crops


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


def _pad(cropped):
    if cropped is None or cropped.size == 0:
        return cropped
    return cv2.copyMakeBorder(
        cropped, _PAD_PX, _PAD_PX, _PAD_PX, _PAD_PX,
        cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )


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