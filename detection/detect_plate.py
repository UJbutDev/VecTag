from ultralytics import YOLO
import cv2
import numpy as np

# Use a pretrained YOLOv8 model for now; swap for a plate-specific
# fine-tuned model later if you find/train one.
_model = YOLO("yolov8n.pt")


def detect_plate_region(image_path: str):
    """
    Returns the cropped plate region as a numpy array, or None if
    no plate-like region found.
    Fallback: if YOLO doesn't find a plate class, use OpenCV contour
    detection as a naive rectangle finder.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    results = _model(img, verbose=False)
    boxes = results[0].boxes

    if boxes is not None and len(boxes) > 0:
        # Take the box with highest confidence
        best = boxes[boxes.conf.argmax()]
        x1, y1, x2, y2 = map(int, best.xyxy[0])
        return img[y1:y2, x1:x2]

    return _contour_fallback(img)


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
            if 2 < aspect_ratio < 6:  # plausible plate shape
                return img[y:y + h, x:x + w]

    return None