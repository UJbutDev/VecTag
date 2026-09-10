from rapidocr import RapidOCR

_engine = None


def _get_engine():
    global _engine

    if _engine is None:
        print("[rapidocr] initializing RapidOCR...")
        _engine = RapidOCR()
        print("[rapidocr] RapidOCR ready.")

    return _engine


def run_rapid_ocr(image):
    """
    Run RapidOCR on a YOLO plate crop.

    Returns:
        list of (text, confidence, box)

    Confidence is returned on a 0-100 scale so it matches
    the confidence convention already used by EasyOCR.
    """
    if image is None or image.size == 0:
        return []

    try:
        engine = _get_engine()
        result = engine(image)
    except Exception as exc:
        print(f"[rapidocr] OCR failed: {exc}")
        return []

    if result is None:
        return []

    texts = getattr(result, "txts", None)
    scores = getattr(result, "scores", None)
    boxes = getattr(result, "boxes", None)

    if texts is None:
        return []

    outputs = []

    for index, text in enumerate(texts):
        if text is None:
            continue

        text = str(text).strip().upper()

        if not text:
            continue

        confidence = 0.0

        if scores is not None and index < len(scores):
            try:
                confidence = float(scores[index]) * 100.0
            except (TypeError, ValueError):
                confidence = 0.0

        box = None

        if boxes is not None and index < len(boxes):
            box = boxes[index]

        outputs.append(
            (
                text,
                round(confidence, 2),
                box
            )
        )

    return outputs
