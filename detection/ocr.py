import easyocr

_reader = easyocr.Reader(['en'], gpu=False)


def run_ocr(processed_img):
    """
    Returns (raw_text, confidence) — confidence as a 0-100 float.
    """
    results = _reader.readtext(
        processed_img,
        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    )

    if not results:
        return "", 0.0

    # Combine all detected text segments, weight confidence by text length
    combined_text = "".join([r[1] for r in results])
    avg_conf = sum(r[2] for r in results) / len(results) * 100

    return combined_text, round(avg_conf, 2)