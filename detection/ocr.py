import easyocr
import re

_reader = easyocr.Reader(['en'], gpu=False)

# Known non-plate text that legitimately appears on Indian plates but
# isn't the plate number — the IND hologram chip. Filtering these out
# before scoring prevents them from winning best-of-N purely because
# they're short and easy to read cleanly.
_NON_PLATE_TEXT = {"IND", "ND", "IN", "NO"}

_PLATE_LENGTH_RANGE = range(8, 11)  # Indian plates are 9-10 chars, allow 8 for partial reads


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

    # Drop known non-plate fragments (IND chip text) before combining,
    # so they can't dominate the result just by being short and clean.
    filtered = [r for r in results if r[1].upper() not in _NON_PLATE_TEXT]
    if not filtered:
        filtered = results  # fall back to unfiltered if everything got dropped

    combined_text = "".join([r[1] for r in filtered])
    avg_conf = sum(r[2] for r in filtered) / len(filtered) * 100

    return combined_text, round(avg_conf, 2)


def _candidate_score(text, conf):
    """
    Scoring prefers plausible plate-length results over raw confidence
    alone. A short, ultra-clean false positive (e.g. a badge/chip) should
    not beat a longer, correctly-shaped read just because it's "cleaner."
    """
    length_bonus = 30 if len(text) in _PLATE_LENGTH_RANGE else 0
    return conf + length_bonus


def run_ocr_best_of(processed_variants):
    """
    Runs OCR on each candidate preprocessing variant and returns the
    (raw_text, confidence) pair with the best score — not just the
    highest raw confidence, since short clean fragments (like the IND
    chip) can out-score a longer, correct, but slightly messier read.
    """
    best_text, best_conf, best_score = "", 0.0, -1

    for variant in processed_variants:
        text, conf = run_ocr(variant)
        if not text:
            continue
        score = _candidate_score(text, conf)
        if score > best_score:
            best_text, best_conf, best_score = text, conf, score
            print(f"[debug] variant result: text={text!r} conf={conf} score={score}")

    return best_text, best_conf