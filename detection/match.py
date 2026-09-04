from rapidfuzz import process, fuzz
from rapidfuzz.distance import Levenshtein

from detection.clean_text import clean_text_candidates

MATCH_THRESHOLD = 85.0
MAX_LENGTH_DIFF = 1
MAX_EDIT_DISTANCE = 2
STRICT_STATUS = {"stolen", "blacklisted"}
STRICT_THRESHOLD = 92.0


def find_best_match(cleaned_text: str, reference_plates: list):
    if not cleaned_text or not reference_plates:
        return None, 0.0, "no_match"

    plate_numbers = [p["plate_number"] for p in reference_plates]
    result = process.extractOne(cleaned_text, plate_numbers, scorer=fuzz.ratio)

    if result is None:
        return None, 0.0, "no_match"

    matched_text, score, idx = result
    candidate = reference_plates[idx]

    length_diff = abs(len(cleaned_text) - len(matched_text))
    edit_distance = Levenshtein.distance(cleaned_text, matched_text)

    if length_diff > MAX_LENGTH_DIFF or edit_distance > MAX_EDIT_DISTANCE:
        return None, score, "no_match"

    threshold = STRICT_THRESHOLD if candidate.get("status") in STRICT_STATUS else MATCH_THRESHOLD

    if score >= threshold:
        return candidate, score, "matched"

    return None, score, "no_match"


def find_best_match_across_candidates(ocr_candidates: list, reference_plates: list):
    best_matched = None
    best_fallback = None

    for raw_text, conf in ocr_candidates:
        if not raw_text:
            continue

        cleaned_options = clean_text_candidates(raw_text)
        if not cleaned_options:
            import re
            cleaned_options = [re.sub(r'[^A-Z0-9]', '', raw_text.upper())]

        for cleaned in cleaned_options:
            matched_plate, score, status = find_best_match(cleaned, reference_plates)
            result = (raw_text, cleaned, conf, matched_plate, score, status)

            if status == "matched" and (best_matched is None or score > best_matched[4]):
                best_matched = result
            if best_fallback is None or score > best_fallback[4]:
                best_fallback = result

    if best_matched:
        return best_matched
    if best_fallback:
        return best_fallback
    return "", "", 0.0, None, 0.0, "no_match"