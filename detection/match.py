from rapidfuzz import process, fuzz
from rapidfuzz.distance import Levenshtein

MATCH_THRESHOLD = 85.0  # tunable, applies to 'normal' status plates

# Indian plates are a fixed-format 9-10 chars. A missing/extra character
# from OCR is a structural red flag, not just noise — a plain similarity
# ratio can score deceptively high even when characters are dropped
# (e.g. "M43CC174" vs "MH43CC1745" scored 88.89 despite missing 2 chars).
MAX_LENGTH_DIFF = 1
MAX_EDIT_DISTANCE = 2  # hard cap, regardless of what the ratio score says

# Higher bar before flagging a plate as stolen/blacklisted — a false
# "no_match" just costs a missed log entry; a false "matched" here has
# real consequences, so it needs stronger evidence.
STRICT_STATUS = {"stolen", "blacklisted"}
STRICT_THRESHOLD = 92.0


def find_best_match(cleaned_text: str, reference_plates: list):
    """
    reference_plates: list of dicts from plates_reference table.
    Returns (matched_plate_dict_or_None, score, match_status).
    """
    if not cleaned_text or not reference_plates:
        return None, 0.0, "no_match"

    plate_numbers = [p["plate_number"] for p in reference_plates]

    result = process.extractOne(
        cleaned_text, plate_numbers, scorer=fuzz.ratio
    )

    if result is None:
        return None, 0.0, "no_match"

    matched_text, score, idx = result
    candidate = reference_plates[idx]

    # Structural guard: reject if characters were dropped/added, even if
    # the ratio score looks high. Catches exactly the failure mode above.
    length_diff = abs(len(cleaned_text) - len(matched_text))
    edit_distance = Levenshtein.distance(cleaned_text, matched_text)

    if length_diff > MAX_LENGTH_DIFF or edit_distance > MAX_EDIT_DISTANCE:
        return None, score, "no_match"

    threshold = STRICT_THRESHOLD if candidate.get("status") in STRICT_STATUS else MATCH_THRESHOLD

    if score >= threshold:
        return candidate, score, "matched"

    return None, score, "no_match"