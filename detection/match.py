from rapidfuzz import process, fuzz

MATCH_THRESHOLD = 85.0  # tunable


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

    if score >= MATCH_THRESHOLD:
        return reference_plates[idx], score, "matched"

    return None, score, "no_match"