import re

PLATE_REGEX = re.compile(r'^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$')

LETTER_TO_DIGIT = {
    'O':'0',
    'D':'0',
    'Q':'0',
    'I':'1',
    'L':'1',
    'J':'1',
    'Y':'1',
    'Z':'2',
    'E':'3',
    'A':'4',
    'S':'5',
    'G':'6',
    'B':'8',
    'T':'7'
}

DIGIT_TO_LETTER = {
    '0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', '6': 'G',
}


def clean_text(raw_text: str):
    """Backward-compatible single-answer version — returns the first
    valid candidate if any exist, else the stripped-but-uncorrected text."""
    candidates = clean_text_candidates(raw_text)
    best = candidates[0]
    return best, bool(PLATE_REGEX.match(best))


def clean_text_candidates(raw_text: str):
    """
    Returns ALL plausible cleaned interpretations of raw_text, not just
    one. Previously, when a read had genuine ambiguity (e.g. RTO code
    could be 1 or 2 digits, and BOTH interpretations independently
    validated against the plate regex), the old logic backed off and
    discarded ALL corrections rather than pick between two valid guesses
    — confirmed on JK08H0088: this silently threw away the correct
    interpretation ('JK08H0088') right after computing it internally,
    because 'JK0BH0088' also happened to validate.

    Instead of discarding ambiguous-but-valid interpretations, return
    every one of them as a separate candidate, and let the reference
    database (via match.find_best_match_across_candidates) decide which
    is actually correct — consistent with the same philosophy already
    applied to OCR-variant selection.
    """
    stripped = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
    candidates = []

    if len(stripped) in (9, 10):
        candidates.extend(_fix_by_segments_all(stripped))

    if stripped not in candidates:
        candidates.append(stripped)  # always keep the raw fallback too

    return candidates


def _fix_by_segments_all(text: str):
    """
    Returns every regex-valid segment-correction interpretation of text.
    Applies the always-safe corrections (state code, back-anchored
    vehicle number) once, then tries both RTO-length splits (1 or 2
    digits) for the ambiguous middle segment — collecting EVERY split
    that produces a fully valid plate, instead of requiring uniqueness.
    """
    chars = list(text)

    for i in range(2):
        if chars[i] in DIGIT_TO_LETTER:
            chars[i] = DIGIT_TO_LETTER[chars[i]]

    for i in range(len(chars) - 4, len(chars)):
        if chars[i] in LETTER_TO_DIGIT:
            chars[i] = LETTER_TO_DIGIT[chars[i]]

    middle_start, middle_end = 2, len(chars) - 4
    middle = chars[middle_start:middle_end]
    remaining = len(middle)

    valid = []
    for rto_len in (1, 2):
        series_len = remaining - rto_len
        if not (1 <= series_len <= 3):
            continue

        candidate = middle.copy()
        for i in range(0, rto_len):
            if candidate[i] in LETTER_TO_DIGIT:
                candidate[i] = LETTER_TO_DIGIT[candidate[i]]
        for i in range(rto_len, rto_len + series_len):
            if candidate[i] in DIGIT_TO_LETTER:
                candidate[i] = DIGIT_TO_LETTER[candidate[i]]

        full = "".join(chars[:middle_start] + candidate + chars[middle_end:])
        if PLATE_REGEX.match(full) and full not in valid:
            valid.append(full)

    return valid