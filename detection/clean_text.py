import re

PLATE_REGEX = re.compile(r'^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$')

LETTER_TO_DIGIT = {
    'O': '0', 'D': '0', 'Q': '0',
    'I': '1', 'L': '1', 'J': '1',
    'Z': '2', 'E': '3', 'A': '4',
    'S': '5', 'G': '6', 'B': '8', 'T': '7',
}

DIGIT_TO_LETTER = {
    '0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', '6': 'G',
}

# Real, published list of Indian state/UT RTO codes -- not a guess from
# one test sample. Since only these ~36 codes are ever legitimately
# valid, a read that doesn't match one of them is DEFINITELY wrong,
# which makes correcting toward the nearest valid one a safe, grounded
# operation (unlike guessing between two visually-similar letters with
# no external constraint to check against).
VALID_STATE_CODES = {
    'AP', 'AR', 'AS', 'BR', 'CG', 'CH', 'DD', 'DL', 'DN', 'GA', 'GJ',
    'HP', 'HR', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD', 'MH', 'ML', 'MN',
    'MP', 'MZ', 'NL', 'OD', 'PB', 'PY', 'RJ', 'SK', 'TN', 'TR', 'TS',
    'UK', 'UP', 'WB',
}


def _correct_state_code(code: str) -> str:
    """
    If `code` is already valid, leave it untouched. If not, find the
    single valid code with edit distance 1 from it -- if there's
    EXACTLY one such candidate, use it (confident correction). If zero
    or multiple valid codes are equally close, leave it as OCR read it
    rather than guess (same "don't force an ambiguous call" principle
    already used for the RTO/series segment).
    """
    if code in VALID_STATE_CODES:
        return code

    from rapidfuzz.distance import Levenshtein
    close = [c for c in VALID_STATE_CODES if Levenshtein.distance(code, c) == 1]

    if len(close) == 1:
        return close[0]
    return code


def clean_text(raw_text: str):
    candidates = clean_text_candidates(raw_text)
    if candidates:
        return candidates[0], True
    stripped = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
    return stripped, bool(PLATE_REGEX.match(stripped))


def clean_text_candidates(raw_text: str):
    """
    Returns EVERY regex-valid segment-correction interpretation of
    raw_text. State code (0-1) now gets a validity-checked correction
    (not just type-correction) since Indian state codes are a known
    finite set. Vehicle number (last 4, back-anchored) keeps its
    unconditional type-correction as before. Middle segment (RTO+series)
    unchanged from the prior version -- tries both RTO-length splits,
    returns every regex-valid result.
    """
    stripped = re.sub(r'[^A-Z0-9]', '', raw_text.upper())

    if len(stripped) < 8 or len(stripped) > 10:
        return []

    chars = list(stripped)

    for i in range(2):
        if chars[i] in DIGIT_TO_LETTER:
            chars[i] = DIGIT_TO_LETTER[chars[i]]
    chars[0:2] = list(_correct_state_code("".join(chars[0:2])))

    for i in range(len(chars) - 4, len(chars)):
        if chars[i] in LETTER_TO_DIGIT:
            chars[i] = LETTER_TO_DIGIT[chars[i]]

    middle_start, middle_end = 2, len(chars) - 4
    if middle_start >= middle_end:
        return []

    middle = chars[middle_start:middle_end]
    remaining = len(middle)

    candidates = set()
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
        if PLATE_REGEX.match(full):
            candidates.add(full)

    return list(candidates)