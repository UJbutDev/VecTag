import re

PLATE_REGEX = re.compile(r'^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$')

# Used when a character sits where a DIGIT is expected but OCR read a LETTER
LETTER_TO_DIGIT = {
    'O': '0', 'D': '0', 'Q': '0',
    'I': '1', 'L': '1', 'J': '1',   # J->1 added after real testing (this session)
    'Z': '2',
    'E': '3',
    'A': '4',
    'S': '5',
    'G': '6', 'B': '8',
    'T': '7',
}

# Used when a character sits where a LETTER is expected but OCR read a DIGIT
DIGIT_TO_LETTER = {
    '0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', '6': 'G',
}


def clean_text(raw_text: str):
    stripped = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
    corrected = _fix_by_segments(stripped)
    is_valid = bool(PLATE_REGEX.match(corrected))
    return corrected, is_valid


def _fix_by_segments(text: str) -> str:
    """
    Indian plate format: [2 letters state][1-2 digit RTO][1-3 letter series][4 digit number]
    Applies position-aware corrections instead of blind character substitution.
    This is a first-pass heuristic — keep tuning LETTER_TO_DIGIT / DIGIT_TO_LETTER
    based on real misreads you log during testing.
    """
    if len(text) < 8 or len(text) > 10:
        return text  # can't safely guess segment boundaries — leave untouched

    chars = list(text)

    # Positions 0-1: state code, expect letters
    for i in range(0, 2):
        if chars[i] in DIGIT_TO_LETTER:
            chars[i] = DIGIT_TO_LETTER[chars[i]]

    # Positions 2-3: RTO code, expect digits
    for i in range(2, 4):
        if i < len(chars) - 4 and chars[i] in LETTER_TO_DIGIT:
            chars[i] = LETTER_TO_DIGIT[chars[i]]

    # Last 4 positions: vehicle number, expect digits
    for i in range(len(chars) - 4, len(chars)):
        if chars[i] in LETTER_TO_DIGIT:
            chars[i] = LETTER_TO_DIGIT[chars[i]]

    return "".join(chars)