import re

PLATE_REGEX = re.compile(r'^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}$')

# Positional correction rules — tune these based on your real test results
CORRECTIONS = {
    'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8'
}


def clean_text(raw_text: str):
    """
    Returns (cleaned_text, is_valid_format).
    """
    stripped = re.sub(r'[^A-Z0-9]', '', raw_text.upper())

    # Apply positional corrections only where digits are expected
    # (this is a starting point — refine per your actual OCR error patterns)
    corrected = _apply_positional_fix(stripped)

    is_valid = bool(PLATE_REGEX.match(corrected))
    return corrected, is_valid


def _apply_positional_fix(text: str) -> str:
    # Placeholder logic — replace with rules tuned from your sample images
    # e.g. if position 2-3 should be digits but OCR read letters
    chars = list(text)
    for i, ch in enumerate(chars):
        if ch in CORRECTIONS and i in (2, 3, len(chars) - 4, len(chars) - 3, len(chars) - 2, len(chars) - 1):
            chars[i] = CORRECTIONS[ch]
    return "".join(chars)