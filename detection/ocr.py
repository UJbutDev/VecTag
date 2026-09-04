import easyocr

_reader = easyocr.Reader(['en'], gpu=True)

_PLATE_LENGTH_RANGE = range(8, 11)  # Indian plates are 9-10 chars, allow 8 for partial reads

# Fragments narrower than this fraction of the MEDIAN fragment width in
# the same image get treated as edge artifacts.
_EDGE_ARTIFACT_WIDTH_RATIO = 0.5


def _filter_edge_artifacts(results):
    """
    results: EasyOCR's raw readtext() output, list of (bbox, text, conf).

    Drops small fragments sitting at the LEFT or RIGHT extreme of the
    horizontal ordering, if they're much narrower than the median
    fragment width in this same result set.
    """

    if len(results) <= 2:
        # Not enough fragments to compute a meaningful median safely.
        return results

    def frag_width(r):
        xs = [p[0] for p in r[0]]
        return max(xs) - min(xs)

    def frag_left(r):
        xs = [p[0] for p in r[0]]
        return min(xs)

    widths = sorted(frag_width(r) for r in results)
    median_width = widths[len(widths) // 2]

    ordered = sorted(results, key=frag_left)
    filtered = list(ordered)

    # Remove small artifacts from the left edge.
    while (
            len(filtered) > 1
            and frag_width(filtered[0])
            < median_width * _EDGE_ARTIFACT_WIDTH_RATIO
    ):
        filtered.pop(0)

    # Remove small artifacts from the right edge.
    while (
            len(filtered) > 1
            and frag_width(filtered[-1])
            < median_width * _EDGE_ARTIFACT_WIDTH_RATIO
    ):
        filtered.pop()

    return filtered


def run_ocr(processed_img):
    """
    Runs EasyOCR on a processed plate image.

    Returns:
        (raw_text, confidence)

    confidence is returned as a 0-100 float.
    """

    results = _reader.readtext(
        processed_img,
        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    )

    if not results:
        return "", 0.0

    filtered = _filter_edge_artifacts(results)

    if not filtered:
        # Safety net — never end up with nothing.
        filtered = results

    combined_text = "".join(
        r[1] for r in filtered
    )

    avg_conf = (
            sum(r[2] for r in filtered)
            / len(filtered)
            * 100
    )

    return combined_text, round(avg_conf, 2)


def _candidate_score(text, conf):
    """
    Calculates a score used when selecting between OCR candidates.
    """

    length_bonus = len(text) * 4

    valid_length_bonus = (
        15 if len(text) in _PLATE_LENGTH_RANGE else 0
    )

    return conf + length_bonus + valid_length_bonus


def run_ocr_best_of(processed_variants, debug=True):
    """
    Legacy OCR selection method.

    Kept for manual debugging. The live pipeline currently uses
    synthesize_voted_candidate() instead.
    """

    best_text = ""
    best_conf = 0.0
    best_score = -1

    for i, variant in enumerate(processed_variants):

        text, conf = run_ocr(variant)

        if debug:
            score = (
                _candidate_score(text, conf)
                if text
                else -1
            )

            print(
                f"[ocr debug] variant_{i}: "
                f"text={text!r} "
                f"conf={conf} "
                f"score={score}"
            )

        if not text:
            continue

        score = _candidate_score(text, conf)

        if score > best_score:
            best_text = text
            best_conf = conf
            best_score = score

    if debug:
        print(
            f"[ocr debug] WINNER: "
            f"text={best_text!r} "
            f"conf={best_conf}"
        )

    return best_text, best_conf


def synthesize_voted_candidate(ocr_candidates):
    """
    Selects the strongest OCR candidate from multiple preprocessing
    variants.

    ocr_candidates is expected to contain tuples like:

        [
            ("DL8CBG2956", 91.2),
            ("DL8CBG2956", 87.4),
            ("DL8CBG2956", 94.1),
            ("DL8CBG295G", 82.5)
        ]

    The function gives preference to:

    1. Candidates appearing multiple times.
    2. Candidates with higher average confidence.
    3. Candidates having a plate-like length.

    Returns:
        (voted_text, voted_conf)
    """

    if not ocr_candidates:
        return "", 0.0

    # Remove empty OCR results.
    candidates = [
        (text, conf)
        for text, conf in ocr_candidates
        if text and text.strip()
    ]

    if not candidates:
        return "", 0.0

    # Normalize OCR output.
    normalized_candidates = [
        (
            text.strip().upper(),
            float(conf)
        )
        for text, conf in candidates
    ]

    # Group identical OCR readings.
    groups = {}

    for text, conf in normalized_candidates:

        if text not in groups:
            groups[text] = {
                "count": 0,
                "total_confidence": 0.0
            }

        groups[text]["count"] += 1
        groups[text]["total_confidence"] += conf

    def group_score(item):
        """
        Score an OCR reading based on:

        - Number of times it appeared.
        - Average confidence.
        - Whether its length looks like an Indian plate.
        """

        text, data = item

        count = data["count"]

        average_confidence = (
                data["total_confidence"] / count
        )

        length_bonus = (
            15
            if len(text) in _PLATE_LENGTH_RANGE
            else 0
        )

        # Repeated readings are heavily favored because multiple
        # preprocessing variants agreeing on the same plate is a
        # strong signal.
        vote_score = count * 100

        return (
                vote_score
                + average_confidence
                + length_bonus
        )

    # Find the strongest candidate.
    voted_text, voted_data = max(
        groups.items(),
        key=group_score
    )

    voted_conf = (
            voted_data["total_confidence"]
            / voted_data["count"]
    )

    return voted_text, round(voted_conf, 2)