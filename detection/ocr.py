import easyocr
from collections import Counter

_reader = easyocr.Reader(['en'], gpu=True)

_PLATE_LENGTH_RANGE = range(8, 11)

_EDGE_WIDTH_RATIO_THRESHOLD = 0.5
_MAX_STRIP_PER_SIDE = 2


def _bbox_width(bbox):
    xs = [p[0] for p in bbox]
    return max(xs) - min(xs)


def _bbox_center_x(bbox):
    xs = [p[0] for p in bbox]
    return (max(xs) + min(xs)) / 2


def _filter_edge_artifacts(results):
    if len(results) <= 1:
        return results

    widths = sorted(_bbox_width(r[0]) for r in results)
    median_width = widths[len(widths) // 2]
    threshold = median_width * _EDGE_WIDTH_RATIO_THRESHOLD

    ordered = sorted(results, key=lambda r: _bbox_center_x(r[0]))

    stripped = 0
    while (len(ordered) > 1 and stripped < _MAX_STRIP_PER_SIDE
           and _bbox_width(ordered[0][0]) < threshold):
        ordered.pop(0)
        stripped += 1

    stripped = 0
    while (len(ordered) > 1 and stripped < _MAX_STRIP_PER_SIDE
           and _bbox_width(ordered[-1][0]) < threshold):
        ordered.pop()
        stripped += 1

    return ordered


def run_ocr(processed_img):
    results = _reader.readtext(
        processed_img,
        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
        text_threshold=0.5,
        low_text=0.3,
    )

    if not results:
        return "", 0.0

    filtered = _filter_edge_artifacts(results)
    if not filtered:
        filtered = results

    combined_text = "".join([r[1] for r in filtered])
    avg_conf = sum(r[2] for r in filtered) / len(filtered) * 100

    return combined_text, round(avg_conf, 2)


def synthesize_voted_candidate(ocr_candidates, plate_length_range=_PLATE_LENGTH_RANGE):
    usable = [(t, c) for t, c in ocr_candidates if t and len(t) in plate_length_range]
    if len(usable) < 2:
        return "", 0.0

    target_length, _ = Counter(len(t) for t, _ in usable).most_common(1)[0]
    same_length = [(t, c) for t, c in usable if len(t) == target_length]
    if len(same_length) < 2:
        return "", 0.0

    voted_chars = []
    for i in range(target_length):
        tally = Counter()
        for text, conf in same_length:
            tally[text[i]] += max(conf, 1.0)
        best_char, _ = tally.most_common(1)[0]
        voted_chars.append(best_char)

    voted_text = "".join(voted_chars)
    avg_conf = sum(c for _, c in same_length) / len(same_length)
    return voted_text, avg_conf


def _candidate_score(text, conf):
    length_bonus = len(text) * 4
    valid_length_bonus = 15 if len(text) in _PLATE_LENGTH_RANGE else 0
    return conf + length_bonus + valid_length_bonus


def run_ocr_best_of(processed_variants, debug=True):
    """LEGACY — not used by the live pipeline. Kept for manual debug."""
    best_text, best_conf, best_score = "", 0.0, -1
    for i, variant in enumerate(processed_variants):
        text, conf = run_ocr(variant)
        if debug:
            score = _candidate_score(text, conf) if text else -1
            print(f"[ocr debug] variant_{i}: text={text!r} conf={conf} score={score}")
        if not text:
            continue
        score = _candidate_score(text, conf)
        if score > best_score:
            best_text, best_conf, best_score = text, conf, score
    if debug:
        print(f"[ocr debug] WINNER: text={best_text!r} conf={best_conf}")
    return best_text, best_conf