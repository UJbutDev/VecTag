"""
Batch test harness — run the detection pipeline over a folder of test
images with known correct plates, and get an aggregate accuracy report.
Not part of the API; run manually during tuning:  python -m detection.test_harness
"""
import os
import csv
import cv2

from detection.detect_plate import detect_plate_region
from detection.preprocess import preprocess_plate_variants
from detection.ocr import run_ocr, synthesize_voted_candidate
from detection.match import find_best_match_across_candidates
from detection.db import get_all_reference_plates

TEST_IMAGE_DIR = "test_images"
LABELS_CSV = "test_images/labels.csv"
REPORT_PATH = "debug_output/test_report.txt"
DEBUG_OUTPUT_DIR = "debug_output"


def load_labels():
    labels = {}
    with open(LABELS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) == 2:
                labels[row[0]] = row[1].strip().upper()
    return labels


def run_batch():
    labels = load_labels()
    results = []
    os.makedirs(DEBUG_OUTPUT_DIR, exist_ok=True)

    reference_plates = get_all_reference_plates()
    reference_plate_numbers = {p["plate_number"] for p in reference_plates}

    for filename, expected in labels.items():
        path = os.path.join(TEST_IMAGE_DIR, filename)
        if not os.path.exists(path):
            print(f"⚠️  Missing file: {filename}")
            continue

        # A no_match on a plate that was never added to plates_reference
        # is CORRECT behavior, not a pipeline failure — flagged up front
        # so it's not misread as an accuracy problem.
        in_reference_db = expected in reference_plate_numbers
        if not in_reference_db:
            print(f"ℹ️  Note: {filename}'s expected plate ({expected}) is NOT in "
                  f"plates_reference — a no_match result for it is expected, not a bug.")

        # detect_plate_region now returns a LIST of candidate crops
        # (usually 1, but 2 when YOLO's best box was low-confidence and
        # the contour fallback was also attempted).
        plate_crops = detect_plate_region(path)
        if not plate_crops:
            results.append({
                "file": filename, "expected": expected,
                "raw": None, "cleaned": None, "confidence": None,
                "match_status": "no_plate_detected",
                "correct": False, "match_correct": False,
                "in_reference_db": in_reference_db,
                "note": "no_plate_detected"
            })
            continue

        ocr_candidates = []
        for crop_i, crop in enumerate(plate_crops):
            cv2.imwrite(os.path.join(DEBUG_OUTPUT_DIR, f"harness_crop_{crop_i}_{filename}"), crop)

            variants = preprocess_plate_variants(crop)
            for i, v in enumerate(variants):
                cv2.imwrite(os.path.join(DEBUG_OUTPUT_DIR, f"harness_processed_{crop_i}_{i}_{filename}"), v)

            ocr_candidates.extend(run_ocr(v) for v in variants)

        # Matches router.py: add a synthetic voted candidate from
        # cross-variant character consensus before matching.
        voted_text, voted_conf = synthesize_voted_candidate(ocr_candidates)
        if voted_text:
            ocr_candidates.append((voted_text, voted_conf))

        raw_text, cleaned, confidence, matched_plate, match_score, match_status = \
            find_best_match_across_candidates(ocr_candidates, reference_plates)

        # Two different notions of "correct", reported separately:
        # - correct: cleaned OCR string exactly equals the expected string
        #   (pure OCR+cleaning accuracy, unforgiving)
        # - match_correct: did the pipeline correctly IDENTIFY the vehicle
        #   (matched_plate's plate_number == expected) — what actually
        #   matters for a real ANPR system, tolerant of small OCR slips
        #   within the matching guards, same as what router.py returns.
        correct = cleaned == expected
        match_correct = bool(matched_plate) and matched_plate.get("plate_number") == expected

        results.append({
            "file": filename, "expected": expected,
            "raw": raw_text, "cleaned": cleaned, "confidence": confidence,
            "match_status": match_status,
            "correct": correct, "match_correct": match_correct,
            "in_reference_db": in_reference_db,
            "note": "" if correct else "mismatch"
        })

    _print_and_save_report(results)


def _print_and_save_report(results):
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    match_correct = sum(1 for r in results if r["match_correct"])
    confs = [r["confidence"] for r in results if r["confidence"] is not None]
    avg_conf = round(sum(confs) / len(confs), 2) if confs else 0.0

    lines = [
        "=" * 60,
        "  DETECTION + OCR — BATCH TEST REPORT",
        "=" * 60,
        f"  Total test images          : {total}",
        f"  Exact OCR match             : {correct} ({round(correct / max(total,1) * 100, 1)}%)",
        f"  Correct vehicle ID (match)  : {match_correct} ({round(match_correct / max(total,1) * 100, 1)}%)",
        f"  Average OCR confidence      : {avg_conf}%",
        "=" * 60,
        "",
        "  Per-image breakdown:",
        ]
    for r in results:
        exact = "✅" if r["correct"] else "❌"
        ident = "✅" if r["match_correct"] else "❌"
        db_note = "" if r["in_reference_db"] else " [not in reference DB]"
        lines.append(
            f"  exact={exact} id={ident} {r['file']:20s} expected={r['expected']:12s} "
            f"got={str(r['cleaned']):12s} conf={r['confidence']} "
            f"match={r.get('match_status',''):10s}{db_note} {r['note']}"
        )

    report = "\n".join(lines)
    print("\n" + report)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n📝 Report saved → {REPORT_PATH}")


if __name__ == "__main__":
    run_batch()