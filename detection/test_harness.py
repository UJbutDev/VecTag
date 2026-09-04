"""
Batch test harness — run the detection pipeline over a folder of test
images with known correct plates, and get an aggregate accuracy report.
Not part of the API; run manually during tuning:  python -m detection.test_harness
"""
import os
import csv
import cv2

from detection.detect_plate import detect_plate_region
from detection.preprocess import preprocess_plate
from detection.ocr import run_ocr
from detection.clean_text import clean_text

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

    for filename, expected in labels.items():
        path = os.path.join(TEST_IMAGE_DIR, filename)
        if not os.path.exists(path):
            print(f"⚠️  Missing file: {filename}")
            continue

        crop = detect_plate_region(path)
        if crop is None:
            results.append({
                "file": filename, "expected": expected,
                "raw": None, "cleaned": None, "confidence": None,
                "correct": False, "note": "no_plate_detected"
            })
            continue

        cv2.imwrite(os.path.join(DEBUG_OUTPUT_DIR, f"harness_crop_{filename}"), crop)

        processed = preprocess_plate(crop)
        cv2.imwrite(os.path.join(DEBUG_OUTPUT_DIR, f"harness_processed_{filename}"), processed)

        raw_text, confidence = run_ocr(processed)
        cleaned, is_valid = clean_text(raw_text) if raw_text else (None, False)

        correct = cleaned == expected
        results.append({
            "file": filename, "expected": expected,
            "raw": raw_text, "cleaned": cleaned, "confidence": confidence,
            "correct": correct, "note": "" if correct else "mismatch"
        })

    _print_and_save_report(results)


def _print_and_save_report(results):
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    confs = [r["confidence"] for r in results if r["confidence"] is not None]
    avg_conf = round(sum(confs) / len(confs), 2) if confs else 0.0

    lines = [
        "=" * 55,
        "  DETECTION + OCR — BATCH TEST REPORT",
        "=" * 55,
        f"  Total test images     : {total}",
        f"  Correct (exact match) : {correct}",
        f"  Accuracy               : {round(correct / max(total,1) * 100, 1)}%",
        f"  Average OCR confidence : {avg_conf}%",
        "=" * 55,
        "",
        "  Per-image breakdown:",
        ]
    for r in results:
        status = "✅" if r["correct"] else "❌"
        lines.append(
            f"  {status} {r['file']:20s} expected={r['expected']:12s} "
            f"got={str(r['cleaned']):12s} conf={r['confidence']} {r['note']}"
        )

    report = "\n".join(lines)
    print("\n" + report)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n📝 Report saved → {REPORT_PATH}")


if __name__ == "__main__":
    run_batch()