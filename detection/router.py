import shutil
import uuid
import os
import cv2
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from detection.detect_plate import detect_plate_region
from detection.preprocess import preprocess_plate_variants
from detection.ocr import run_ocr, synthesize_voted_candidate
from detection.match import find_best_match_across_candidates
from detection.db import get_all_reference_plates, camera_exists, insert_detection

router = APIRouter()

UPLOAD_DIR = "uploads"
DEBUG_DIR = "debug_output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

SAVE_DEBUG_IMAGES = True
DEBUG_LOG = True


@router.post("/scan")
async def scan_plate(
        image: UploadFile = File(...),
        camera_id: int = Form(...),
        detected_at: str = Form(None)
):
    if not camera_exists(camera_id):
        raise HTTPException(status_code=400, detail="invalid camera_id")

    temp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{image.filename}")
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    # detect_plate_region now returns a LIST of candidate crops (usually
    # 1, but 2 when YOLO's best box was low-confidence and the contour
    # fallback was also tried). Cross-candidate OCR+DB-matching decides
    # which crop, if either, actually produced a real plate reading.
    plate_crops = detect_plate_region(temp_path)
    if not plate_crops:
        return {"error": "no_plate_detected"}

    ocr_candidates = []
    for crop_i, plate_crop in enumerate(plate_crops):
        if SAVE_DEBUG_IMAGES:
            cv2.imwrite(os.path.join(DEBUG_DIR, f"crop_{crop_i}_{image.filename}"), plate_crop)

        variants = preprocess_plate_variants(plate_crop)

        if SAVE_DEBUG_IMAGES:
            for i, v in enumerate(variants):
                cv2.imwrite(os.path.join(DEBUG_DIR, f"processed_{crop_i}_{i}_{image.filename}"), v)

        ocr_candidates.extend(run_ocr(v) for v in variants)

    voted_text, voted_conf = synthesize_voted_candidate(ocr_candidates)
    if voted_text:
        ocr_candidates.append((voted_text, voted_conf))

    reference_plates = get_all_reference_plates()

    raw_text, cleaned, confidence, matched_plate, match_score, match_status = \
        find_best_match_across_candidates(ocr_candidates, reference_plates)

    if DEBUG_LOG:
        for i, (text, conf) in enumerate(ocr_candidates):
            label = "voted" if voted_text and i == len(ocr_candidates) - 1 else f"candidate_{i}"
            print(f"[router debug] {label}: raw={text!r} conf={conf}")
        print(f"[router debug] CHOSEN: raw={raw_text!r} cleaned={cleaned!r} "
              f"score={match_score} status={match_status}")

    if not raw_text:
        return {"error": "no_plate_detected"}

    row = {
        "raw_text": raw_text,
        "cleaned_text": cleaned,
        "confidence": confidence,
        "camera_id": camera_id,
        "matched_plate_id": matched_plate["id"] if matched_plate else None,
        "match_score": match_score,
        "match_status": match_status,
        "detected_at": detected_at or datetime.utcnow().isoformat(),
        "image_path": temp_path
    }

    inserted = insert_detection(row)

    response = {
        "raw_text": raw_text,
        "cleaned_text": cleaned,
        "confidence": confidence,
        "camera_id": camera_id,
        "match_status": match_status,
        "matched_plate": {
            "plate_number": matched_plate["plate_number"],
            "owner_name": matched_plate.get("owner_name"),
            "status": matched_plate.get("status")
        } if matched_plate else None,
        "match_score": match_score,
        "detection_id": inserted["id"] if inserted else None
    }

    return response