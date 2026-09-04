import shutil
import uuid
import os
import cv2
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from detection.detect_plate import detect_plate_region
from detection.preprocess import preprocess_plate_variants
from detection.ocr import run_ocr_best_of
from detection.clean_text import clean_text
from detection.match import find_best_match
from detection.db import get_all_reference_plates, camera_exists, insert_detection

router = APIRouter()

UPLOAD_DIR = "uploads"
DEBUG_DIR = "debug_output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

# Set False once you're done tuning — saves a crop + processed image(s) per scan
SAVE_DEBUG_IMAGES = True


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

    plate_crop = detect_plate_region(temp_path)
    if plate_crop is None:
        return {"error": "no_plate_detected"}

    if SAVE_DEBUG_IMAGES:
        cv2.imwrite(os.path.join(DEBUG_DIR, f"crop_{image.filename}"), plate_crop)

    # Multiple threshold strategies tried per image, since no single fixed
    # method has worked well across all real test cases (dotted/embossed
    # plates vs clean flat-printed plates respond very differently).
    variants = preprocess_plate_variants(plate_crop)

    if SAVE_DEBUG_IMAGES:
        for i, v in enumerate(variants):
            cv2.imwrite(os.path.join(DEBUG_DIR, f"processed_{i}_{image.filename}"), v)

    raw_text, confidence = run_ocr_best_of(variants)
    if not raw_text:
        return {"error": "no_plate_detected"}

    cleaned, is_valid = clean_text(raw_text)

    reference_plates = get_all_reference_plates()
    matched_plate, match_score, match_status = find_best_match(cleaned, reference_plates)

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