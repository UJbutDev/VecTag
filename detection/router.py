import os
import uuid
import cv2

from datetime import datetime

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form,
    HTTPException
)

from detection.detect_plate import (
    detect_plate_region
)

from detection.preprocess import (
    preprocess_plate_variants
)

from detection.ocr import (
    run_ocr,
    synthesize_voted_candidate
)

from detection.rapid_ocr import (
    run_rapid_ocr
)

from detection.match import (
    find_best_match_across_candidates
)

from detection.db import (
    get_all_reference_plates,
    camera_exists,
    insert_detection
)


router = APIRouter()


# ============================================================
# DIRECTORIES
# ============================================================

UPLOAD_DIR = "uploads"

DEBUG_DIR = "debug_output"

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)

os.makedirs(
    DEBUG_DIR,
    exist_ok=True
)


# ============================================================
# DEBUG
# ============================================================

SAVE_DEBUG_IMAGES = True


# ============================================================
# DATABASE MATCH
# ============================================================

def _database_match(
        candidates,
        reference_plates
):
    """
    Call the existing database-aware matcher.
    """

    if not candidates:

        return (
            "",
            "",
            0.0,
            None,
            0.0,
            "no_match"
        )

    return find_best_match_across_candidates(
        candidates,
        reference_plates
    )


# ============================================================
# OCR STAGE
# ============================================================

def _run_ocr_stage(
        variants,
        ocr_candidates,
        reference_plates,
        stage_name,
        crop_i
):
    """
    OCR each variant and perform a DB cross-check immediately
    after useful OCR output.

    Returns:
        matched result or None
    """

    for variant_i, variant in enumerate(
            variants
    ):

        # ----------------------------------------------------
        # Save debug image
        # ----------------------------------------------------

        if SAVE_DEBUG_IMAGES:

            filename = (
                f"processed_{crop_i}_"
                f"{stage_name}_{variant_i}.png"
            )

            try:

                cv2.imwrite(
                    os.path.join(
                        DEBUG_DIR,
                        filename
                    ),
                    variant
                )

            except Exception as exc:

                print(
                    "[router debug] "
                    f"debug image save failed: {exc}"
                )

        # ----------------------------------------------------
        # OCR
        # ----------------------------------------------------

        try:

            text, confidence = run_ocr(
                variant
            )

        except Exception as exc:

            print(
                "[router debug] "
                f"OCR failed: {exc}"
            )

            text = ""
            confidence = 0.0

        ocr_candidates.append(
            (
                text,
                confidence
            )
        )

        candidate_number = (
                len(ocr_candidates) - 1
        )

        print(
            "[router debug] "
            f"candidate_{candidate_number}: "
            f"raw='{text}' "
            f"conf={confidence}"
        )

        # Empty OCR cannot produce a DB match.
        if not text:
            continue

        # ----------------------------------------------------
        # DATABASE CROSS-CHECK
        # ----------------------------------------------------

        try:

            result = _database_match(
                ocr_candidates,
                reference_plates
            )

        except Exception as exc:

            print(
                "[router debug] "
                f"database check failed: {exc}"
            )

            continue

        (
            raw_text,
            cleaned_text,
            final_confidence,
            matched_plate,
            match_score,
            match_status
        ) = result

        if (
                match_status == "matched"
                and matched_plate is not None
        ):

            print(
                "[router debug] "
                "EARLY DATABASE MATCH"
            )

            print(
                "[router debug] "
                f"matched_plate="
                f"'{matched_plate.get('plate_number')}' "
                f"score={match_score}"
            )

            return result

    return None


# ============================================================
# SCAN
# ============================================================

@router.post("/scan")
async def scan_plate(
        image: UploadFile = File(...),
        camera_id: int = Form(...),
        detected_at: str = Form(None)
):

    # ========================================================
    # CAMERA VALIDATION
    # ========================================================

    if not camera_exists(
            camera_id
    ):

        raise HTTPException(
            status_code=400,
            detail="invalid_camera_id"
        )

    # ========================================================
    # SAVE INPUT
    # ========================================================

    temp_filename = (
        f"{uuid.uuid4()}_"
        f"{image.filename}"
    )

    temp_path = os.path.join(
        UPLOAD_DIR,
        temp_filename
    )

    with open(
            temp_path,
            "wb"
    ) as file:

        file.write(
            await image.read()
        )

    # ========================================================
    # YOLO
    # ========================================================

    plate_crops = detect_plate_region(
        temp_path
    )

    if not plate_crops:

        return {
            "error":
                "no_plate_detected",

            "camera_id":
                camera_id
        }

    print(
        "[router debug] "
        f"YOLO returned "
        f"{len(plate_crops)} crop(s)"
    )

    # ========================================================
    # LOAD REFERENCE DB ONCE
    # ========================================================

    reference_plates = (
        get_all_reference_plates()
    )

    # ========================================================
    # OCR CANDIDATES
    # ========================================================

    ocr_candidates = []

    early_match = None

    # ========================================================
    # EACH YOLO CROP
    # ========================================================

    for crop_i, plate_crop in enumerate(
            plate_crops
    ):

        # ----------------------------------------------------
        # Save raw crop
        # ----------------------------------------------------

        if SAVE_DEBUG_IMAGES:

            crop_filename = (
                f"crop_{crop_i}_"
                f"{image.filename}"
            )

            try:

                cv2.imwrite(
                    os.path.join(
                        DEBUG_DIR,
                        crop_filename
                    ),
                    plate_crop
                )

            except Exception as exc:

                print(
                    "[router debug] "
                    f"crop save failed: {exc}"
                )

        # ====================================================
        # RAPIDOCR PATH
        # ====================================================
        #
        # RapidOCR is added to the SAME candidate pool used by
        # detection.match. We do not modify match.py.
        #
        # A strong RapidOCR result can therefore identify a plate
        # before FAST/HARD EasyOCR gets a chance to overwhelm the
        # candidate pool with noisy single-character reads.
        # ====================================================

        print(
            "[router debug] "
            f"starting RapidOCR candidate scan for crop {crop_i}"
        )

        rapid_results = run_rapid_ocr(plate_crop)

        for rapid_i, (
                rapid_text,
                rapid_confidence,
                rapid_box
        ) in enumerate(rapid_results):

            if not rapid_text:
                continue

            print(
                "[rapidocr] "
                f"candidate text='{rapid_text}' "
                f"conf={rapid_confidence}"
            )

            if rapid_box is not None:
                print(
                    "[rapidocr] "
                    f"box={rapid_box}"
                )

            # Feed RapidOCR directly into the same matcher input
            # format as EasyOCR: (raw_text, confidence).
            ocr_candidates.append(
                (
                    rapid_text,
                    rapid_confidence
                )
            )

            rapid_candidate_number = (
                    len(ocr_candidates) - 1
            )

            print(
                "[router debug] "
                f"candidate_{rapid_candidate_number}: "
                f"RapidOCR raw='{rapid_text}' "
                f"conf={rapid_confidence}"
            )

            # ----------------------------------------------------
            # Immediate database cross-check.
            # ----------------------------------------------------
            #
            # This is intentionally the existing matcher.
            # We do NOT add RapidOCR-specific fuzzy rules here.
            # ----------------------------------------------------

            try:
                rapid_match = _database_match(
                    [(
                        rapid_text,
                        rapid_confidence
                    )],
                    reference_plates
                )
            except Exception as exc:
                print(
                    "[router debug] "
                    f"RapidOCR database check failed: {exc}"
                )
                rapid_match = None

            if rapid_match is not None:
                (
                    rapid_raw_text,
                    rapid_cleaned_text,
                    rapid_final_confidence,
                    rapid_matched_plate,
                    rapid_match_score,
                    rapid_match_status
                ) = rapid_match

                if (
                        rapid_match_status == "matched"
                        and rapid_matched_plate is not None
                ):
                    print(
                        "[router debug] "
                        "EARLY RAPIDOCR DATABASE MATCH"
                    )

                    print(
                        "[router debug] "
                        f"matched_plate="
                        f"'{rapid_matched_plate.get('plate_number')}' "
                        f"score={rapid_match_score}"
                    )

                    early_match = rapid_match
                    break

        if early_match is not None:
            break

        # ====================================================
        # FAST PATH
        # ====================================================

        print(
            "[router debug] "
            f"starting FAST scan for crop {crop_i}"
        )

        fast_variants = (
            preprocess_plate_variants(
                plate_crop,
                hard=False
            )
        )

        print(
            "[router debug] "
            f"FAST variants={len(fast_variants)}"
        )

        early_match = _run_ocr_stage(
            fast_variants,
            ocr_candidates,
            reference_plates,
            "fast",
            crop_i
        )

        if early_match is not None:
            break

        # ====================================================
        # HARD PATH
        # ====================================================

        print(
            "[router debug] "
            f"FAST exhausted for crop {crop_i}"
        )

        print(
            "[router debug] "
            "starting HARD recovery"
        )

        hard_variants = (
            preprocess_plate_variants(
                plate_crop,
                hard=True
            )
        )

        print(
            "[router debug] "
            f"HARD variants={len(hard_variants)}"
        )

        early_match = _run_ocr_stage(
            hard_variants,
            ocr_candidates,
            reference_plates,
            "hard",
            crop_i
        )

        if early_match is not None:
            break

    # ========================================================
    # OCR FAILED COMPLETELY
    # ========================================================

    if not any(
            text
            for text, _ in ocr_candidates
    ):

        return {
            "error":
                "ocr_failed",

            "camera_id":
                camera_id
        }

    # ========================================================
    # EXISTING OCR VOTING
    # ========================================================

    voted_text, voted_conf = (
        synthesize_voted_candidate(
            ocr_candidates
        )
    )

    print(
        "[router debug] "
        f"voted: raw='{voted_text}' "
        f"conf={voted_conf}"
    )

    if voted_text:

        ocr_candidates.append(
            (
                voted_text,
                voted_conf
            )
        )

    # ========================================================
    # FINAL DATABASE MATCH
    # ========================================================

    (
        raw_text,
        cleaned_text,
        confidence,
        matched_plate,
        match_score,
        match_status
    ) = _database_match(
        ocr_candidates,
        reference_plates
    )

    # ========================================================
    # EARLY MATCH SAFETY
    #
    # If a strong early match existed but the voted result
    # doesn't reproduce it, keep the already validated match.
    # ========================================================

    if (
            early_match is not None
            and (
            match_status != "matched"
            or matched_plate is None
    )
    ):

        (
            raw_text,
            cleaned_text,
            confidence,
            matched_plate,
            match_score,
            match_status
        ) = early_match

    # ========================================================
    # DEBUG SUMMARY
    # ========================================================

    for i, (
            text,
            conf
    ) in enumerate(
        ocr_candidates
    ):

        print(
            "[router debug] "
            f"candidate_{i}: "
            f"raw={text!r} "
            f"conf={conf}"
        )

    print(
        "[router debug] "
        f"CHOSEN: "
        f"raw={raw_text!r} "
        f"cleaned={cleaned_text!r} "
        f"score={match_score} "
        f"status={match_status}"
    )

    # ========================================================
    # SAVE DETECTION
    # ========================================================

    row = {
        "raw_text":
            raw_text,

        "cleaned_text":
            cleaned_text,

        "confidence":
            confidence,

        "camera_id":
            camera_id,

        "matched_plate_id":
            (
                matched_plate["id"]
                if matched_plate
                else None
            ),

        "match_score":
            match_score,

        "match_status":
            match_status,

        "detected_at":
            (
                    detected_at
                    or datetime.utcnow().isoformat()
            ),

        "image_path":
            temp_path
    }

    inserted = insert_detection(
        row
    )

    # ========================================================
    # API RESPONSE
    # ========================================================

    return {
        "raw_text":
            raw_text,

        "cleaned_text":
            cleaned_text,

        "confidence":
            confidence,

        "camera_id":
            camera_id,

        "match_status":
            match_status,

        "matched_plate":
            (
                {
                    "plate_number":
                        matched_plate[
                            "plate_number"
                        ],

                    "owner_name":
                        matched_plate.get(
                            "owner_name"
                        ),

                    "status":
                        matched_plate.get(
                            "status"
                        )
                }
                if matched_plate
                else None
            ),

        "match_score":
            match_score,

        "detection_id":
            (
                inserted["id"]
                if inserted
                else None
            )
    }