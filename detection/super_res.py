"""
Optional super-resolution preprocessing candidate.

Uses FSRCNN (small, x3) via OpenCV's dnn_superres module — a tiny
(~10KB) pretrained model, free/MIT-licensed, no training required.
This is ONE MORE candidate fed into the existing OCR ensemble in
router.py / test_harness.py — it does not replace or modify the
existing cubic-resize preprocessing in preprocess.py.

Setup (one-time):
    pip install opencv-contrib-python
    (safe alongside opencv-python — both installed is fine)

    Download the model file into detection/models/:
    curl -L -o detection/models/FSRCNN-small_x3.pb \
        https://raw.githubusercontent.com/Saafke/FSRCNN_Tensorflow/master/models/FSRCNN-small_x3.pb

If the model file is missing, maybe_super_resolve() returns None and
the caller should just skip this candidate — it must never crash the
pipeline over a missing optional model.
"""

import os
import cv2

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "FSRCNN-small_x3.pb")
_SCALE = 3

_sr = None
_load_attempted = False

# Below this crop width (px), plain cubic upscaling (already done in
# preprocess.py) starts losing too much stroke detail for OCR — this is
# where a learned SR model has a chance to add something real. Above
# it, cubic is close enough and SR isn't worth the extra runtime.
_SMALL_CROP_WIDTH_THRESHOLD = 200


def _get_sr():
    global _sr, _load_attempted
    if _sr is not None or _load_attempted:
        return _sr

    _load_attempted = True

    if not os.path.exists(_MODEL_PATH):
        print(f"[super_res] WARNING: model not found at {_MODEL_PATH} — "
              f"SR candidate will be skipped for this run. See module "
              f"docstring for the one-line download command.")
        return None

    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(_MODEL_PATH)
        sr.setModel("fsrcnn", _SCALE)
        _sr = sr
    except Exception as e:
        print(f"[super_res] WARNING: failed to load SR model ({e}) — skipping.")

    return _sr


def maybe_super_resolve(color_crop):
    """
    Returns an FSRCNN-upscaled BGR image if the crop is small enough to
    plausibly benefit AND the model is available, else None.

    Callers must treat None as "skip this candidate", not an error —
    this is an optional extra ensemble member, never a required step.
    """
    if color_crop is None or color_crop.size == 0:
        return None

    h, w = color_crop.shape[:2]
    if w >= _SMALL_CROP_WIDTH_THRESHOLD:
        return None

    sr = _get_sr()
    if sr is None:
        return None

    try:
        return sr.upsample(color_crop)
    except Exception as e:
        print(f"[super_res] WARNING: upsample failed ({e}) — skipping.")
        return None