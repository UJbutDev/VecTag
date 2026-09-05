import cv2
import numpy as np

from detection.super_res import maybe_super_resolve


# ============================================================
# GENERAL HELPERS
# ============================================================

def _to_gray(image):
    """
    Safely convert grayscale, BGR, or BGRA images to grayscale.
    """

    if image is None:
        return None

    if len(image.shape) == 2:
        return image

    if len(image.shape) != 3:
        return None

    channels = image.shape[2]

    if channels == 1:
        return image[:, :, 0]

    if channels == 3:
        return cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    if channels == 4:
        return cv2.cvtColor(
            image,
            cv2.COLOR_BGRA2GRAY
        )

    return None


def _resize_for_ocr(gray):
    """
    Upscale small plate images to a useful OCR height.
    """

    if gray is None:
        return None

    h, w = gray.shape[:2]

    if h <= 0 or w <= 0:
        return gray

    target_height = 100

    scale = target_height / float(h)

    # Never downscale.
    scale = max(
        1.0,
        scale
    )

    new_w = max(
        120,
        int(round(w * scale))
    )

    new_h = max(
        40,
        int(round(h * scale))
    )

    return cv2.resize(
        gray,
        (new_w, new_h),
        interpolation=cv2.INTER_CUBIC
    )


def _adaptive_enhance(gray):
    """
    Moderate CLAHE/gamma enhancement.
    """

    if gray is None:
        return None

    mean_brightness = float(
        gray.mean()
    )

    if mean_brightness < 80:
        clip = 3.0

    elif mean_brightness < 160:
        clip = 2.0

    else:
        clip = 1.3

    clahe = cv2.createCLAHE(
        clipLimit=clip,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(
        gray
    )

    # Only apply gamma to very dark crops.
    if mean_brightness < 65:

        gamma = 1.35

        inv_gamma = 1.0 / gamma

        table = np.array(
            [
                ((i / 255.0) ** inv_gamma) * 255
                for i in range(256)
            ],
            dtype=np.uint8
        )

        enhanced = cv2.LUT(
            enhanced,
            table
        )

    return enhanced


def _order_points(points):
    """
    Return points in:

        top-left
        top-right
        bottom-right
        bottom-left
    """

    points = np.asarray(
        points,
        dtype=np.float32
    )

    result = np.zeros(
        (4, 2),
        dtype=np.float32
    )

    sums = points.sum(
        axis=1
    )

    diffs = np.diff(
        points,
        axis=1
    ).reshape(-1)

    result[0] = points[
        np.argmin(sums)
    ]

    result[2] = points[
        np.argmax(sums)
    ]

    result[1] = points[
        np.argmin(diffs)
    ]

    result[3] = points[
        np.argmax(diffs)
    ]

    return result


def _rotate_bound(
        image,
        angle
):
    """
    Rotate without clipping the image.
    """

    if image is None:
        return None

    h, w = image.shape[:2]

    center = (
        w / 2.0,
        h / 2.0
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    cos = abs(
        matrix[0, 0]
    )

    sin = abs(
        matrix[0, 1]
    )

    new_w = int(
        (h * sin)
        + (w * cos)
    )

    new_h = int(
        (h * cos)
        + (w * sin)
    )

    matrix[0, 2] += (
            new_w / 2.0
            - center[0]
    )

    matrix[1, 2] += (
            new_h / 2.0
            - center[1]
    )

    return cv2.warpAffine(
        image,
        matrix,
        (
            new_w,
            new_h
        ),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )


# ============================================================
# PLATE MASK GENERATION
# ============================================================

def _plate_masks(
        gray,
        color
):
    """
    Generate masks that may isolate the physical plate.
    """

    masks = []

    if gray is None:
        return masks

    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # --------------------------------------------------------
    # Otsu
    # --------------------------------------------------------

    _, otsu = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY +
        cv2.THRESH_OTSU
    )

    masks.append(
        otsu
    )

    # --------------------------------------------------------
    # Adaptive
    # --------------------------------------------------------

    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        -5
    )

    masks.append(
        adaptive
    )

    # --------------------------------------------------------
    # Bright / low-saturation region
    # --------------------------------------------------------

    if (
            color is not None
            and len(color.shape) == 3
            and color.shape[2] == 3
    ):

        hsv = cv2.cvtColor(
            color,
            cv2.COLOR_BGR2HSV
        )

        lower_white = np.array(
            [0, 0, 125],
            dtype=np.uint8
        )

        upper_white = np.array(
            [180, 110, 255],
            dtype=np.uint8
        )

        white_mask = cv2.inRange(
            hsv,
            lower_white,
            upper_white
        )

        masks.append(
            white_mask
        )

    return masks


# ============================================================
# QUADRILATERAL PLATE DETECTION
# ============================================================

def _find_plate_quad(
        plate_img
):
    """
    Search for a wide quadrilateral representing the physical
    number plate inside the YOLO crop.
    """

    if plate_img is None:
        return None

    gray = _to_gray(
        plate_img
    )

    if gray is None:
        return None

    h, w = gray.shape[:2]

    if w < 80 or h < 30:
        return None

    image_area = float(
        w * h
    )

    contours = []

    masks = _plate_masks(
        gray,
        plate_img
    )

    for mask in masks:

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (9, 5)
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=2
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            np.ones(
                (3, 3),
                dtype=np.uint8
            ),
            iterations=1
        )

        found, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if found:
            contours.extend(
                found
            )

    # --------------------------------------------------------
    # Edge contours
    # --------------------------------------------------------

    edges = cv2.Canny(
        gray,
        40,
        140
    )

    edge_contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if edge_contours:
        contours.extend(
            edge_contours
        )

    if not contours:
        return None

    best_quad = None
    best_score = -1.0

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area < image_area * 0.015:
            continue

        if area > image_area * 0.85:
            continue

        perimeter = cv2.arcLength(
            contour,
            True
        )

        if perimeter <= 0:
            continue

        for epsilon_factor in (
                0.018,
                0.025,
                0.035,
                0.045,
                0.055
        ):

            approx = cv2.approxPolyDP(
                contour,
                epsilon_factor * perimeter,
                True
            )

            if len(approx) != 4:
                continue

            points = approx.reshape(
                4,
                2
            ).astype(
                np.float32
            )

            if not cv2.isContourConvex(
                    points.astype(
                        np.int32
                    )
            ):
                continue

            ordered = _order_points(
                points
            )

            tl, tr, br, bl = ordered

            top_width = np.linalg.norm(
                tr - tl
            )

            bottom_width = np.linalg.norm(
                br - bl
            )

            left_height = np.linalg.norm(
                bl - tl
            )

            right_height = np.linalg.norm(
                br - tr
            )

            avg_width = (
                                top_width
                                + bottom_width
                        ) / 2.0

            avg_height = (
                                 left_height
                                 + right_height
                         ) / 2.0

            if avg_height <= 1:
                continue

            aspect = (
                    avg_width
                    / avg_height
            )

            # Broad range to tolerate perspective distortion.
            if not (
                    2.0 <= aspect <= 8.5
            ):
                continue

            rectangle_area = (
                    avg_width
                    * avg_height
            )

            if rectangle_area <= 0:
                continue

            fill_ratio = (
                    area
                    / rectangle_area
            )

            if fill_ratio < 0.12:
                continue

            aspect_score = 1.0 - min(
                abs(
                    aspect - 4.5
                ) / 4.5,
                1.0
            )

            area_score = min(
                area / image_area,
                0.50
            ) / 0.50

            fill_score = min(
                fill_ratio,
                1.0
            )

            cx = float(
                points[:, 0].mean()
            )

            cy = float(
                points[:, 1].mean()
            )

            center_distance = np.sqrt(
                (
                        (cx - w / 2.0)
                        / max(
                    w / 2.0,
                    1.0
                )
                ) ** 2
                +
                (
                        (cy - h / 2.0)
                        / max(
                    h / 2.0,
                    1.0
                )
                ) ** 2
            )

            center_score = 1.0 - min(
                center_distance,
                1.0
            )

            score = (
                    area_score * 0.30
                    + aspect_score * 0.30
                    + fill_score * 0.25
                    + center_score * 0.15
            )

            if score > best_score:

                best_score = score
                best_quad = ordered

    return best_quad


# ============================================================
# PERSPECTIVE WARP
# ============================================================

def _warp_quad(
        image,
        quad
):
    """
    Perspective rectify a detected plate quadrilateral.
    """

    if image is None:
        return None

    if quad is None:
        return None

    tl, tr, br, bl = quad

    width_top = np.linalg.norm(
        tr - tl
    )

    width_bottom = np.linalg.norm(
        br - bl
    )

    height_left = np.linalg.norm(
        bl - tl
    )

    height_right = np.linalg.norm(
        br - tr
    )

    output_width = int(
        max(
            width_top,
            width_bottom
        )
    )

    output_height = int(
        max(
            height_left,
            height_right
        )
    )

    if output_width < 80:
        return None

    if output_height < 20:
        return None

    output_width = max(
        output_width,
        220
    )

    output_height = max(
        output_height,
        55
    )

    output_height = min(
        output_height,
        int(
            output_width / 2.0
        )
    )

    destination = np.array(
        [
            [0, 0],
            [
                output_width - 1,
                0
            ],
            [
                output_width - 1,
                output_height - 1
            ],
            [
                0,
                output_height - 1
            ],
        ],
        dtype=np.float32
    )

    matrix = cv2.getPerspectiveTransform(
        quad.astype(
            np.float32
        ),
        destination
    )

    return cv2.warpPerspective(
        image,
        matrix,
        (
            output_width,
            output_height
        ),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )


# ============================================================
# ROTATED RECTANGLE EXTRACTION
# ============================================================

def _find_rotated_plate(
        image
):
    """
    Try to detect the plate as a rotated rectangle.
    """

    if image is None:
        return None

    gray = _to_gray(
        image
    )

    if gray is None:
        return None

    h, w = gray.shape[:2]

    if w < 80 or h < 30:
        return None

    image_area = float(
        w * h
    )

    best = None
    best_score = -1.0

    masks = _plate_masks(
        gray,
        image
    )

    for mask in masks:

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (9, 5)
            ),
            iterations=2
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:

            area = cv2.contourArea(
                contour
            )

            if area < image_area * 0.015:
                continue

            if area > image_area * 0.85:
                continue

            rect = cv2.minAreaRect(
                contour
            )

            (
                (cx, cy),
                (rw, rh),
                angle
            ) = rect

            if rw <= 1 or rh <= 1:
                continue

            if rh > rw:

                rw, rh = rh, rw
                angle += 90.0

            aspect = (
                    rw / rh
            )

            if not (
                    2.0 <= aspect <= 8.5
            ):
                continue

            rectangle_area = (
                    rw * rh
            )

            if rectangle_area <= 0:
                continue

            fill_ratio = (
                    area
                    / rectangle_area
            )

            if fill_ratio < 0.15:
                continue

            aspect_score = 1.0 - min(
                abs(
                    aspect - 4.5
                ) / 4.5,
                1.0
            )

            area_score = min(
                area / image_area,
                0.50
            ) / 0.50

            fill_score = min(
                fill_ratio,
                1.0
            )

            center_distance = np.sqrt(
                (
                        (cx - w / 2.0)
                        / max(
                    w / 2.0,
                    1.0
                )
                ) ** 2
                +
                (
                        (cy - h / 2.0)
                        / max(
                    h / 2.0,
                    1.0
                )
                ) ** 2
            )

            center_score = 1.0 - min(
                center_distance,
                1.0
            )

            score = (
                    area_score * 0.30
                    + aspect_score * 0.30
                    + fill_score * 0.25
                    + center_score * 0.15
            )

            if score > best_score:

                best_score = score

                best = (
                    float(cx),
                    float(cy),
                    float(rw),
                    float(rh),
                    float(angle)
                )

    if best is None:
        return None

    cx, cy, rw, rh, angle = best

    # Safety margin.
    rw *= 1.08
    rh *= 1.18

    box = cv2.boxPoints(
        (
            (cx, cy),
            (rw, rh),
            angle
        )
    )

    box = _order_points(
        box
    )

    corrected = _warp_quad(
        image,
        box
    )

    if corrected is None:
        return None

    print(
        "[preprocess debug] "
        f"rotated plate extracted angle={angle:.2f}"
    )

    return corrected


# ============================================================
# HOUGH ROTATION FALLBACK
# ============================================================

def _hough_rotation(
        image
):
    """
    Estimate dominant plate-like rotation using long lines.

    This is a fallback only.
    """

    if image is None:
        return None

    gray = _to_gray(
        image
    )

    if gray is None:
        return None

    h, w = gray.shape[:2]

    if w < 80 or h < 30:
        return None

    edges = cv2.Canny(
        gray,
        50,
        150
    )

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(
            20,
            int(w * 0.10)
        ),
        minLineLength=max(
            25,
            int(w * 0.22)
        ),
        maxLineGap=25
    )

    if lines is None:
        return None

    angles = []
    lengths = []

    for line in lines:

        coords = np.asarray(
            line
        ).reshape(-1)

        if coords.size < 4:
            continue

        x1, y1, x2, y2 = coords[:4]

        dx = x2 - x1
        dy = y2 - y1

        length = np.hypot(
            dx,
            dy
        )

        if length < max(
                25,
                w * 0.18
        ):
            continue

        angle = np.degrees(
            np.arctan2(
                dy,
                dx
            )
        )

        if -30 <= angle <= 30:

            angles.append(
                float(angle)
            )

            lengths.append(
                float(length)
            )

    if not angles:
        return None

    angles = np.asarray(
        angles,
        dtype=np.float32
    )

    lengths = np.asarray(
        lengths,
        dtype=np.float32
    )

    angle = float(
        np.average(
            angles,
            weights=lengths
        )
    )

    if abs(angle) < 2.0:
        return None

    if abs(angle) > 28.0:
        return None

    correction = -angle

    print(
        "[preprocess debug] "
        f"Hough angle={angle:.2f}, "
        f"correction={correction:.2f}"
    )

    return _rotate_bound(
        image,
        correction
    )


# ============================================================
# GEOMETRIC CANDIDATES
# ============================================================

def _geometric_candidates(
        image
):
    """
    Generate genuinely different geometric representations.
    """

    candidates = []

    if image is None:
        return candidates

    candidates.append(
        (
            "original",
            image
        )
    )

    # --------------------------------------------------------
    # Perspective correction
    # --------------------------------------------------------

    try:

        quad = _find_plate_quad(
            image
        )

        if quad is not None:

            corrected = _warp_quad(
                image,
                quad
            )

            if corrected is not None:

                print(
                    "[preprocess debug] "
                    "perspective correction: SUCCESS"
                )

                candidates.append(
                    (
                        "perspective",
                        corrected
                    )
                )

    except Exception as exc:

        print(
            "[preprocess debug] "
            f"perspective correction failed: {exc}"
        )

    # --------------------------------------------------------
    # Rotated rectangle
    # --------------------------------------------------------

    try:

        corrected = _find_rotated_plate(
            image
        )

        if corrected is not None:

            candidates.append(
                (
                    "rotated_rect",
                    corrected
                )
            )

    except Exception as exc:

        print(
            "[preprocess debug] "
            f"rotated rectangle failed: {exc}"
        )

    # --------------------------------------------------------
    # Hough fallback
    # --------------------------------------------------------

    try:

        corrected = _hough_rotation(
            image
        )

        if corrected is not None:

            candidates.append(
                (
                    "hough",
                    corrected
                )
            )

    except Exception as exc:

        print(
            "[preprocess debug] "
            f"Hough correction failed: {exc}"
        )

    return candidates


# ============================================================
# OCR PREPROCESSING
# ============================================================

def _ocr_variants(
        image,
        compact=False
):
    """
    Produce a small set of meaningful OCR representations.
    """

    if image is None:
        return []

    if image.size == 0:
        return []

    gray = _to_gray(
        image
    )

    if gray is None:
        return []

    # --------------------------------------------------------
    # Remove only a tiny outer border.
    # --------------------------------------------------------

    h, w = gray.shape[:2]

    border_y = max(
        2,
        int(h * 0.04)
    )

    border_x = max(
        2,
        int(w * 0.02)
    )

    if (
            h > border_y * 2
            and w > border_x * 2
    ):

        gray = gray[
            border_y:h - border_y,
            border_x:w - border_x
        ]

    gray = _resize_for_ocr(
        gray
    )

    enhanced = _adaptive_enhance(
        gray
    )

    variants = []

    # --------------------------------------------------------
    # 1. Enhanced grayscale
    # --------------------------------------------------------

    variants.append(
        enhanced
    )

    # --------------------------------------------------------
    # 2. Otsu
    # --------------------------------------------------------

    _, otsu = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY +
        cv2.THRESH_OTSU
    )

    variants.append(
        otsu
    )

    # --------------------------------------------------------
    # 3. Adaptive threshold
    # --------------------------------------------------------

    adaptive = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        7
    )

    variants.append(
        adaptive
    )

    # --------------------------------------------------------
    # 4. Sharpened grayscale
    # --------------------------------------------------------

    if not compact:

        blur = cv2.GaussianBlur(
            enhanced,
            (0, 0),
            2
        )

        sharpened = cv2.addWeighted(
            enhanced,
            1.4,
            blur,
            -0.4,
            0
        )

        variants.append(
            sharpened
        )

    return variants


# ============================================================
# IMAGE DEDUPLICATION
# ============================================================

def _deduplicate_images(
        images,
        threshold=3.0
):
    """
    Remove visually near-identical images.

    Supports:

        image
        (label, image)
    """

    unique = []

    for item in images:

        if (
                isinstance(item, tuple)
                and len(item) == 2
        ):

            label, image = item

        else:

            label = None
            image = item

        if image is None:
            continue

        if image.size == 0:
            continue

        gray = _to_gray(
            image
        )

        if gray is None:
            continue

        small = cv2.resize(
            gray,
            (64, 24),
            interpolation=cv2.INTER_AREA
        )

        duplicate = False

        for existing_item in unique:

            if (
                    isinstance(
                        existing_item,
                        tuple
                    )
                    and len(existing_item) == 2
            ):

                existing_image = (
                    existing_item[1]
                )

            else:

                existing_image = (
                    existing_item
                )

            existing_gray = _to_gray(
                existing_image
            )

            if existing_gray is None:
                continue

            existing_small = cv2.resize(
                existing_gray,
                (64, 24),
                interpolation=cv2.INTER_AREA
            )

            difference = float(
                np.mean(
                    cv2.absdiff(
                        small,
                        existing_small
                    )
                )
            )

            if difference < threshold:

                duplicate = True
                break

        if not duplicate:

            unique.append(
                (
                    label,
                    image
                )
            )

    return unique


# ============================================================
# FAST SCAN
# ============================================================

def _fast_variants(
        plate_img
):
    """
    Fast path:

        original
        perspective
        rotated rectangle
        Hough fallback

    followed by a small OCR variant set.
    """

    geometric = _geometric_candidates(
        plate_img
    )

    geometric = _deduplicate_images(
        geometric,
        threshold=3.0
    )

    print(
        "[preprocess debug] "
        f"fast geometric candidates={len(geometric)}"
    )

    variants = []

    for method, image in geometric:

        ocr_images = _ocr_variants(
            image,
            compact=False
        )

        for ocr_image in ocr_images:

            variants.append(
                (
                    method,
                    ocr_image
                )
            )

    variants = _deduplicate_images(
        variants,
        threshold=3.0
    )

    return variants


# ============================================================
# HARD RECOVERY
# ============================================================

def _hard_variants(
        plate_img
):
    """
    Hard recovery.

    IMPORTANT:

    We first try to isolate/rectify the plate.

    Only after those methods do we use the angle sweep.
    """

    sources = []

    # --------------------------------------------------------
    # Original
    # --------------------------------------------------------

    sources.append(
        (
            "original",
            plate_img
        )
    )

    # --------------------------------------------------------
    # Super resolution
    # --------------------------------------------------------

    try:

        sr = maybe_super_resolve(
            plate_img
        )

        if sr is not None:

            print(
                "[preprocess debug] "
                "hard recovery: SR SUCCESS"
            )

            sources.append(
                (
                    "sr",
                    sr
                )
            )

    except Exception as exc:

        print(
            "[preprocess debug] "
            f"SR failed: {exc}"
        )

    # --------------------------------------------------------
    # Geometry on original and SR
    # --------------------------------------------------------

    base_sources = list(
        sources
    )

    for source_name, source in base_sources:

        try:

            geometric = _geometric_candidates(
                source
            )

            for method, corrected in geometric:

                if method == "original":
                    continue

                sources.append(
                    (
                        f"{source_name}_{method}",
                        corrected
                    )
                )

        except Exception as exc:

            print(
                "[preprocess debug] "
                f"hard geometry failed: {exc}"
            )

    # --------------------------------------------------------
    # Deduplicate before sweep.
    # --------------------------------------------------------

    sources = _deduplicate_images(
        sources,
        threshold=3.0
    )

    print(
        "[preprocess debug] "
        f"pre-sweep sources={len(sources)}"
    )

    # --------------------------------------------------------
    # CONTROLLED ANGLE SWEEP
    #
    # Only sweep a small number of the best unique sources.
    # --------------------------------------------------------

    sweep_sources = sources[
        :4
    ]

    sweep_angles = (
        -24,
        -18,
        -12,
        -6,
        6,
        12,
        18,
        24
    )

    for source_name, source in sweep_sources:

        for angle in sweep_angles:

            rotated = _rotate_bound(
                source,
                angle
            )

            if rotated is None:
                continue

            sources.append(
                (
                    f"{source_name}_angle_{angle}",
                    rotated
                )
            )

    # --------------------------------------------------------
    # Remove repeated rotations.
    # --------------------------------------------------------

    sources = _deduplicate_images(
        sources,
        threshold=3.0
    )

    print(
        "[preprocess debug] "
        f"hard geometric candidates={len(sources)}"
    )

    # --------------------------------------------------------
    # Compact OCR representations.
    # --------------------------------------------------------

    variants = []

    for source_name, source in sources:

        ocr_images = _ocr_variants(
            source,
            compact=True
        )

        for ocr_image in ocr_images:

            variants.append(
                (
                    source_name,
                    ocr_image
                )
            )

    variants = _deduplicate_images(
        variants,
        threshold=3.0
    )

    return variants


# ============================================================
# PUBLIC API
# ============================================================

def preprocess_plate_variants(
        plate_img,
        hard=False
):
    """
    Public preprocessing interface.

    hard=False:
        FAST scan.

    hard=True:
        HARD recovery.
    """

    if plate_img is None:
        return []

    if plate_img.size == 0:
        return []

    if hard:

        print(
            "[preprocess debug] "
            "ENTERING HARD RECOVERY"
        )

        variants = _hard_variants(
            plate_img
        )

    else:

        variants = _fast_variants(
            plate_img
        )

    # The router expects plain image objects.
    images = [
        image
        for _, image in variants
    ]

    print(
        "[preprocess debug] "
        f"generated {len(images)} "
        f"unique OCR variants"
    )

    return images


def preprocess_plate(
        plate_img
):
    """
    Convenience function returning the first preprocessing
    result.
    """

    variants = preprocess_plate_variants(
        plate_img
    )

    if not variants:
        return plate_img

    return variants[0]