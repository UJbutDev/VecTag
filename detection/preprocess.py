import cv2
import numpy as np

from detection.super_res import maybe_super_resolve


def _deskew(gray_img):
    edges = cv2.Canny(gray_img, 50, 150)
    coords = np.column_stack(np.where(edges > 0))

    if len(coords) < 20:
        return gray_img

    rect = cv2.minAreaRect(coords)
    angle = rect[-1]

    if angle < -45:
        angle = 90 + angle

    if abs(angle) > 20:
        return gray_img

    (h, w) = gray_img.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, -angle, 1.0)

    return cv2.warpAffine(
        gray_img,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )


def _adaptive_enhance(gray_img):
    mean_brightness = gray_img.mean()

    if mean_brightness < 80:
        clip = 3.5
    elif mean_brightness < 150:
        clip = 2.0
    else:
        clip = 1.2

    clahe = cv2.createCLAHE(
        clipLimit=clip,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray_img)

    if mean_brightness < 90:
        gamma = 1.5
    elif mean_brightness > 180:
        gamma = 0.8
    else:
        gamma = 1.0

    if gamma != 1.0:
        inv_gamma = 1.0 / gamma

        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in range(256)
        ]).astype("uint8")

        enhanced = cv2.LUT(enhanced, table)

    return enhanced


def _order_points(points):
    """
    Return points in:
        top-left
        top-right
        bottom-right
        bottom-left
    order.
    """

    points = np.asarray(
        points,
        dtype=np.float32
    )

    result = np.zeros(
        (4, 2),
        dtype=np.float32
    )

    sums = points.sum(axis=1)
    diffs = np.diff(
        points,
        axis=1
    ).reshape(-1)

    result[0] = points[np.argmin(sums)]   # top-left
    result[2] = points[np.argmax(sums)]   # bottom-right
    result[1] = points[np.argmin(diffs)]  # top-right
    result[3] = points[np.argmax(diffs)]  # bottom-left

    return result


def _perspective_correct(plate_img):
    """
    Straighten a tilted license plate.

    First tries to detect the plate rectangle.
    If that fails, falls back to detecting the dominant
    long horizontal edge angle and rotating the crop.

    Returns:
        corrected image, or None
    """

    if plate_img is None or plate_img.size == 0:
        return None

    h, w = plate_img.shape[:2]

    if w < 80 or h < 30:
        return None

    gray = cv2.cvtColor(
        plate_img,
        cv2.COLOR_BGR2GRAY
    )

    # ---------------------------------------------------------
    # 1. TRY TO FIND THE WHITE PLATE AS A QUADRILATERAL
    # ---------------------------------------------------------

    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # White/light plate region
    _, thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Connect the plate area
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (15, 7)
    )

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    image_area = float(w * h)

    best_quad = None
    best_score = 0.0

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < image_area * 0.03:
            continue

        if area > image_area * 0.95:
            continue

        perimeter = cv2.arcLength(
            contour,
            True
        )

        if perimeter <= 0:
            continue

        approx = cv2.approxPolyDP(
            contour,
            0.04 * perimeter,
            True
        )

        if len(approx) != 4:
            continue

        points = approx.reshape(
            4,
            2
        ).astype(np.float32)

        rect = _order_points(points)

        tl, tr, br, bl = rect

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
                            top_width +
                            bottom_width
                    ) / 2.0

        avg_height = (
                             left_height +
                             right_height
                     ) / 2.0

        if avg_height <= 0:
            continue

        aspect_ratio = (
                avg_width /
                avg_height
        )

        if not 2.0 <= aspect_ratio <= 8.0:
            continue

        score = area * min(
            aspect_ratio / 4.0,
            1.0
        )

        if score > best_score:
            best_score = score
            best_quad = rect

    # ---------------------------------------------------------
    # 2. QUADRILATERAL FOUND
    # ---------------------------------------------------------

    if best_quad is not None:

        print(
            "[preprocess debug] "
            "perspective correction: QUAD"
        )

        tl, tr, br, bl = best_quad

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

        output_width = max(
            output_width,
            240
        )

        output_height = max(
            output_height,
            int(output_width / 5.0)
        )

        destination = np.array(
            [
                [0, 0],
                [output_width - 1, 0],
                [
                    output_width - 1,
                    output_height - 1
                ],
                [
                    0,
                    output_height - 1
                ]
            ],
            dtype=np.float32
        )

        matrix = cv2.getPerspectiveTransform(
            best_quad,
            destination
        )

        corrected = cv2.warpPerspective(
            plate_img,
            matrix,
            (
                output_width,
                output_height
            ),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

        return corrected

    # ---------------------------------------------------------
    # 3. FALLBACK: DETECT THE PLATE'S ROTATION ANGLE
    # ---------------------------------------------------------

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
            25,
            int(w * 0.12)
        ),
        minLineLength=max(
            30,
            int(w * 0.25)
        ),
        maxLineGap=20
    )

    if lines is None:
        print(
            "[preprocess debug] "
            "plate angle: not found"
        )
        return None

    angles = []
    lengths = []

    for line in lines:

        # HoughLinesP can return different array
        # shapes depending on OpenCV/version.
        # Flatten it so we always get x1,y1,x2,y2.
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
                30,
                w * 0.20
        ):
            continue

        angle = np.degrees(
            np.arctan2(
                dy,
                dx
            )
        )

        # We are interested in long,
        # roughly-horizontal plate edges.
        if -35 <= angle <= 35:

            angles.append(
                float(angle)
            )

            lengths.append(
                float(length)
            )

    if not angles:
        print(
            "[preprocess debug] "
            "plate angle: not found"
        )
        return None

    angles = np.asarray(
        angles,
        dtype=np.float32
    )

    lengths = np.asarray(
        lengths,
        dtype=np.float32
    )

    # Weighted average so long plate edges
    # matter more than short edges.
    angle = float(
        np.average(
            angles,
            weights=lengths
        )
    )

    # Ignore essentially straight plates.
    if abs(angle) < 2.0:
        print(
            "[preprocess debug] "
            "plate angle: already straight"
        )
        return None

    # Don't perform extreme corrections.
    if abs(angle) > 25:
        print(
            "[preprocess debug] "
            f"plate angle rejected: {angle:.2f}"
        )
        return None

    print(
        "[preprocess debug] "
        f"plate rotation detected: {angle:.2f} degrees, "
        f"correcting with {-angle:.2f} degrees"
    )

    center = (
        w // 2,
        h // 2
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    corrected = cv2.warpAffine(
        plate_img,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return corrected


def _variants_from_crop(plate_img):
    """
    Existing preprocessing pipeline.
    """

    padded = cv2.copyMakeBorder(
        plate_img,
        top=10,
        bottom=10,
        left=15,
        right=15,
        borderType=cv2.BORDER_REPLICATE
    )

    gray = cv2.cvtColor(
        padded,
        cv2.COLOR_BGR2GRAY
    )

    gray = _deskew(gray)

    h, w = gray.shape[:2]

    if w < 150:
        scale = 8
    elif w < 300:
        scale = 6
    else:
        scale = 4

    resized = cv2.resize(
        gray,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )

    blurred = cv2.GaussianBlur(
        resized,
        (3, 3),
        0
    )

    enhanced = _adaptive_enhance(
        blurred
    )

    variants = [
        enhanced
    ]

    # ---------------------------------------------------------
    # Adaptive threshold
    # ---------------------------------------------------------

    adaptive = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15,
        8
    )

    kernel_light = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    variants.append(
        cv2.morphologyEx(
            adaptive,
            cv2.MORPH_CLOSE,
            kernel_light,
            iterations=1
        )
    )

    # ---------------------------------------------------------
    # Otsu
    # ---------------------------------------------------------

    _, otsu = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    variants.append(
        otsu
    )

    # ---------------------------------------------------------
    # Softer adaptive threshold
    # ---------------------------------------------------------

    adaptive_soft = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10
    )

    variants.append(
        adaptive_soft
    )

    # ---------------------------------------------------------
    # Inverse adaptive threshold
    # ---------------------------------------------------------

    adaptive_inv = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        8
    )

    variants.append(
        cv2.morphologyEx(
            adaptive_inv,
            cv2.MORPH_CLOSE,
            kernel_light,
            iterations=1
        )
    )

    # ---------------------------------------------------------
    # Inverse Otsu
    # ---------------------------------------------------------

    _, otsu_inv = cv2.threshold(
        enhanced,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    variants.append(
        otsu_inv
    )

    # ---------------------------------------------------------
    # Strong close
    # ---------------------------------------------------------

    kernel_wide = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 3)
    )

    strong_close = cv2.morphologyEx(
        adaptive,
        cv2.MORPH_CLOSE,
        kernel_wide,
        iterations=2
    )

    variants.append(
        strong_close
    )

    # ---------------------------------------------------------
    # Sharpened Otsu
    # ---------------------------------------------------------

    gaussian = cv2.GaussianBlur(
        enhanced,
        (0, 0),
        3
    )

    sharpened = cv2.addWeighted(
        enhanced,
        1.5,
        gaussian,
        -0.5,
        0
    )

    _, sharp_otsu = cv2.threshold(
        sharpened,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    variants.append(
        sharp_otsu
    )

    # ---------------------------------------------------------
    # Denoised Otsu
    # ---------------------------------------------------------

    denoised = cv2.bilateralFilter(
        enhanced,
        9,
        75,
        75
    )

    _, denoised_otsu = cv2.threshold(
        denoised,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    variants.append(
        denoised_otsu
    )

    # ---------------------------------------------------------
    # Small rotation nudge
    # ---------------------------------------------------------

    (h2, w2) = enhanced.shape[:2]

    center2 = (
        w2 // 2,
        h2 // 2
    )

    M_pos = cv2.getRotationMatrix2D(
        center2,
        3,
        1.0
    )

    nudged = cv2.warpAffine(
        enhanced,
        M_pos,
        (w2, h2),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    _, nudged_otsu = cv2.threshold(
        nudged,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    variants.append(
        nudged_otsu
    )

    return variants


def preprocess_plate_variants(plate_img):
    """
    Generate OCR variants from:

    1. Original YOLO crop.
    2. Perspective/rotation-corrected crop.
    3. Super-resolved crop.
    4. Super-resolved + corrected crop.

    The original path is preserved so the new correction
    cannot break images that already work.
    """

    variants = []

    # =========================================================
    # 1. ORIGINAL YOLO CROP
    # =========================================================

    original_variants = _variants_from_crop(
        plate_img
    )

    variants.extend(
        original_variants
    )

    # =========================================================
    # 2. PERSPECTIVE / ROTATION CORRECTED CROP
    # =========================================================

    corrected = _perspective_correct(
        plate_img
    )

    if corrected is not None:

        print(
            "[preprocess debug] "
            "perspective correction: SUCCESS"
        )

        corrected_variants = _variants_from_crop(
            corrected
        )

        variants.extend(
            corrected_variants
        )

    else:

        print(
            "[preprocess debug] "
            "perspective correction: not found"
        )

    # =========================================================
    # 3. SUPER RESOLUTION
    # =========================================================

    sr_crop = maybe_super_resolve(
        plate_img
    )

    if sr_crop is not None:

        print(
            "[preprocess debug] "
            "super-resolution: SUCCESS"
        )

        variants.extend(
            _variants_from_crop(
                sr_crop
            )
        )

        # -----------------------------------------------------
        # 4. SUPER RESOLUTION + PERSPECTIVE CORRECTION
        # -----------------------------------------------------

        sr_corrected = _perspective_correct(
            sr_crop
        )

        if sr_corrected is not None:

            print(
                "[preprocess debug] "
                "SR perspective correction: SUCCESS"
            )

            variants.extend(
                _variants_from_crop(
                    sr_corrected
                )
            )

    return variants


def preprocess_plate(plate_img):
    return preprocess_plate_variants(
        plate_img
    )[0]