import re

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from detection.clean_text import clean_text_candidates


# ============================================================
# CONFIGURATION
# ============================================================

MAX_EDIT_DISTANCE = 2

# Minimum similarity before a DB record is considered
# remotely plausible.
#
# This is NOT the final acceptance threshold.
MIN_SIMILARITY = 70.0

# These statuses are handled with the same strong structural
# matching rules. We do not lower the matching requirements
# just because a vehicle is normal/stolen/blacklisted.
STRICT_STATUS = {
    "stolen",
    "blacklisted",
}

# When two different DB records are nearly equally plausible,
# avoid guessing.
MIN_MATCH_MARGIN = 3.0


# ============================================================
# OCR CHARACTER CONFUSIONS
# ============================================================

OCR_CONFUSION_PAIRS = {
    frozenset(("0", "O")),
    frozenset(("0", "D")),
    frozenset(("0", "Q")),

    frozenset(("1", "I")),
    frozenset(("1", "L")),
    frozenset(("1", "J")),
    frozenset(("1", "Y")),

    frozenset(("2", "Z")),

    frozenset(("3", "E")),

    frozenset(("4", "A")),

    frozenset(("5", "S")),

    frozenset(("6", "G")),

    frozenset(("7", "T")),

    frozenset(("8", "B")),
}


# ============================================================
# NORMALIZATION
# ============================================================

def _normalize_plate(text):
    """
    Keep only A-Z and 0-9.
    """

    if not text:
        return ""

    return re.sub(
        r"[^A-Z0-9]",
        "",
        str(text).upper()
    )


def _is_confusion_pair(a, b):
    """
    Check whether two different characters are a known
    OCR confusion pair.
    """

    if a == b:
        return False

    return frozenset(
        (a, b)
    ) in OCR_CONFUSION_PAIRS


# ============================================================
# DATABASE PLATE STRUCTURE
# ============================================================

def _possible_layouts(plate):
    """
    Generate possible structures for a database plate.

    Example:

        MH01EE2388

    becomes:

        state  = MH
        rto    = 01
        series = EE
        number = 2388

    We allow:
        RTO = 1 or 2 digits
        Series = 1 to 3 characters

    This function is primarily used for trusted DB plates.
    """

    plate = _normalize_plate(
        plate
    )

    if len(plate) < 8:
        return []

    state = plate[:2]
    number = plate[-4:]
    middle = plate[2:-4]

    if not state.isalpha():
        return []

    if not number.isdigit():
        return []

    layouts = []

    for rto_len in (1, 2):

        if len(middle) <= rto_len:
            continue

        rto = middle[:rto_len]
        series = middle[rto_len:]

        if not rto.isdigit():
            continue

        if not series.isalpha():
            continue

        if not 1 <= len(series) <= 3:
            continue

        layouts.append(
            {
                "state": state,
                "rto": rto,
                "series": series,
                "number": number,
            }
        )

    return layouts


def _best_layout_pair(
        ocr_text,
        db_text
):
    """
    Use the DB plate's trusted structure as the reference.

    IMPORTANT:

    The OCR string is NOT required to obey the expected
    letter/digit types.

    This is intentional.

    Example:

        DB:
        DL8CBG2956

        DB layout:
        DL | 8 | CBG | 2956

        OCR:
        QL8C9G2956

        OCR interpreted using DB layout:
        QL | 8 | C9G | 2956

    This lets us reason about OCR errors without throwing
    the candidate away just because OCR put a digit where
    a letter should be.
    """

    ocr_text = _normalize_plate(
        ocr_text
    )

    db_text = _normalize_plate(
        db_text
    )

    if not ocr_text or not db_text:
        return None

    if len(ocr_text) != len(db_text):
        return None

    db_layouts = _possible_layouts(
        db_text
    )

    if not db_layouts:
        return None

    best = None

    for db in db_layouts:

        state_len = len(
            db["state"]
        )

        rto_len = len(
            db["rto"]
        )

        series_len = len(
            db["series"]
        )

        number_len = len(
            db["number"]
        )

        expected_length = (
                state_len
                + rto_len
                + series_len
                + number_len
        )

        if expected_length != len(
                db_text
        ):
            continue

        # ----------------------------------------------------
        # Split OCR using the DB's known structure.
        #
        # No letter/digit validation is performed here.
        # ----------------------------------------------------

        position = 0

        ocr_state = ocr_text[
            position:
            position + state_len
        ]

        position += state_len

        ocr_rto = ocr_text[
            position:
            position + rto_len
        ]

        position += rto_len

        ocr_series = ocr_text[
            position:
            position + series_len
        ]

        position += series_len

        ocr_number = ocr_text[
            position:
            position + number_len
        ]

        # ----------------------------------------------------
        # Compare each section.
        # ----------------------------------------------------

        state_score = fuzz.ratio(
            ocr_state,
            db["state"]
        )

        rto_score = fuzz.ratio(
            ocr_rto,
            db["rto"]
        )

        series_score = fuzz.ratio(
            ocr_series,
            db["series"]
        )

        number_score = fuzz.ratio(
            ocr_number,
            db["number"]
        )

        # State and final registration number carry the
        # largest weight.
        structural_score = (
                state_score * 0.30
                + rto_score * 0.15
                + series_score * 0.15
                + number_score * 0.40
        )

        ocr_layout = {
            "state": ocr_state,
            "rto": ocr_rto,
            "series": ocr_series,
            "number": ocr_number,
        }

        candidate = (
            structural_score,
            ocr_layout,
            db,
            state_score,
            rto_score,
            series_score,
            number_score,
        )

        if (
                best is None
                or structural_score > best[0]
        ):
            best = candidate

    return best


# ============================================================
# CHARACTER-BY-CHARACTER COMPARISON
# ============================================================

def _compare_characters(
        ocr_text,
        db_text
):
    """
    Compare two same-length strings character-by-character.
    """

    length = min(
        len(ocr_text),
        len(db_text)
    )

    exact_positions = 0
    confusion_positions = 0

    mismatches = []

    for i in range(length):

        ocr_char = ocr_text[i]
        db_char = db_text[i]

        if ocr_char == db_char:

            exact_positions += 1

        else:

            if _is_confusion_pair(
                    ocr_char,
                    db_char
            ):
                confusion_positions += 1

            mismatches.append(
                (
                    i,
                    ocr_char,
                    db_char
                )
            )

    # --------------------------------------------------------
    # State prefix
    # --------------------------------------------------------

    prefix_length = min(
        2,
        length
    )

    prefix_matches = sum(
        1
        for i in range(
            prefix_length
        )
        if ocr_text[i] == db_text[i]
    )

    # --------------------------------------------------------
    # Final four registration digits
    # --------------------------------------------------------

    suffix_start = max(
        0,
        length - 4
    )

    suffix_matches = sum(
        1
        for i in range(
            suffix_start,
            length
        )
        if ocr_text[i] == db_text[i]
    )

    return {
        "exact_positions":
            exact_positions,

        "confusion_positions":
            confusion_positions,

        "mismatches":
            mismatches,

        "prefix_matches":
            prefix_matches,

        "suffix_matches":
            suffix_matches,
    }


# ============================================================
# COMBINED QUALITY SCORE
# ============================================================

def _quality_score(
        ocr_text,
        db_text,
        comparison,
        structural
):
    """
    Database-oriented score.

    This deliberately isn't just RapidFuzz.
    """

    fuzzy_score = fuzz.ratio(
        ocr_text,
        db_text
    )

    length = max(
        len(db_text),
        1
    )

    position_score = (
                             comparison["exact_positions"]
                             / length
                     ) * 100.0

    prefix_score = (
                           comparison["prefix_matches"]
                           / min(2, length)
                   ) * 100.0

    suffix_score = (
                           comparison["suffix_matches"]
                           / min(4, length)
                   ) * 100.0

    structural_score = (
        structural[0]
        if structural is not None
        else 0.0
    )

    return (
            fuzzy_score * 0.25
            + position_score * 0.25
            + prefix_score * 0.15
            + suffix_score * 0.20
            + structural_score * 0.15
    )


# ============================================================
# STRONG MATCH LOGIC
# ============================================================

def _strong_match(
        ocr_text,
        db_text,
        comparison,
        structural
):
    """
    Determine whether an OCR candidate has enough evidence
    to identify this specific DB plate.

    This intentionally does NOT mean:

        fuzzy score > X = match

    Instead it looks at:
        - edit distance
        - exact character positions
        - state code
        - RTO
        - series
        - final four digits
        - OCR confusion pairs
        - DB structure
    """

    if not ocr_text or not db_text:
        return False

    if len(ocr_text) != len(db_text):
        return False

    distance = Levenshtein.distance(
        ocr_text,
        db_text
    )

    if distance > MAX_EDIT_DISTANCE:
        return False

    exact_positions = comparison[
        "exact_positions"
    ]

    confusion_positions = comparison[
        "confusion_positions"
    ]

    prefix_matches = comparison[
        "prefix_matches"
    ]

    suffix_matches = comparison[
        "suffix_matches"
    ]

    # ========================================================
    # EXACT MATCH
    # ========================================================

    if distance == 0:
        return True

    # ========================================================
    # ONE CHARACTER ERROR
    # ========================================================

    if distance == 1:

        # State is correct and most of the registration number
        # is also correct.
        if (
                prefix_matches == 2
                and suffix_matches >= 3
        ):
            return True

        # Final four digits are exact and the state still has
        # at least one matching character.
        #
        # Example:
        #
        # HH01EE2388
        # MH01EE2388
        if (
                suffix_matches == 4
                and prefix_matches >= 1
        ):
            return True

        # Known OCR confusion + strong anchor.
        if (
                confusion_positions == 1
                and (
                prefix_matches == 2
                or suffix_matches >= 3
        )
        ):
            return True

    # ========================================================
    # TWO CHARACTER ERRORS
    # ========================================================

    if distance == 2:

        # ----------------------------------------------------
        # Two known OCR confusions + strong anchors.
        # ----------------------------------------------------

        if (
                confusion_positions == 2
                and prefix_matches == 2
                and suffix_matches >= 2
        ):
            return True

        # ----------------------------------------------------
        # Very strong positional agreement.
        # ----------------------------------------------------

        if (
                prefix_matches == 2
                and suffix_matches >= 3
                and exact_positions >= len(db_text) - 2
        ):
            return True

        # ----------------------------------------------------
        # Strong DB structure:
        #
        # exact RTO
        # exact final four digits
        # only two total errors
        #
        # Example:
        #
        # QL8C9G2956
        # DL8CBG2956
        # ----------------------------------------------------

        if structural is not None:

            (
                structural_score,
                ocr_layout,
                db_layout,
                state_score,
                rto_score,
                series_score,
                number_score,
            ) = structural

            exact_rto = (
                    ocr_layout["rto"]
                    == db_layout["rto"]
            )

            exact_number = (
                    ocr_layout["number"]
                    == db_layout["number"]
            )

            if (
                    exact_rto
                    and exact_number
                    and exact_positions
                    >= len(db_text) - 2
            ):
                return True

        # ----------------------------------------------------
        # Strong structure even where the character substitutions
        # aren't explicitly known OCR confusions.
        #
        # Example:
        #
        # JK0EH0082
        # JK08H0088
        # ----------------------------------------------------

        if structural is not None:

            (
                structural_score,
                _,
                _,
                state_score,
                rto_score,
                series_score,
                number_score,
            ) = structural

            if (
                    state_score >= 99.0
                    and number_score >= 75.0
                    and structural_score >= 78.0
                    and exact_positions
                    >= len(db_text) - 2
            ):
                return True

    return False


# ============================================================
# EVALUATE ONE OCR RESULT AGAINST ONE DB RECORD
# ============================================================

def _evaluate_reference(
        cleaned_text,
        reference_plate
):
    """
    Produce all matching evidence for one OCR result
    versus one database plate.
    """

    ocr_text = _normalize_plate(
        cleaned_text
    )

    db_text = _normalize_plate(
        reference_plate.get(
            "plate_number",
            ""
        )
    )

    if not ocr_text or not db_text:
        return None

    # Do not silently add/remove characters when comparing
    # registration numbers.
    if len(ocr_text) != len(db_text):
        return None

    distance = Levenshtein.distance(
        ocr_text,
        db_text
    )

    if distance > MAX_EDIT_DISTANCE:
        return None

    fuzzy_score = fuzz.ratio(
        ocr_text,
        db_text
    )

    if fuzzy_score < MIN_SIMILARITY:
        return None

    comparison = _compare_characters(
        ocr_text,
        db_text
    )

    structural = _best_layout_pair(
        ocr_text,
        db_text
    )

    quality = _quality_score(
        ocr_text,
        db_text,
        comparison,
        structural
    )

    strong = _strong_match(
        ocr_text,
        db_text,
        comparison,
        structural
    )

    return {
        "reference":
            reference_plate,

        "ocr_text":
            ocr_text,

        "db_text":
            db_text,

        "distance":
            distance,

        "fuzzy_score":
            float(fuzzy_score),

        "quality":
            float(quality),

        "strong_match":
            strong,

        "exact_positions":
            comparison[
                "exact_positions"
            ],

        "confusion_positions":
            comparison[
                "confusion_positions"
            ],

        "prefix_matches":
            comparison[
                "prefix_matches"
            ],

        "suffix_matches":
            comparison[
                "suffix_matches"
            ],

        "structural_score":
            (
                structural[0]
                if structural is not None
                else 0.0
            ),

        "mismatches":
            comparison[
                "mismatches"
            ],
    }


# ============================================================
# SINGLE OCR STRING → DATABASE
# ============================================================

def find_best_match(
        cleaned_text: str,
        reference_plates: list
):
    """
    Compare a single cleaned OCR result against the entire DB.
    """

    cleaned_text = _normalize_plate(
        cleaned_text
    )

    if not cleaned_text:
        return (
            None,
            0.0,
            "no_match"
        )

    if not reference_plates:
        return (
            None,
            0.0,
            "no_match"
        )

    evaluations = []

    for reference_plate in reference_plates:

        evaluation = _evaluate_reference(
            cleaned_text,
            reference_plate
        )

        if evaluation is not None:
            evaluations.append(
                evaluation
            )

    if not evaluations:
        return (
            None,
            0.0,
            "no_match"
        )

    strong_matches = [
        item
        for item in evaluations
        if item["strong_match"]
    ]

    # --------------------------------------------------------
    # Nothing strong enough.
    # --------------------------------------------------------

    if not strong_matches:

        evaluations.sort(
            key=lambda item: (
                item["quality"],
                item["fuzzy_score"],
                -item["distance"],
            ),
            reverse=True
        )

        return (
            None,
            evaluations[0]["fuzzy_score"],
            "no_match"
        )

    # --------------------------------------------------------
    # Rank strong matches.
    # --------------------------------------------------------

    strong_matches.sort(
        key=lambda item: (
            item["quality"],
            item["fuzzy_score"],
            -item["distance"],
        ),
        reverse=True
    )

    best = strong_matches[0]

    # --------------------------------------------------------
    # Ambiguity protection.
    # --------------------------------------------------------

    if len(strong_matches) > 1:

        second = strong_matches[1]

        margin = (
                best["quality"]
                - second["quality"]
        )

        if (
                best["fuzzy_score"] < 100.0
                and margin < MIN_MATCH_MARGIN
        ):

            return (
                None,
                best["fuzzy_score"],
                "ambiguous"
            )

    return (
        best["reference"],
        best["fuzzy_score"],
        "matched"
    )


# ============================================================
# OCR CANDIDATE QUALITY
# ============================================================

def _ocr_candidate_quality(
        cleaned_text,
        confidence
):
    """
    Rank OCR results by usefulness as vehicle registrations.

    OCR confidence is only one small part of this score.

    For example:

        TT
        confidence = 80

    should rank below:

        QL8C9G2956
        confidence = 33

    because the second one contains much more registration
    information.
    """

    text = _normalize_plate(
        cleaned_text
    )

    if not text:
        return -1.0

    # --------------------------------------------------------
    # Length
    # --------------------------------------------------------

    if len(text) == 10:
        length_score = 40.0

    elif len(text) == 9:
        length_score = 32.0

    elif len(text) == 8:
        length_score = 20.0

    elif len(text) >= 6:
        length_score = 10.0

    else:
        length_score = 0.0

    # --------------------------------------------------------
    # Mixed letters + digits
    # --------------------------------------------------------

    has_letters = any(
        c.isalpha()
        for c in text
    )

    has_digits = any(
        c.isdigit()
        for c in text
    )

    mixed_score = (
        15.0
        if has_letters and has_digits
        else 0.0
    )

    # --------------------------------------------------------
    # OCR confidence
    #
    # Deliberately low weighting.
    # --------------------------------------------------------

    try:
        confidence_value = float(
            confidence
        )
    except (
            TypeError,
            ValueError
    ):
        confidence_value = 0.0

    confidence_value = min(
        max(
            confidence_value,
            0.0
        ),
        100.0
    )

    confidence_score = (
            confidence_value * 0.15
    )

    return (
            length_score
            + mixed_score
            + confidence_score
    )


# ============================================================
# ALL OCR CANDIDATES → DATABASE
# ============================================================

def find_best_match_across_candidates(
        ocr_candidates,
        reference_plates
):
    """
    Compare all OCR candidates against the entire database.

    This is the main database-aware matching function.

    It does NOT blindly trust the highest EasyOCR confidence.

    It can identify a lower-confidence OCR result when that
    result strongly agrees with a registered DB plate.
    """

    # ========================================================
    # NO OCR
    # ========================================================

    if not ocr_candidates:

        return (
            "",
            "",
            0.0,
            None,
            0.0,
            "no_match"
        )

    # ========================================================
    # BEST OCR FALLBACK
    #
    # Used only when there is no database match.
    # ========================================================

    best_ocr = None

    for raw_text, confidence in ocr_candidates:

        if not raw_text:
            continue

        try:
            conf = float(
                confidence
            )
        except (
                TypeError,
                ValueError
        ):
            conf = 0.0

        cleaned_options = (
            clean_text_candidates(
                raw_text
            )
        )

        if not cleaned_options:

            cleaned_options = [
                _normalize_plate(
                    raw_text
                )
            ]

        # Remove duplicates.
        cleaned_options = list(
            dict.fromkeys(
                cleaned_options
            )
        )

        for cleaned in cleaned_options:

            cleaned = _normalize_plate(
                cleaned
            )

            if not cleaned:
                continue

            candidate_quality = (
                _ocr_candidate_quality(
                    cleaned,
                    conf
                )
            )

            candidate = (
                raw_text,
                cleaned,
                conf,
                candidate_quality
            )

            if (
                    best_ocr is None
                    or candidate_quality
                    > best_ocr[3]
            ):
                best_ocr = candidate

    # ========================================================
    # DATABASE MISSING
    # ========================================================

    if not reference_plates:

        if best_ocr is not None:

            return (
                best_ocr[0],
                best_ocr[1],
                best_ocr[2],
                None,
                0.0,
                "no_match"
            )

        return (
            "",
            "",
            0.0,
            None,
            0.0,
            "no_match"
        )

    # ========================================================
    # DATABASE EVIDENCE
    # ========================================================

    evidence = {}

    # Best fuzzy relation, retained for debugging.
    best_fallback = None

    # ========================================================
    # TEST EVERY OCR CANDIDATE AGAINST EVERY DB PLATE
    # ========================================================

    for raw_text, confidence in ocr_candidates:

        if not raw_text:
            continue

        try:
            conf = float(
                confidence
            )
        except (
                TypeError,
                ValueError
        ):
            conf = 0.0

        cleaned_options = (
            clean_text_candidates(
                raw_text
            )
        )

        if not cleaned_options:

            cleaned_options = [
                _normalize_plate(
                    raw_text
                )
            ]

        cleaned_options = list(
            dict.fromkeys(
                cleaned_options
            )
        )

        for cleaned in cleaned_options:

            cleaned = _normalize_plate(
                cleaned
            )

            if not cleaned:
                continue

            for reference_plate in reference_plates:

                evaluation = _evaluate_reference(
                    cleaned,
                    reference_plate
                )

                if evaluation is None:
                    continue

                # ------------------------------------------------
                # Preserve best fuzzy/structural fallback.
                # ------------------------------------------------

                fallback = (
                    raw_text,
                    cleaned,
                    conf,
                    evaluation
                )

                if (
                        best_fallback is None
                        or evaluation["quality"]
                        > best_fallback[3]["quality"]
                ):
                    best_fallback = fallback

                # ------------------------------------------------
                # Only strong evidence can identify a plate.
                # ------------------------------------------------

                if not evaluation[
                    "strong_match"
                ]:
                    continue

                reference_id = (
                    reference_plate.get(
                        "id",
                        reference_plate.get(
                            "plate_number"
                        )
                    )
                )

                if reference_id not in evidence:

                    evidence[
                        reference_id
                    ] = {
                        "reference":
                            reference_plate,

                        "observations":
                            {}
                    }

                observations = evidence[
                    reference_id
                ]["observations"]

                # Same cleaned OCR interpretation repeated over
                # many preprocessing variants counts as one
                # distinct observation.
                existing = observations.get(
                    cleaned
                )

                if (
                        existing is None
                        or conf > existing[
                    "confidence"
                ]
                        or evaluation["quality"]
                        > existing[
                    "evaluation"
                ]["quality"]
                ):

                    observations[
                        cleaned
                    ] = {
                        "raw_text":
                            raw_text,

                        "cleaned_text":
                            cleaned,

                        "confidence":
                            conf,

                        "evaluation":
                            evaluation,
                    }

    # ========================================================
    # NO STRONG DATABASE MATCH
    # ========================================================

    if not evidence:

        # IMPORTANT:
        #
        # We still return useful OCR information.
        #
        # A failed DB match is not the same thing as
        # "no plate detected".
        #

        if best_ocr is not None:

            fallback_score = 0.0

            if best_fallback is not None:

                fallback_score = (
                    best_fallback[3][
                        "fuzzy_score"
                    ]
                )

            return (
                best_ocr[0],
                best_ocr[1],
                best_ocr[2],
                None,
                fallback_score,
                "no_match"
            )

        return (
            "",
            "",
            0.0,
            None,
            0.0,
            "no_match"
        )

    # ========================================================
    # RANK DATABASE RECORDS
    # ========================================================

    ranked = []

    for reference_id, data in evidence.items():

        observations = list(
            data["observations"].values()
        )

        if not observations:
            continue

        # Strongest observation first.
        observations.sort(
            key=lambda item: (
                item["evaluation"]["quality"],
                item["evaluation"]["fuzzy_score"],
                item["confidence"],
            ),
            reverse=True
        )

        best_observation = observations[
            0
        ]

        # ----------------------------------------------------
        # Main score
        # ----------------------------------------------------

        rank_score = (
            best_observation[
                "evaluation"
            ]["quality"]
        )

        # OCR confidence contributes only a little.
        rank_score += (
                min(
                    max(
                        best_observation[
                            "confidence"
                        ],
                        0.0
                    ),
                    100.0
                )
                * 0.02
        )

        # ----------------------------------------------------
        # Distinct OCR consensus
        #
        # Repeating one identical result 20 times does not
        # create 20 votes.
        # ----------------------------------------------------

        support = len(
            observations
        )

        if support >= 2:
            rank_score += 3.0

        if support >= 3:
            rank_score += 2.0

        if support >= 4:
            rank_score += min(
                support - 3,
                3
            )

        ranked.append(
            {
                "reference":
                    data["reference"],

                "best_observation":
                    best_observation,

                "support":
                    support,

                "score":
                    rank_score,
            }
        )

    # ========================================================
    # SORT
    # ========================================================

    ranked.sort(
        key=lambda item: (
            item["score"],
            item["best_observation"][
                "evaluation"
            ]["quality"],
            item["best_observation"][
                "evaluation"
            ]["fuzzy_score"],
        ),
        reverse=True
    )

    if not ranked:

        if best_ocr is not None:

            return (
                best_ocr[0],
                best_ocr[1],
                best_ocr[2],
                None,
                0.0,
                "no_match"
            )

        return (
            "",
            "",
            0.0,
            None,
            0.0,
            "no_match"
        )

    best = ranked[0]

    # ========================================================
    # AMBIGUITY PROTECTION
    # ========================================================

    if len(ranked) > 1:

        second = ranked[1]

        margin = (
                best["score"]
                - second["score"]
        )

        best_evaluation = (
            best["best_observation"][
                "evaluation"
            ]
        )

        # Exact matches don't need ambiguity protection.
        if (
                best_evaluation[
                    "fuzzy_score"
                ] < 100.0
                and margin < MIN_MATCH_MARGIN
        ):

            observation = (
                best["best_observation"]
            )

            return (
                observation[
                    "raw_text"
                ],
                observation[
                    "cleaned_text"
                ],
                observation[
                    "confidence"
                ],
                None,
                best_evaluation[
                    "fuzzy_score"
                ],
                "ambiguous"
            )

    # ========================================================
    # FINAL MATCH
    # ========================================================

    observation = (
        best["best_observation"]
    )

    candidate = (
        best["reference"]
    )

    evaluation = observation[
        "evaluation"
    ]

    # --------------------------------------------------------
    # Strict statuses
    #
    # They already passed _strong_match().
    # We therefore do not use a separate arbitrary fuzzy
    # threshold such as 92%.
    # --------------------------------------------------------

    status = str(
        candidate.get(
            "status",
            ""
        )
    ).lower()

    if status in STRICT_STATUS:

        if not evaluation[
            "strong_match"
        ]:

            return (
                observation[
                    "raw_text"
                ],
                observation[
                    "cleaned_text"
                ],
                observation[
                    "confidence"
                ],
                None,
                evaluation[
                    "fuzzy_score"
                ],
                "no_match"
            )

    return (
        observation[
            "raw_text"
        ],
        observation[
            "cleaned_text"
        ],
        observation[
            "confidence"
        ],
        candidate,
        evaluation[
            "fuzzy_score"
        ],
        "matched"
    )