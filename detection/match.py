import re

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from detection.clean_text import clean_text_candidates


# ============================================================
# CONFIGURATION
# ============================================================

# Maximum character edits we are willing to consider.
MAX_EDIT_DISTANCE = 2

# Minimum general fuzzy similarity for a candidate to even be
# considered plausible.
MIN_SIMILARITY = 80.0

# Candidate must have the same number of characters as the DB
# plate for an actual match.
#
# This is intentionally strict. A missing/extra character can
# completely change a registration number.
REQUIRE_SAME_LENGTH = True

# Statuses that deserve stricter acceptance rules.
STRICT_STATUS = {
    "stolen",
    "blacklisted",
}

# Minimum ranking margin between the best and second-best DB
# plate when the match is not exact.
MIN_MATCH_MARGIN = 2.0


# ============================================================
# OCR CHARACTER CONFUSIONS
# ============================================================

# These are common OCR confusions between letters and digits.
#
# We use them ONLY when comparing OCR output against a
# database plate. We do NOT globally replace characters here.
#
# Example:
#
#     O ↔ 0
#     I ↔ 1
#     B ↔ 8
#
# This preserves the original OCR text while allowing the
# matcher to recognize plausible OCR mistakes.
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
    Normalize a plate for comparison.

    Keeps only A-Z and 0-9.
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
    Return True if a and b are a known OCR confusion pair.
    """

    if a == b:
        return False

    return frozenset(
        (a, b)
    ) in OCR_CONFUSION_PAIRS


# ============================================================
# CHARACTER / POSITION ANALYSIS
# ============================================================

def _compare_characters(ocr_text, db_text):
    """
    Compare two same-length strings character by character.

    Returns detailed structural information.
    """

    length = len(db_text)

    exact_positions = 0
    confusion_positions = 0
    mismatched_positions = []

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

            mismatched_positions.append(
                (
                    i,
                    ocr_char,
                    db_char
                )
            )

    # --------------------------------------------------------
    # Prefix agreement
    #
    # First two characters are normally the state code.
    # --------------------------------------------------------

    prefix_length = min(
        2,
        length
    )

    prefix_matches = sum(
        1
        for i in range(prefix_length)
        if ocr_text[i] == db_text[i]
    )

    # --------------------------------------------------------
    # Suffix agreement
    #
    # Last four characters are an extremely useful anchor
    # because they normally represent the registration number.
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
        "exact_positions": exact_positions,
        "confusion_positions": confusion_positions,
        "mismatched_positions": mismatched_positions,
        "prefix_matches": prefix_matches,
        "suffix_matches": suffix_matches,
    }


# ============================================================
# MATCH QUALITY
# ============================================================

def _match_quality(
        ocr_text,
        db_text,
        comparison
):
    """
    Calculate an evidence-based quality score.

    This is NOT the same as RapidFuzz score.

    It combines:

        - overall string similarity
        - exact character positions
        - state prefix agreement
        - final registration number agreement
    """

    length = len(db_text)

    if length == 0:
        return 0.0

    fuzzy_score = fuzz.ratio(
        ocr_text,
        db_text
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

    # Weighted toward actual positional agreement.
    quality = (
            fuzzy_score * 0.40 +
            position_score * 0.30 +
            prefix_score * 0.15 +
            suffix_score * 0.15
    )

    return quality


# ============================================================
# ACCEPTANCE RULES
# ============================================================

def _strong_match_rules(
        ocr_text,
        db_text,
        comparison
):
    """
    Decide whether this OCR string is structurally strong
    enough to identify the database plate.

    The goal is to avoid "85% fuzzy = match".

    A strong match should have meaningful positional evidence.
    """

    length = len(db_text)

    if length < 8:
        return False

    edit_distance = Levenshtein.distance(
        ocr_text,
        db_text
    )

    if edit_distance > MAX_EDIT_DISTANCE:
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

    mismatched_positions = comparison[
        "mismatched_positions"
    ]

    # --------------------------------------------------------
    # RULE 1: EXACT MATCH
    # --------------------------------------------------------

    if ocr_text == db_text:
        return True

    # --------------------------------------------------------
    # RULE 2: ONE-CHARACTER ERROR WITH STRONG ANCHORS
    #
    # Example:
    #
    # DL7CO1939
    # DL7CQ1939
    #
    # or:
    #
    # WB22U4I16
    # WB22U4116
    # --------------------------------------------------------

    if edit_distance == 1:

        # State + registration number strongly agree.
        if (
                prefix_matches == 2
                and suffix_matches >= 3
        ):
            return True

        # Registration number is completely correct and only
        # one of the first/state characters is wrong.
        #
        # Example:
        #
        # HH01EE2388
        # MH01EE2388
        if (
                suffix_matches == 4
                and exact_positions >= length - 1
                and prefix_matches >= 1
        ):
            return True

        # One OCR confusion with a strong DB anchor.
        if (
                confusion_positions == 1
                and (
                prefix_matches == 2
                or suffix_matches >= 3
        )
        ):
            return True

    # --------------------------------------------------------
    # RULE 3: TWO OCR ERRORS
    #
    # We allow two errors only when they are both plausible
    # OCR confusions and the plate still has strong anchors.
    #
    # Example:
    #
    # WB22U4I1B
    # WB22U4118
    # --------------------------------------------------------

    if edit_distance == 2:

        if (
                confusion_positions == 2
                and prefix_matches == 2
                and suffix_matches >= 2
        ):
            return True

        # One state-code error + one OCR confusion can still be
        # strong if the rest of the plate agrees exactly.
        if (
                suffix_matches == 4
                and exact_positions >= length - 2
                and confusion_positions >= 1
        ):
            return True

    # --------------------------------------------------------
    # Otherwise reject.
    # --------------------------------------------------------

    return False


# ============================================================
# ONE OCR STRING AGAINST THE WHOLE DATABASE
# ============================================================

def _evaluate_reference(
        cleaned_text,
        reference_plate
):
    """
    Evaluate one cleaned OCR string against one DB plate.

    Returns detailed information.
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

    # Don't try to compare a completely different length.
    if (
            REQUIRE_SAME_LENGTH
            and len(ocr_text) != len(db_text)
    ):
        return None

    edit_distance = Levenshtein.distance(
        ocr_text,
        db_text
    )

    if edit_distance > MAX_EDIT_DISTANCE:
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

    quality = _match_quality(
        ocr_text,
        db_text,
        comparison
    )

    strong_match = _strong_match_rules(
        ocr_text,
        db_text,
        comparison
    )

    return {
        "reference": reference_plate,
        "ocr_text": ocr_text,
        "db_text": db_text,
        "fuzzy_score": float(
            fuzzy_score
        ),
        "quality": float(
            quality
        ),
        "edit_distance": int(
            edit_distance
        ),
        "exact_positions": comparison[
            "exact_positions"
        ],
        "confusion_positions": comparison[
            "confusion_positions"
        ],
        "prefix_matches": comparison[
            "prefix_matches"
        ],
        "suffix_matches": comparison[
            "suffix_matches"
        ],
        "mismatched_positions": comparison[
            "mismatched_positions"
        ],
        "strong_match": strong_match,
    }


# ============================================================
# BEST MATCH FOR ONE CLEANED STRING
# ============================================================

def find_best_match(
        cleaned_text: str,
        reference_plates: list
):
    """
    Match one cleaned OCR candidate against the complete DB.

    Unlike the previous implementation, this does NOT use
    RapidFuzz extractOne() to select one candidate and then
    stop.

    Every plausible reference plate is evaluated.
    """

    cleaned_text = _normalize_plate(
        cleaned_text
    )

    if not cleaned_text or not reference_plates:
        return None, 0.0, "no_match"

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
        return None, 0.0, "no_match"

    # --------------------------------------------------------
    # First prefer actual strong matches.
    # --------------------------------------------------------

    strong_matches = [
        item
        for item in evaluations
        if item["strong_match"]
    ]

    if strong_matches:

        strong_matches.sort(
            key=lambda item: (
                item["quality"],
                item["fuzzy_score"],
                -item["edit_distance"],
                item["suffix_matches"],
                item["prefix_matches"],
            ),
            reverse=True
        )

        best = strong_matches[0]

        # ----------------------------------------------------
        # Check if there is another DB plate almost as strong.
        #
        # This protects against ambiguous registrations.
        # ----------------------------------------------------

        if len(strong_matches) > 1:

            second = strong_matches[1]

            margin = (
                    best["quality"]
                    - second["quality"]
            )

            if (
                    margin < MIN_MATCH_MARGIN
                    and best["fuzzy_score"] < 100.0
            ):
                return (
                    None,
                    best["fuzzy_score"],
                    "ambiguous"
                )

        candidate = best["reference"]

        # ----------------------------------------------------
        # Strict-status handling
        #
        # We do NOT reduce strict status thresholds.
        #
        # Instead, a stolen/blacklisted vehicle must satisfy
        # one of the explicit strong structural rules above.
        # ----------------------------------------------------

        status = str(
            candidate.get(
                "status",
                ""
            )
        ).lower()

        if status in STRICT_STATUS:

            if not best["strong_match"]:
                return (
                    None,
                    best["fuzzy_score"],
                    "no_match"
                )

        return (
            candidate,
            best["fuzzy_score"],
            "matched"
        )

    # --------------------------------------------------------
    # If no strong match exists, return the best fuzzy fallback
    # for debugging/API visibility, but do NOT match it.
    # --------------------------------------------------------

    evaluations.sort(
        key=lambda item: (
            item["quality"],
            item["fuzzy_score"],
            -item["edit_distance"],
            item["suffix_matches"],
            item["prefix_matches"],
        ),
        reverse=True
    )

    best = evaluations[0]

    return (
        None,
        best["fuzzy_score"],
        "no_match"
    )


# ============================================================
# DATABASE CONSENSUS
# ============================================================

def find_best_match_across_candidates(
        ocr_candidates: list,
        reference_plates: list
):
    """
    Match ALL OCR candidates against the complete database.

    This is where the system becomes much more database-aware.

    Multiple preprocessing/OCR variants that independently point
    toward the same DB plate increase confidence.

    We intentionally use diminishing returns so that generating
    20 identical preprocessing variants does not artificially
    make one plate unbeatable.
    """

    if not ocr_candidates or not reference_plates:
        return (
            "",
            "",
            0.0,
            None,
            0.0,
            "no_match"
        )

    # --------------------------------------------------------
    # For each DB plate, retain the strongest distinct OCR
    # observations.
    #
    # Structure:
    #
    # {
    #     reference_id: {
    #         cleaned_text: observation
    #     }
    # }
    # --------------------------------------------------------

    evidence_by_reference = {}

    # Also retain the best fallback candidate for debugging.
    best_fallback = None

    for raw_text, confidence in ocr_candidates:

        if not raw_text:
            continue

        try:
            ocr_confidence = float(
                confidence
            )
        except (
                TypeError,
                ValueError
        ):
            ocr_confidence = 0.0

        cleaned_options = (
            clean_text_candidates(
                raw_text
            )
        )

        if not cleaned_options:

            fallback_cleaned = _normalize_plate(
                raw_text
            )

            if fallback_cleaned:
                cleaned_options = [
                    fallback_cleaned
                ]

        # Avoid duplicate cleaner outputs for the same OCR.
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

            # ------------------------------------------------
            # Evaluate against EVERY database plate.
            # ------------------------------------------------

            for reference_plate in reference_plates:

                evaluation = _evaluate_reference(
                    cleaned,
                    reference_plate
                )

                if evaluation is None:
                    continue

                # ------------------------------------------------
                # Keep fallback regardless of whether it qualifies
                # as a real match.
                # ------------------------------------------------

                fallback_item = (
                    raw_text,
                    cleaned,
                    ocr_confidence,
                    evaluation
                )

                if (
                        best_fallback is None
                        or evaluation["quality"]
                        > best_fallback[3]["quality"]
                ):
                    best_fallback = fallback_item

                # ------------------------------------------------
                # Only strong relations participate in DB
                # consensus.
                # ------------------------------------------------

                if not evaluation[
                    "strong_match"
                ]:
                    continue

                reference_id = reference_plate.get(
                    "id",
                    reference_plate.get(
                        "plate_number"
                    )
                )

                if reference_id not in evidence_by_reference:

                    evidence_by_reference[
                        reference_id
                    ] = {
                        "reference": reference_plate,
                        "observations": {}
                    }

                observations = (
                    evidence_by_reference[
                        reference_id
                    ]["observations"]
                )

                # ------------------------------------------------
                # Same cleaned text appearing in 10 variants should
                # count as ONE distinct observation.
                #
                # Keep only the strongest version.
                # ------------------------------------------------

                existing = observations.get(
                    cleaned
                )

                if (
                        existing is None
                        or ocr_confidence
                        > existing[
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
                        "raw_text": raw_text,
                        "cleaned_text": cleaned,
                        "confidence": ocr_confidence,
                        "evaluation": evaluation,
                    }

    # --------------------------------------------------------
    # No strong DB relations.
    # --------------------------------------------------------

    if not evidence_by_reference:

        if best_fallback is not None:

            raw_text = best_fallback[0]
            cleaned = best_fallback[1]
            confidence = best_fallback[2]
            evaluation = best_fallback[3]

            return (
                raw_text,
                cleaned,
                confidence,
                None,
                evaluation["fuzzy_score"],
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
    # SCORE EACH DATABASE PLATE USING CONSENSUS
    # ========================================================

    ranked_references = []

    for reference_id, data in evidence_by_reference.items():

        observations = list(
            data["observations"].values()
        )

        if not observations:
            continue

        # Best individual OCR interpretation.
        observations.sort(
            key=lambda item: (
                item["evaluation"]["quality"],
                item["evaluation"]["fuzzy_score"],
                item["confidence"],
            ),
            reverse=True
        )

        best_observation = observations[0]

        # ----------------------------------------------------
        # Base = best structural evidence.
        # ----------------------------------------------------

        rank_score = (
            best_observation[
                "evaluation"
            ]["quality"]
        )

        # ----------------------------------------------------
        # OCR confidence is deliberately secondary.
        #
        # OCR confidence tells us how confident EasyOCR was,
        # not whether the database identification is correct.
        # ----------------------------------------------------

        confidence_bonus = min(
            max(
                best_observation[
                    "confidence"
                ],
                0.0
            ),
            100.0
        ) * 0.03

        rank_score += (
            confidence_bonus
        )

        # ----------------------------------------------------
        # Consensus bonus.
        #
        # Distinct cleaned strings independently pointing at the
        # same DB plate are useful evidence.
        #
        # Diminishing returns:
        #
        # 2nd interpretation → +3
        # 3rd interpretation → +2
        # 4th+              → +1 each up to +3 total
        #
        # This prevents 20 identical preprocessing variants
        # from overwhelming the system.
        # ----------------------------------------------------

        distinct_support = len(
            observations
        )

        if distinct_support >= 2:
            rank_score += 3.0

        if distinct_support >= 3:
            rank_score += 2.0

        if distinct_support >= 4:
            rank_score += min(
                distinct_support - 3,
                3
            ) * 1.0

        ranked_references.append(
            {
                "reference": data[
                    "reference"
                ],
                "best_observation":
                    best_observation,
                "support_count":
                    distinct_support,
                "rank_score":
                    rank_score,
            }
        )

    # --------------------------------------------------------
    # Sort DB candidates.
    # --------------------------------------------------------

    ranked_references.sort(
        key=lambda item: (
            item["rank_score"],
            item["best_observation"][
                "evaluation"
            ]["quality"],
            item["best_observation"][
                "evaluation"
            ]["fuzzy_score"],
        ),
        reverse=True
    )

    best = ranked_references[0]

    # --------------------------------------------------------
    # Ambiguity protection.
    # --------------------------------------------------------

    if len(ranked_references) > 1:

        second = ranked_references[1]

        margin = (
                best["rank_score"]
                - second["rank_score"]
        )

        best_evaluation = (
            best["best_observation"][
                "evaluation"
            ]
        )

        # Exact match is never considered ambiguous.
        if (
                best_evaluation[
                    "fuzzy_score"
                ] < 100.0
                and margin < MIN_MATCH_MARGIN
        ):

            observation = best[
                "best_observation"
            ]

            return (
                observation["raw_text"],
                observation["cleaned_text"],
                observation["confidence"],
                None,
                observation[
                    "evaluation"
                ]["fuzzy_score"],
                "ambiguous"
            )

    # --------------------------------------------------------
    # Final matched DB record.
    # --------------------------------------------------------

    observation = best[
        "best_observation"
    ]

    candidate = best[
        "reference"
    ]

    evaluation = observation[
        "evaluation"
    ]

    # --------------------------------------------------------
    # Strict status protection.
    #
    # There is intentionally NO reduced threshold here.
    #
    # A stolen/blacklisted record must already satisfy our
    # structural strong-match rules.
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
                observation["raw_text"],
                observation["cleaned_text"],
                observation["confidence"],
                None,
                evaluation["fuzzy_score"],
                "no_match"
            )

    return (
        observation["raw_text"],
        observation["cleaned_text"],
        observation["confidence"],
        candidate,
        evaluation["fuzzy_score"],
        "matched"
    )