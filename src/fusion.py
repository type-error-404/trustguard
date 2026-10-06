"""
fusion.py
---------
Combines available modality probabilities
into a TrustGuard risk score.
"""


# ============================================================
# CLAMP VALUE
# ============================================================

def clamp(value):

    return max(
        0.0,
        min(
            1.0,
            float(value)
        )
    )


# ============================================================
# CALCULATE RISK
# ============================================================

def calculate_risk(
    image_fake_probability=None,
    video_fake_probability=None,
    audio_fake_probability=None,
    document_risk=None
):

    values = []
    weights = []


    # ========================================================
    # IMAGE
    # ========================================================

    if image_fake_probability is not None:

        values.append(
            clamp(
                image_fake_probability
            )
        )

        weights.append(
            0.40
        )


    # ========================================================
    # VIDEO
    # ========================================================

    if video_fake_probability is not None:

        values.append(
            clamp(
                video_fake_probability
            )
        )

        weights.append(
            0.30
        )


    # ========================================================
    # AUDIO
    # ========================================================

    if audio_fake_probability is not None:

        values.append(
            clamp(
                audio_fake_probability
            )
        )

        weights.append(
            0.20
        )


    # ========================================================
    # DOCUMENT
    # ========================================================

    if document_risk is not None:

        values.append(
            clamp(
                document_risk
            )
        )

        weights.append(
            0.10
        )


    # ========================================================
    # NO MODALITY
    # ========================================================

    if not values:

        return {

            "risk_score":
                0.0,

            "risk_percentage":
                0.0,

            "risk_level":
                "UNKNOWN",

            "modalities_used":
                0
        }


    # ========================================================
    # NORMALIZE WEIGHTS
    # ========================================================

    total_weight = sum(
        weights
    )


    # ========================================================
    # CALCULATE RISK
    # ========================================================

    risk = sum(

        value * weight

        for value, weight
        in zip(
            values,
            weights
        )

    ) / total_weight


    # ========================================================
    # RISK LEVEL
    # ========================================================

    if risk >= 0.75:

        level = "HIGH"

    elif risk >= 0.50:

        level = "MEDIUM"

    else:

        level = "LOW"


    # ========================================================
    # RESULT
    # ========================================================

    return {

        "risk_score":
            round(
                risk,
                4
            ),

        "risk_percentage":
            round(
                risk * 100,
                2
            ),

        "risk_level":
            level,

        "modalities_used":
            len(values)
    }