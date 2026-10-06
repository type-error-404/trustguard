import os
import sys
import uuid

from flask import (
    Flask,
    request,
    render_template,
    jsonify,
    url_for
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = os.path.dirname(
    BASE_DIR
)

SRC_DIR = os.path.join(
    PROJECT_DIR,
    "src"
)

sys.path.append(
    SRC_DIR
)


# ============================================================
# IMAGE / VIDEO MODEL
# ============================================================

from infer import (
    predict_image,
    predict_video
)


# ============================================================
# BEHAVIORAL VIDEO MODEL
# ============================================================

import numpy as np

from behavioral_features import process_video
from behavioral_detector import predict as predict_behavioral

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# AUDIO MODEL
# ============================================================

# Audio inference is loaded lazily inside the audio route.
# This prevents a missing optional audio checkpoint from crashing startup.


# ============================================================
# DOCUMENT FORENSICS
# ============================================================

from document_forensics import (
    analyze_document
)


# ============================================================
# MEDIA FORENSICS
# ============================================================

from forensic_metadata import (
    analyze_media_metadata
)


# ============================================================
# FUSION
# ============================================================

from fusion import (
    calculate_risk
)


# ============================================================
# STORAGE
# ============================================================

from storage import (
    initialize_database,
    save_scan
)


# ============================================================
# FOLDERS
# ============================================================

UPLOAD_DIR = os.path.join(
    BASE_DIR,
    "uploads"
)

HEATMAP_DIR = os.path.join(
    BASE_DIR,
    "static",
    "heatmaps"
)


os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)

os.makedirs(
    HEATMAP_DIR,
    exist_ok=True
)


# ============================================================
# FLASK
# ============================================================

app = Flask(
    __name__
)

app.config[
    "MAX_CONTENT_LENGTH"
] = 50 * 1024 * 1024


# ============================================================
# DATABASE
# ============================================================

initialize_database()


# ============================================================
# HOME
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# IMAGE / VIDEO / AUDIO SCAN
# ============================================================

@app.route(
    "/api/scan",
    methods=["POST"]
)
def scan():

    file = request.files.get(
        "file"
    )

    if (
        file is None
        or file.filename == ""
    ):

        return jsonify({
            "error":
                "No file uploaded"
        }), 400


    ext = os.path.splitext(
        file.filename
    )[1].lower()


    filename = (
        uuid.uuid4().hex
        + ext
    )


    save_path = os.path.join(
        UPLOAD_DIR,
        filename
    )


    file.save(
        save_path
    )


    heatmap_filename = (
        uuid.uuid4().hex
        + ".jpg"
    )


    heatmap_path = os.path.join(
        HEATMAP_DIR,
        heatmap_filename
    )


    try:

        # ====================================================
        # VIDEO
        # ====================================================

        if ext in (
            ".mp4",
            ".mov",
            ".avi",
            ".mkv",
            ".webm"
        ):

            result = {
                "modality": "video"
            }


            behavioral_model = os.path.join(
                PROJECT_DIR,
                "models",
                "mediapipe",
                "face_landmarker.task"
            )


            if not os.path.exists(
                behavioral_model
            ):

                result.update({
                    "status":
                        "MODEL_NOT_READY",

                    "message":
                        "MediaPipe face landmark model is missing."
                })

            else:

                try:

                    options = vision.FaceLandmarkerOptions(
                        base_options=python.BaseOptions(
                            model_asset_path=behavioral_model
                        ),

                        running_mode=
                            vision.RunningMode.IMAGE,

                        num_faces=1,

                        min_face_detection_confidence=0.5,

                        min_face_presence_confidence=0.5,

                        min_tracking_confidence=0.5
                    )


                    landmarker = (
                        vision.FaceLandmarker
                        .create_from_options(
                            options
                        )
                    )


                    behavioral_output = os.path.join(
                        PROJECT_DIR,
                        "data",
                        "behavioral_features",
                        f"{uuid.uuid4().hex}.npy"
                    )


                    os.makedirs(
                        os.path.dirname(
                            behavioral_output
                        ),
                        exist_ok=True
                    )


                    extracted = process_video(
                        save_path,
                        behavioral_output,
                        landmarker
                    )


                    landmarker.close()


                    if (
                        not extracted
                        or not os.path.exists(
                            behavioral_output
                        )
                    ):

                        result.update({
                            "status":
                                "ANALYSIS_FAILED",

                            "message":
                                "No usable facial movement sequence was detected."
                        })

                    else:

                        behavioral_result = (
                            predict_behavioral(
                                behavioral_output
                            )
                        )


                        frames_analyzed = int(
                            np.load(
                                behavioral_output
                            ).shape[0]
                        )


                        fake_probability = (
                            behavioral_result[
                                "fake_probability"
                            ]
                        )


                        real_probability = (
                            behavioral_result[
                                "real_probability"
                            ]
                        )


                        verdict = (
                            behavioral_result[
                                "verdict"
                            ]
                        )


                        result.update({

                            "modality":
                                "video",

                            "status":
                                "ANALYZED",

                            "label":
                                verdict.lower(),

                            "verdict":
                                verdict,

                            "model":
                                "Behavioral GRU",

                            "framework":
                                "PyTorch",

                            "detection_type":
                                "Behavioral Deepfake Analysis",

                            "fake_probability":
                                fake_probability,

                            "real_probability":
                                real_probability,

                            "confidence":
                                max(
                                    fake_probability,
                                    real_probability
                                ),

                            "frames_analyzed":
                                frames_analyzed,

                            "evidence_count":
                                behavioral_result[
                                    "evidence_count"
                                ],

                            "evidence":
                                behavioral_result[
                                    "evidence"
                                ],

                            "metrics":
                                behavioral_result[
                                    "metrics"
                                ],

                            "metric_breakdown":
                                behavioral_result.get("metric_breakdown", []),

                            "reasons":
                                [
                                    item["message"]
                                    for item in
                                    behavioral_result[
                                        "evidence"
                                    ]
                                ]

                        })


                        # ====================================================
                        # MEDIA FORENSICS
                        # ====================================================

                        result[
                            "forensics"
                        ] = analyze_media_metadata(
                            save_path
                        )


                        try:

                            os.remove(
                                behavioral_output
                            )

                        except OSError:

                            pass


                except Exception as model_error:

                    result.update({

                        "status":
                            "ANALYSIS_FAILED",

                        "model_error":
                            str(model_error),

                        "message":
                            "Behavioral video analysis failed."

                    })


            # ====================================================
            # VIDEO FORENSICS EVEN IF BEHAVIORAL MODEL FAILS
            # ====================================================

            if "forensics" not in result:

                try:

                    result[
                        "forensics"
                    ] = analyze_media_metadata(
                        save_path
                    )

                except Exception as forensic_error:

                    result[
                        "forensics"
                    ] = {
                        "status":
                            "FORENSIC_ANALYSIS_FAILED",

                        "error":
                            str(forensic_error)
                    }


        # ====================================================
        # AUDIO
        # ====================================================

        elif ext in (
            ".wav",
            ".mp3",
            ".flac",
            ".ogg",
            ".m4a"
        ):

            try:

                from audio_infer import predict_audio

                result = predict_audio(
                    save_path
                )

                result[
                    "modality"
                ] = "audio"


            except (
                FileNotFoundError,
                RuntimeError
            ):

                from audio_forensics import analyze_audio

                result = analyze_audio(
                    save_path
                )

                result[
                    "modality"
                ] = "audio"

                result[
                    "status"
                ] = "MODEL_NOT_READY"

                result[
                    "model_error"
                ] = str(
                    model_error
                )


        # ====================================================
        # IMAGE
        # ====================================================

        else:

            try:

                result = predict_image(
                    save_path,
                    heatmap_path
                )

                result[
                    "modality"
                ] = "image"


            except (
                FileNotFoundError,
                RuntimeError
            ) as model_error:

                result = {
                    "modality":
                        "image",

                    "status":
                        "MODEL_NOT_READY",

                    "model_error":
                        str(model_error),

                    "message":
                        "Train the image/video checkpoint first: models/resnet18_deepfake.pt"
                }


            # ====================================================
            # IMAGE FORENSICS
            # ====================================================

            try:

                result[
                    "forensics"
                ] = analyze_media_metadata(
                    save_path
                )

            except Exception as forensic_error:

                result[
                    "forensics"
                ] = {
                    "status":
                        "FORENSIC_ANALYSIS_FAILED",

                    "error":
                        str(forensic_error)
                }


            # ====================================================
            # HEATMAP
            # ====================================================

            if os.path.exists(
                heatmap_path
            ):

                result[
                    "heatmap_url"
                ] = url_for(
                    "static",
                    filename=(
                        f"heatmaps/"
                        f"{heatmap_filename}"
                    )
                )


        # ====================================================
        # RISK CALCULATION
        # ====================================================

        risk = calculate_risk(

            image_fake_probability=(
                result.get(
                    "fake_probability"
                )

                if result["modality"]
                == "image"

                else None
            ),


            video_fake_probability=(
                result.get(
                    "fake_probability"
                )

                if result["modality"]
                == "video"

                else None
            ),


            audio_fake_probability=(
                result.get(
                    "fake_probability",
                    0
                ) / 100

                if result["modality"]
                == "audio"

                else None
            ),


            document_risk=(
                result.get(
                    "document_risk"
                )

                if result["modality"]
                == "document"

                else None
            )

        )


        result[
            "risk"
        ] = risk


        # ====================================================
        # SAVE SCAN
        # ====================================================

        save_scan(

            modality=result.get(
                "modality"
            ),

            label=result.get(
                "label",
                "unknown"
            ),

            confidence=result.get(
                "confidence",
                0
            ),

            risk_score=risk[
                "risk_score"
            ]

        )


        return jsonify(
            result
        )


    except Exception as e:

        print(
            "SCAN ERROR:",
            e
        )


        if os.path.exists(
            heatmap_path
        ):

            try:

                os.remove(
                    heatmap_path
                )

            except OSError:

                pass


        return jsonify({
            "error":
                str(e)
        }), 500


    finally:

        if os.path.exists(
            save_path
        ):

            try:

                os.remove(
                    save_path
                )

            except OSError:

                pass


# ============================================================
# AUDIO ANALYSIS
# ============================================================

@app.route(
    "/api/audio",
    methods=["POST"]
)
def audio_scan():

    file = request.files.get(
        "file"
    )


    if (
        file is None
        or file.filename == ""
    ):

        return jsonify({
            "error":
                "No audio uploaded"
        }), 400


    ext = os.path.splitext(
        file.filename
    )[1].lower()


    allowed = (
        ".wav",
        ".mp3",
        ".flac",
        ".ogg",
        ".m4a"
    )


    if ext not in allowed:

        return jsonify({
            "error":
                "Unsupported audio format"
        }), 400


    filename = (
        uuid.uuid4().hex
        + ext
    )


    path = os.path.join(
        UPLOAD_DIR,
        filename
    )


    file.save(
        path
    )


    try:

        try:

            from audio_infer import predict_audio

            result = predict_audio(
                path
            )

            result[
                "modality"
            ] = "audio"


        except (
            FileNotFoundError,
            RuntimeError
        ) as model_error:

            # Keep the product usable when the optional
            # trained audio model is not bundled.

            from audio_forensics import analyze_audio

            result = analyze_audio(
                path
            )

            result[
                "modality"
            ] = "audio"

            result[
                "status"
            ] = "MODEL_NOT_READY"

            result[
                "model_error"
            ] = str(
                model_error
            )


        return jsonify(
            result
        )


    except Exception as e:

        print(
            "AUDIO ERROR:",
            e
        )


        return jsonify({
            "error":
                str(e)
        }), 500


    finally:

        if os.path.exists(
            path
        ):

            try:

                os.remove(
                    path
                )

            except OSError:

                pass


# ============================================================
# DOCUMENT FORENSICS
# ============================================================

@app.route(
    "/api/document",
    methods=["POST"]
)
def document_scan():

    file = request.files.get(
        "file"
    )


    if (
        file is None
        or file.filename == ""
    ):

        return jsonify({
            "error":
                "No document uploaded"
        }), 400


    ext = os.path.splitext(
        file.filename
    )[1].lower()


    allowed = (
        ".pdf",
        ".jpg",
        ".jpeg",
        ".png",
        ".tiff",
        ".bmp"
    )


    if ext not in allowed:

        return jsonify({
            "error":
                "Unsupported document format"
        }), 400


    filename = (
        uuid.uuid4().hex
        + ext
    )


    path = os.path.join(
        UPLOAD_DIR,
        filename
    )


    file.save(
        path
    )


    try:

        result = analyze_document(
            path
        )


        return jsonify(
            result
        )


    except Exception as e:

        print(
            "DOCUMENT ERROR:",
            e
        )


        return jsonify({
            "error":
                str(e)
        }), 500


    finally:

        if os.path.exists(
            path
        ):

            try:

                os.remove(
                    path
                )

            except OSError:

                pass


# ============================================================
# HEALTH
# ============================================================

@app.route(
    "/api/health"
)
def health():

    return jsonify({

        "model_files": {

            "image_video":
                os.path.exists(
                    os.path.join(
                        PROJECT_DIR,
                        "models",
                        "resnet18_deepfake.pt"
                    )
                ),

            "audio":
                os.path.exists(
                    os.path.join(
                        PROJECT_DIR,
                        "models",
                        "audio_classifier.pt"
                    )
                )

        },


        "project":
            "TrustGuard AI",


        "status":
            "online",


        "modules": {

            "image":
                "active",

            "video":
                "active",

            "audio":
                "ready_if_checkpoint_present",

            "document":
                "forensics_ready",

            "media_forensics":
                "active",

            "fusion":
                "active",

            "biometrics":
                "not_implemented",

            "real_time_meeting":
                "not_implemented"

        }

    })


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5050,
        debug=False
    )