"""
audio_forensics.py
------------------
TrustGuard audio analysis.

Uses:
- Librosa for signal-level features
- Wav2Vec2 for learned audio representations

Important:
Without a trained fake/real audio checkpoint,
the system does NOT claim an AI deepfake prediction.
It returns extracted forensic information and
MODEL_NOT_READY status.
"""

import os

import numpy as np
import librosa


WAV2VEC_MODEL = (
    "facebook/wav2vec2-base"
)


def extract_audio_features(
    audio_path
):

    y, sr = librosa.load(
        audio_path,
        sr=16000,
        mono=True
    )

    duration = (
        len(y) / sr
        if sr
        else 0
    )

    rms = float(
        np.mean(
            librosa.feature.rms(
                y=y
            )
        )
    )

    zcr = float(
        np.mean(
            librosa.feature.zero_crossing_rate(
                y
            )
        )
    )

    spectral_centroid = float(
        np.mean(
            librosa.feature.spectral_centroid(
                y=y,
                sr=sr
            )
        )
    )

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    mfcc_mean = [
        float(x)
        for x in np.mean(
            mfcc,
            axis=1
        )
    ]

    return {
        "sample_rate": sr,
        "duration_seconds":
            round(duration, 3),
        "rms_energy":
            round(rms, 6),
        "zero_crossing_rate":
            round(zcr, 6),
        "spectral_centroid":
            round(
                spectral_centroid,
                3
            ),
        "mfcc_mean":
            mfcc_mean
    }


def extract_wav2vec_embedding(
    audio_path
):

    try:

        import torch

        from transformers import (
            Wav2Vec2Processor,
            Wav2Vec2Model
        )

        processor = (
            Wav2Vec2Processor
            .from_pretrained(
                WAV2VEC_MODEL
            )
        )

        model = (
            Wav2Vec2Model
            .from_pretrained(
                WAV2VEC_MODEL
            )
        )

        model.eval()

        audio, _ = librosa.load(
            audio_path,
            sr=16000,
            mono=True
        )

        inputs = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        )

        with torch.no_grad():

            output = model(
                **inputs
            )

        embedding = (
            output.last_hidden_state
            .mean(dim=1)
            .squeeze()
            .numpy()
        )

        return {
            "available": True,
            "dimension":
                int(len(embedding)),
            "embedding":
                embedding.tolist()
        }

    except Exception as e:

        return {
            "available": False,
            "error": str(e)
        }


def analyze_audio(
    audio_path
):

    features = (
        extract_audio_features(
            audio_path
        )
    )

    wav2vec = (
        extract_wav2vec_embedding(
            audio_path
        )
    )

    return {
        "modality": "audio",

        "status":
            "MODEL_NOT_READY",

        "message":
            "Audio forensic features and Wav2Vec2 representation were extracted. A trained real/fake audio classifier is required for final deepfake classification.",

        "features":
            features,

        "wav2vec2":
            {
                "model":
                    WAV2VEC_MODEL,

                "available":
                    wav2vec.get(
                        "available",
                        False
                    ),

                "dimension":
                    wav2vec.get(
                        "dimension"
                    )
            }
    }