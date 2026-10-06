import os
import torch
import numpy as np
import librosa
import torch.nn as nn

from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2Model
)


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "audio_classifier.pt"
)


# ============================================================
# SETTINGS
# ============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

SAMPLE_RATE = 16000
DURATION = 4
MAX_LENGTH = SAMPLE_RATE * DURATION


# ============================================================
# AUDIO CLASSIFIER
# ============================================================

class AudioClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                768,
                128
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                128,
                2
            )
        )

    def forward(self, x):

        return self.net(x)


# ============================================================
# LOAD WAV2VEC2
# ============================================================

print("Loading Wav2Vec2...")

processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base"
)

wav2vec = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-base"
).to(DEVICE)

wav2vec.eval()


# ============================================================
# LOAD AUDIO CLASSIFIER
# ============================================================

print("Loading audio classifier...")

classifier = AudioClassifier().to(
    DEVICE
)

classifier.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

classifier.eval()

print("Audio model loaded successfully.")
print("Device:", DEVICE)


# ============================================================
# AUDIO PREDICTION
# ============================================================

def predict_audio(path):

    # --------------------------------------------------------
    # Load audio
    # --------------------------------------------------------

    audio, _ = librosa.load(
        path,
        sr=SAMPLE_RATE,
        mono=True
    )


    # --------------------------------------------------------
    # Make audio exactly 4 seconds
    # --------------------------------------------------------

    if len(audio) < MAX_LENGTH:

        audio = np.pad(
            audio,
            (
                0,
                MAX_LENGTH - len(audio)
            )
        )

    else:

        audio = audio[
            :MAX_LENGTH
        ]


    # --------------------------------------------------------
    # Wav2Vec2 preprocessing
    # --------------------------------------------------------

    inputs = processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt"
    )


    input_values = (
        inputs.input_values
        .to(DEVICE)
    )


    # --------------------------------------------------------
    # Feature extraction + classification
    # --------------------------------------------------------

    with torch.no_grad():

        output = wav2vec(
            input_values
        ).last_hidden_state


        # 768-dimensional feature vector
        features = output.mean(
            dim=1
        )


        logits = classifier(
            features
        )


        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]


    # --------------------------------------------------------
    # Probabilities
    # --------------------------------------------------------

    real_probability = float(
        probabilities[0]
    )

    fake_probability = float(
        probabilities[1]
    )


    # --------------------------------------------------------
    # Final prediction
    # --------------------------------------------------------

    if fake_probability >= real_probability:

        label = "FAKE"

        confidence = (
            fake_probability
        )

    else:

        label = "REAL"

        confidence = (
            real_probability
        )


    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    return {

        "label": label,

        "confidence": round(
            confidence * 100,
            2
        ),

        "real_probability": round(
            real_probability * 100,
            2
        ),

        "fake_probability": round(
            fake_probability * 100,
            2
        )
    }