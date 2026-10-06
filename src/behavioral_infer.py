import os
import tempfile
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp

from behavioral_features import process_video

VIDEO = "data/raw_videos/real/1.mp4"
GRU_MODEL = "models/behavioral_gru.pt"
FACE_MODEL = "models/mediapipe/face_landmarker.task"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BehavioralGRU(nn.Module):
    def __init__(self):
        super().__init__()

        self.gru = nn.GRU(
            input_size=76,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        output, _ = self.gru(x)
        return self.classifier(output[:, -1, :])


checkpoint = torch.load(
    GRU_MODEL,
    map_location=device
)

model = BehavioralGRU().to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=FACE_MODEL
    ),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
    options
)


temp_file = os.path.join(
    tempfile.gettempdir(),
    "trustguard_behavioral_test.npy"
)

try:
    success = process_video(
        VIDEO,
        temp_file,
        landmarker
    )
finally:
    landmarker.close()


if not success:
    raise RuntimeError(
        "Behavioral feature extraction failed."
    )


features = np.load(temp_file).astype(
    np.float32
)

print()
print("Extracted sequence:", features.shape)

if features.shape[1] != 76:
    raise RuntimeError(
        f"Expected 76 features, got {features.shape[1]}"
    )


# The GRU expects exactly 30 frames.
if features.shape[0] < 30:
    last = features[-1:]
    padding = np.repeat(
        last,
        30 - features.shape[0],
        axis=0
    )
    features = np.concatenate(
        [features, padding],
        axis=0
    )
else:
    indexes = np.linspace(
        0,
        features.shape[0] - 1,
        30
    ).astype(int)

    features = features[indexes]


x = torch.tensor(
    features,
    dtype=torch.float32
).unsqueeze(0).to(device)


with torch.no_grad():
    logits = model(x)

    probabilities = torch.softmax(
        logits,
        dim=1
    )[0]

    prediction = int(
        torch.argmax(probabilities)
    )


real_probability = float(
    probabilities[0]
)

fake_probability = float(
    probabilities[1]
)


print()
print("=" * 50)
print("TRUSTGUARD BEHAVIORAL ANALYSIS")
print("=" * 50)
print("Video:", VIDEO)
print("Device:", device)
print("Sequence:", features.shape)
print(
    f"REAL probability: {real_probability:.2%}"
)
print(
    f"FAKE probability: {fake_probability:.2%}"
)
print(
    "VERDICT:",
    "FAKE" if prediction == 1 else "REAL"
)
print("=" * 50)

os.remove(temp_file)
