import os
import torch
import numpy as np
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from tqdm import tqdm

REAL_DIR = r"C:\Users\Pooja\TrustGuard\data\audio\real"
FAKE_DIR = r"C:\Users\Pooja\TrustGuard\data\audio\fake"

OUTPUT = r"C:\Users\Pooja\TrustGuard\data\audio_features.npz"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
DURATION = 4
MAX_LENGTH = SAMPLE_RATE * DURATION
BATCH_SIZE = 4

print("Device:", DEVICE)

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-base"
).to(DEVICE)

model.eval()

files = []
labels = []

for f in os.listdir(REAL_DIR):
    if f.endswith(".flac"):
        files.append(os.path.join(REAL_DIR, f))
        labels.append(0)

for f in os.listdir(FAKE_DIR):
    if f.endswith(".flac"):
        files.append(os.path.join(FAKE_DIR, f))
        labels.append(1)

print("Total files:", len(files))

features = []

for start in tqdm(range(0, len(files), BATCH_SIZE)):

    batch_files = files[start:start + BATCH_SIZE]
    batch_audio = []

    for path in batch_files:

        audio, _ = librosa.load(
            path,
            sr=SAMPLE_RATE,
            mono=True
        )

        if len(audio) < MAX_LENGTH:
            audio = np.pad(
                audio,
                (0, MAX_LENGTH - len(audio))
            )
        else:
            audio = audio[:MAX_LENGTH]

        batch_audio.append(audio)

    inputs = processor(
        batch_audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True
    )

    input_values = inputs.input_values.to(DEVICE)

    with torch.no_grad():
        output = model(input_values).last_hidden_state
        batch_features = output.mean(dim=1)

    features.extend(batch_features.cpu().numpy())

    if (start // BATCH_SIZE) % 100 == 0:
        print("Processed:", min(start + BATCH_SIZE, len(files)))

np.savez_compressed(
    OUTPUT,
    features=np.array(features, dtype=np.float32),
    labels=np.array(labels, dtype=np.int64)
)

print("Feature extraction complete!")
print("Features shape:", np.array(features).shape)
print("Saved to:", OUTPUT)