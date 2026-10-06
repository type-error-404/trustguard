import os
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset


REAL_DIR = r"C:\Users\Pooja\TrustGuard\data\audio\real"
FAKE_DIR = r"C:\Users\Pooja\TrustGuard\data\audio\fake"

SAMPLE_RATE = 16000
DURATION = 4
MAX_LENGTH = SAMPLE_RATE * DURATION


class AudioDataset(Dataset):

    def __init__(self):
        self.files = []
        self.labels = []

        for file in os.listdir(REAL_DIR):
            if file.endswith(".flac"):
                self.files.append(os.path.join(REAL_DIR, file))
                self.labels.append(0)

        for file in os.listdir(FAKE_DIR):
            if file.endswith(".flac"):
                self.files.append(os.path.join(FAKE_DIR, file))
                self.labels.append(1)

        print("Real:", self.labels.count(0))
        print("Fake:", self.labels.count(1))
        print("Total:", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        path = self.files[index]
        label = self.labels[index]

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

        audio = torch.tensor(audio, dtype=torch.float32)

        return audio, torch.tensor(label, dtype=torch.long)


if __name__ == "__main__":
    dataset = AudioDataset()

    audio, label = dataset[0]

    print("Audio shape:", audio.shape)
    print("Label:", label.item())