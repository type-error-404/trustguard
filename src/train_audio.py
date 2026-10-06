import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import Wav2Vec2Model
from audio_dataset import AudioDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
EPOCHS = 5
LR = 0.001

dataset = AudioDataset()

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_set, val_set = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

print("Train:", len(train_set))
print("Validation:", len(val_set))
print("Device:", DEVICE)

wav2vec = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-base"
).to(DEVICE)

for p in wav2vec.parameters():
    p.requires_grad = False

classifier = nn.Sequential(
    nn.Linear(768, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 2)
).to(DEVICE)

weights = torch.tensor([4.5, 0.5], device=DEVICE)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(classifier.parameters(), lr=LR)

for epoch in range(EPOCHS):

    classifier.train()
    total_loss = 0

    for audio, labels in train_loader:

        audio = audio.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.no_grad():
            features = wav2vec(audio).last_hidden_state.mean(dim=1)

        output = classifier(features)

        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    classifier.eval()
    correct = 0
    total = 0

    with torch.no_grad():

        for audio, labels in val_loader:

            audio = audio.to(DEVICE)
            labels = labels.to(DEVICE)

            features = wav2vec(audio).last_hidden_state.mean(dim=1)
            output = classifier(features)

            predicted = output.argmax(dim=1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"Loss: {total_loss / len(train_loader):.4f} "
        f"Val Accuracy: {accuracy:.4f}"
    )

torch.save(
    classifier.state_dict(),
    "models/audio_classifier.pt"
)

print("Audio model saved!")