import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

DATA = "data/audio_features.npz"
MODEL = "models/audio_classifier.pt"

data = np.load(DATA)

X = data["features"]
y = data["labels"]

print("Features:", X.shape)
print("Labels:", y.shape)

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)

train_data = TensorDataset(X_train, y_train)
val_data = TensorDataset(X_val, y_val)

train_loader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    val_data,
    batch_size=64
)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Train:", len(train_data))
print("Validation:", len(val_data))
print("Device:", device)


class AudioClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.net(x)


model = AudioClassifier().to(device)

# Handle dataset imbalance
class_counts = np.bincount(y_train.numpy())

weights = len(y_train) / (
    2 * torch.tensor(class_counts, dtype=torch.float32)
)

weights = weights.to(device)

criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

EPOCHS = 10

for epoch in range(EPOCHS):

    model.train()
    total_loss = 0

    for features, labels in train_loader:

        features = features.to(device)
        labels = labels.to(device)

        output = model(features)

        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    model.eval()

    predictions = []
    actual = []

    with torch.no_grad():

        for features, labels in val_loader:

            features = features.to(device)

            output = model(features)

            pred = output.argmax(dim=1)

            predictions.extend(pred.cpu().numpy())
            actual.extend(labels.numpy())

    acc = accuracy_score(actual, predictions)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"Loss: {total_loss / len(train_loader):.4f} "
        f"Val Accuracy: {acc:.4f}"
    )


# Final evaluation

precision = precision_score(
    actual,
    predictions,
    zero_division=0
)

recall = recall_score(
    actual,
    predictions,
    zero_division=0
)

f1 = f1_score(
    actual,
    predictions,
    zero_division=0
)

print("\n===== AUDIO MODEL RESULTS =====")
print(f"Accuracy : {accuracy_score(actual, predictions):.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(actual, predictions))

print("\nClassification Report:")
print(
    classification_report(
        actual,
        predictions,
        target_names=["Real", "Fake"],
        zero_division=0
    )
)

torch.save(model.state_dict(), MODEL)

print("\nAudio classifier saved to:")
print(MODEL)