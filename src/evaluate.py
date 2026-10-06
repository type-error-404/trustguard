"""
evaluate.py
-----------
Evaluates the ResNet-18 model using:

Accuracy
Precision
Recall
F1
AUC
Confusion Matrix
"""

import os

import torch

from torch.utils.data import DataLoader

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from dataset import DeepfakeFaceDataset
from model import build_resnet18


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


def main():

    root = os.path.join(
        os.path.dirname(__file__),
        ".."
    )

    ckpt_path = os.path.join(
        root,
        "models",
        "resnet18_deepfake.pt"
    )

    val_ds = DeepfakeFaceDataset(
        os.path.join(root, "data"),
        split="val"
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=32,
        shuffle=False
    )

    model = build_resnet18(
        pretrained=False
    ).to(DEVICE)

    ckpt = torch.load(
        ckpt_path,
        map_location=DEVICE
    )

    model.load_state_dict(
        ckpt["model_state"]
    )

    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            fake_probs = (
                probabilities[:, 1]
                .cpu()
                .numpy()
            )

            preds = (
                outputs.argmax(dim=1)
                .cpu()
                .numpy()
            )

            all_probs.extend(
                fake_probs
            )

            all_preds.extend(
                preds
            )

            all_labels.extend(
                labels.numpy()
            )

    report = classification_report(
        all_labels,
        all_preds,
        target_names=[
            "real",
            "fake"
        ],
        digits=3
    )

    auc = roc_auc_score(
        all_labels,
        all_probs
    )

    print(report)

    print(
        f"AUC: {auc:.3f}"
    )

    cm = confusion_matrix(
        all_labels,
        all_preds
    )

    fig, ax = plt.subplots(
        figsize=(4.5, 4)
    )

    im = ax.imshow(
        cm,
        cmap="Blues"
    )

    ax.set_xticks(
        [0, 1]
    )

    ax.set_xticklabels(
        ["real", "fake"]
    )

    ax.set_yticks(
        [0, 1]
    )

    ax.set_yticklabels(
        ["real", "fake"]
    )

    ax.set_xlabel(
        "Predicted"
    )

    ax.set_ylabel(
        "Actual"
    )

    ax.set_title(
        f"Confusion Matrix "
        f"(val_acc={ckpt['val_acc']:.3f})"
    )

    for i in range(2):

        for j in range(2):

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color=(
                    "white"
                    if cm[i, j] >
                    cm.max() / 2
                    else "black"
                ),
                fontsize=14,
                fontweight="bold"
            )

    fig.colorbar(im)

    output_dir = os.path.join(
        root,
        "outputs"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    matrix_path = os.path.join(
        output_dir,
        "confusion_matrix.png"
    )

    fig.tight_layout()

    fig.savefig(
        matrix_path,
        dpi=150
    )

    plt.close(fig)

    report_path = os.path.join(
        output_dir,
        "eval_report.txt"
    )

    with open(
        report_path,
        "w"
    ) as f:

        f.write(report)

        f.write(
            f"\nAUC: {auc:.3f}\n"
        )

        f.write(
            f"\nValidation Accuracy: "
            f"{ckpt['val_acc']:.3f}\n"
        )

    print(
        f"Confusion matrix saved: "
        f"{matrix_path}"
    )

    print(
        f"Evaluation report saved: "
        f"{report_path}"
    )


if __name__ == "__main__":
    main()