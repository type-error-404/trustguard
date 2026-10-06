"""
train.py
--------
ResNet-18 training with validation and optional MLflow logging.
"""

import os
import argparse

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from dataset import DeepfakeFaceDataset
from model import build_resnet18


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


def run_epoch(
    model,
    loader,
    criterion,
    optimizer=None
):

    is_train = optimizer is not None

    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_train):

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            if is_train:

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

            total_loss += (
                loss.item()
                * images.size(0)
            )

            preds = outputs.argmax(
                dim=1
            )

            correct += (
                preds == labels
            ).sum().item()

            total += images.size(0)

    return (
        total_loss / total,
        correct / total
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_root",
        default=os.path.join(
            os.path.dirname(__file__),
            "..",
            "data"
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--out",
        default=os.path.join(
            os.path.dirname(__file__),
            "..",
            "models",
            "resnet18_deepfake.pt"
        )
    )

    args = parser.parse_args()

    train_ds = DeepfakeFaceDataset(
        args.data_root,
        split="train"
    )

    val_ds = DeepfakeFaceDataset(
        args.data_root,
        split="val"
    )

    print(
        f"Train samples: {len(train_ds)}"
    )

    print(
        f"Val samples: {len(val_ds)}"
    )

    print(
        f"Device: {DEVICE}"
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    model = build_resnet18(
        pretrained=True
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr
    )

    best_val_acc = 0.0

    os.makedirs(
        os.path.dirname(args.out),
        exist_ok=True
    )

    history = []

    # Optional MLflow
    try:
        import mlflow

        mlflow_available = True

        mlflow.set_experiment(
            "TrustGuard-ResNet18"
        )

        mlflow.start_run()

        mlflow.log_params({
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "architecture": "ResNet-18"
        })

    except Exception:
        mlflow_available = False

    for epoch in range(
        1,
        args.epochs + 1
    ):

        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            criterion
        )

        history.append(
            (
                epoch,
                train_loss,
                train_acc,
                val_loss,
                val_acc
            )
        )

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f}"
        )

        if mlflow_available:

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc
                },
                step=epoch
            )

        if val_acc >= best_val_acc:

            best_val_acc = val_acc

            torch.save(
                {
                    "model_state":
                        model.state_dict(),

                    "val_acc":
                        val_acc
                },
                args.out
            )

    print(
        f"Best validation accuracy: "
        f"{best_val_acc:.4f}"
    )

    root_dir = os.path.join(
        os.path.dirname(__file__),
        ".."
    )

    output_dir = os.path.join(
        root_dir,
        "outputs"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    log_path = os.path.join(
        output_dir,
        "training_log.csv"
    )

    with open(
        log_path,
        "w"
    ) as f:

        f.write(
            "epoch,train_loss,train_acc,"
            "val_loss,val_acc\n"
        )

        for row in history:

            f.write(
                ",".join(
                    str(x)
                    for x in row
                )
                + "\n"
            )

    if mlflow_available:

        mlflow.log_artifact(
            log_path
        )

        mlflow.end_run()


if __name__ == "__main__":
    main()