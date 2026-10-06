"""
dataset.py
----------
Loads real/fake images and applies augmentation.
"""

import os
import random

from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as T


IMG_SIZE = 128


BASE_TRANSFORM = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),

    T.ToTensor(),

    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


TRAIN_AUGMENT = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),

    T.RandomHorizontalFlip(),

    T.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.1
    ),

    T.RandomApply(
        [
            T.GaussianBlur(
                kernel_size=3,
                sigma=(0.3, 1.2)
            )
        ],
        p=0.3
    ),

    T.ToTensor(),

    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


HARD_FAKE_AUGMENT = T.Compose([
    T.GaussianBlur(
        kernel_size=3,
        sigma=(0.6, 1.6)
    ),

    T.ColorJitter(
        brightness=0.2,
        contrast=0.2
    )
])


class DeepfakeFaceDataset(Dataset):

    def __init__(
        self,
        root,
        split="train",
        train_frac=0.8,
        seed=42,
        augment_hard_fake_p=0.35
    ):

        self.samples = []

        for label, cls in enumerate(
            ["real", "fake"]
        ):

            cls_dir = os.path.join(
                root,
                cls
            )

            if not os.path.isdir(cls_dir):
                continue

            # Walk recursively so frames extracted into
            # data/{real,fake}/<video_name>/ are included.
            for dirpath, _, filenames in os.walk(cls_dir):
                for fname in sorted(filenames):
                    if fname.lower().endswith(
                        (".jpg", ".jpeg", ".png")
                    ):
                        self.samples.append(
                            (
                                os.path.join(dirpath, fname),
                                label
                            )
                        )

        rng = random.Random(seed)

        rng.shuffle(self.samples)

        n_train = int(
            len(self.samples) * train_frac
        )

        if split == "train":
            self.samples = self.samples[:n_train]

        else:
            self.samples = self.samples[n_train:]

        self.split = split

        self.augment_hard_fake_p = (
            augment_hard_fake_p
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path, label = self.samples[idx]

        img = Image.open(
            path
        ).convert("RGB")

        if self.split == "train":

            if (
                label == 1
                and random.random()
                < self.augment_hard_fake_p
            ):

                img = HARD_FAKE_AUGMENT(img)

            img = TRAIN_AUGMENT(img)

        else:

            img = BASE_TRANSFORM(img)

        return img, label