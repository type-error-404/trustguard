"""
generate_demo_data.py
----------------------
TEMPORARY DATA STAND-IN.

The base paper (Hafezi et al., 2026) trains on FFHQ/CelebA (real) vs.
StarGAN/GDWCT/AttGAN/StyleGAN/StyleGAN2/StyleGAN3 (fake) face images.
Those datasets live on Kaggle/Google-Drive and are not reachable from this
sandboxed environment's network allow-list.

This script generates a small SYNTHETIC placeholder dataset with the same
folder layout (data/real, data/fake) so the full pipeline -- loading,
augmentation, ResNet-18 training, evaluation, inference, front-end -- can be
built, run, and demonstrated end-to-end right now.

TO GO TO THE REAL DATASET: download FaceForensics++ / the Kaggle
"140k Real and Fake Faces" set (as used in the base paper) into
data/real/ and data/fake/ with the same *.jpg layout, delete this script's
output, and rerun train.py. No other code changes are required.
"""
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random

IMG_SIZE = 128
N_PER_CLASS = 260  # small, fast to train on CPU for a demo


def make_real_face(rng):
    """Smooth, low-frequency synthetic 'face-like' blob -> stands in for a
    genuine camera-captured face: soft gradients, no repeating artifacts."""
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
    base = rng.randint(120, 220)
    arr = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    cx, cy = IMG_SIZE / 2 + rng.uniform(-8, 8), IMG_SIZE / 2 + rng.uniform(-8, 8)
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    tone = np.array([base + rng.uniform(-10, 10), base * 0.8, base * 0.7])
    for c in range(3):
        arr[..., c] = np.clip(tone[c] - dist * rng.uniform(0.5, 0.9), 20, 255)
    img = Image.fromarray(arr.astype(np.uint8))
    img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(1.5, 3.0)))
    noise = (np.random.default_rng(rng.randint(0, 10_000)).normal(0, 3, (IMG_SIZE, IMG_SIZE, 3)))
    arr2 = np.clip(np.array(img).astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr2)


def make_fake_face(rng):
    """Same base gradient as a 'real' image but with added periodic /
    checkerboard artifacts -- a stand-in for the texture-sticking / aliasing
    signatures the base paper specifically targets in GAN-generated faces."""
    img = make_real_face(rng)
    arr = np.array(img).astype(np.float32)

    # periodic grid artifact (aliasing / texture-sticking stand-in)
    period = rng.choice([6, 8, 10])
    xx, yy = np.meshgrid(np.arange(IMG_SIZE), np.arange(IMG_SIZE))
    grid = (np.sin(2 * np.pi * xx / period) * np.sin(2 * np.pi * yy / period))
    strength = rng.uniform(10, 22)
    for c in range(3):
        arr[..., c] += grid * strength

    # mild checkerboard upsampling artifact in a random patch
    x0, y0 = rng.randint(0, IMG_SIZE - 40), rng.randint(0, IMG_SIZE - 40)
    arr[y0:y0 + 40:2, x0:x0 + 40:2] += rng.uniform(15, 25)

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def main():
    root = os.path.join(os.path.dirname(__file__), "..", "data")
    real_dir = os.path.join(root, "real")
    fake_dir = os.path.join(root, "fake")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    rng = random.Random(42)
    for i in range(N_PER_CLASS):
        make_real_face(rng).save(os.path.join(real_dir, f"real_{i:04d}.jpg"), quality=92)
        make_fake_face(rng).save(os.path.join(fake_dir, f"fake_{i:04d}.jpg"), quality=92)

    print(f"Wrote {N_PER_CLASS} real + {N_PER_CLASS} fake demo images to {root}")


if __name__ == "__main__":
    main()
