# TrustGuard AI

TrustGuard AI is a Flask + PyTorch deepfake/forensics application with an Image/Video ResNet-18 pipeline, audio forensics hooks, document forensics, risk fusion, and a web interface.

## What is included

- Image classification with ResNet-18
- Video classification by extracting face frames and aggregating frame probabilities
- Grad-CAM focus map for image predictions
- Audio forensics endpoint and optional trained audio classifier
- Document metadata/OCR/integrity analysis
- Risk fusion layer
- SQLite scan storage
- Flask web UI
- Windows setup/training scripts

## Important: trained weights are not bundled

The source archive supplied for this build did not contain the trained `.pt`/`.pth` model binaries. This package therefore does **not** pretend to contain a trained detector.

The application starts without the weights and reports `MODEL_NOT_READY` instead of inventing a fake/real verdict.

After training, the main checkpoint must exist at:

```text
models/resnet18_deepfake.pt
```

Optional audio checkpoint:

```text
models/audio_classifier.pt
```

## Train the Image/Video model on your real dataset

Use Python 3.11 on Windows.

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Put videos here:

```text
data/raw_videos/real/*.mp4
data/raw_videos/fake/*.mp4
```

Extract face frames:

```bat
python src\frame_extraction.py --raw_root data\raw_videos --out_root data
```

Train:

```bat
python src\train.py --data_root data --epochs 15 --batch_size 32 --lr 0.0001 --out models\resnet18_deepfake.pt
```

Evaluate:

```bat
python src\evaluate.py
```

Then start the application:

```bat
cd app
python app.py
```

Open `http://127.0.0.1:5050`.

For the exact Windows workflow, see `SETUP_AND_TRAIN.md`.

## Do not report training accuracy as product accuracy

Use a held-out test set containing videos that were not used to train the model. Report precision, recall, F1, confusion matrix and false-positive/false-negative counts. The project should not hide incorrect classifications in the UI.
