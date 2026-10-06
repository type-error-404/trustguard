# TrustGuard AI — Product Build

## Start

On Windows, double-click `start.bat` or run:

```text
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
cd app
python app.py
```

Open `http://127.0.0.1:5050`.

## Important model note

This source archive did not contain the trained model checkpoint binaries. The application has been hardened so that a missing checkpoint does not crash the server.

For actual image/video AI classification, add:

`models/resnet18_deepfake.pt`

For actual audio AI classification, add:

`models/audio_classifier.pt`

Until those files are present, the corresponding classifier reports `MODEL_NOT_READY` rather than inventing a verdict.

## Health check

Open:

`http://127.0.0.1:5050/api/health`

The response reports whether each checkpoint is present.
