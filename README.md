# TrustGuard AI

Multi-modal deepfake detection (video, image, audio, document).

## Run
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install torch   # pick the build for your machine at pytorch.org
python app/app.py   # http://127.0.0.1:5050

## Not in this repo (share via Google Drive)
- Raw videos / images / audio datasets
- Original deepfake datasets (request access yourself: FaceForensics++, Celeb-DF, DFDC)
