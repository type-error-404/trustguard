# TrustGuard AI — Windows Setup & Training

This package contains the application and the real training pipeline. It does **not** ship a fake/random trained checkpoint. The checkpoint is created from your own real/fake dataset.

## 1. Install Python 3.11

Use Python 3.11 on Windows. Python 3.14 can cause compatibility problems with some ML packages.

Check:

```bat
py -3.11 --version
```

## 2. Create the environment

From the project root:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Check the GPU

```bat
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 4. Put your video dataset here

```text
data\raw_videos\real\*.mp4
 data\raw_videos\fake\*.mp4
```

Keep real and fake videos in separate folders.

## 5. Convert videos into face frames

```bat
python src\frame_extraction.py --raw_root data\raw_videos --out_root data
```

This creates:

```text
data\real\<video_name>\frame_0000.jpg
 data\fake\<video_name>\frame_0000.jpg
```

The training dataset loader searches recursively for these frames.

## 6. Train the ResNet-18 detector

Recommended first run:

```bat
python src\train.py --data_root data --epochs 15 --batch_size 32 --lr 0.0001 --out models\resnet18_deepfake.pt
```

The trained checkpoint will be created at:

```text
models\resnet18_deepfake.pt
```

## 7. Evaluate it

```bat
python src\evaluate.py
```

Results are written to:

```text
outputs\eval_report.txt
outputs\confusion_matrix.png
outputs\training_log.csv
```

## 8. Start TrustGuard

```bat
cd app
python app.py
```

Open:

```text
http://127.0.0.1:5050
```

Health check:

```text
http://127.0.0.1:5050/api/health
```

The health response must show:

```json
"image_video": true
```

before Image/Video AI classification is considered ready.

## Important

Do not judge the model from training accuracy alone. Use videos that were not used for training and inspect precision, recall, F1 and the confusion matrix. If real videos are being flagged as fake, adjust the dataset/split and retrain rather than hiding the error in the UI.
