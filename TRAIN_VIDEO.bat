@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Creating Python 3.11 environment...
  py -3.11 -m venv .venv || exit /b 1
)
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if not exist "data\raw_videos\real" mkdir "data\raw_videos\real"
if not exist "data\raw_videos\fake" mkdir "data\raw_videos\fake"
echo.
echo Put REAL videos in data\raw_videos\real and FAKE videos in data\raw_videos\fake.
pause
python src\frame_extraction.py --raw_root data\raw_videos --out_root data
python src\train.py --data_root data --epochs 15 --batch_size 32 --lr 0.0001 --out models\resnet18_deepfake.pt
python src\evaluate.py
echo.
echo Training and evaluation finished.
echo Check models\resnet18_deepfake.pt and outputs\eval_report.txt
pause
