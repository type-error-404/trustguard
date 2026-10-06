@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo No .venv found. Run TRAIN_VIDEO.bat first.
  pause
  exit /b 1
)
call ".venv\Scripts\activate.bat"
python -c "import torch, cv2, flask; print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); print('OpenCV:', cv2.__version__); print('Flask: OK')"
echo.
if exist "models\resnet18_deepfake.pt" (echo IMAGE/VIDEO CHECKPOINT: FOUND) else (echo IMAGE/VIDEO CHECKPOINT: MISSING)
if exist "models\audio_classifier.pt" (echo AUDIO CHECKPOINT: FOUND) else (echo AUDIO CHECKPOINT: MISSING)
pause
