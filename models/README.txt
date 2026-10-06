TRUSTGUARD MODEL FILES
=======================

The uploaded source archive did not contain trained checkpoint binaries.
Do NOT substitute random weights and do NOT claim those are trained results.

For image/video AI classification, place:
    models/resnet18_deepfake.pt

Expected checkpoint format:
    {"model_state": <PyTorch state dict>}

For audio AI classification, place:
    models/audio_classifier.pt

The application now starts without these files and reports MODEL_NOT_READY
for the affected AI classifier instead of crashing.

Document forensics can still run without an AI document classifier.
