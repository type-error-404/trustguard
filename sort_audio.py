import os
import shutil

PROTOCOL = r"C:\Users\Pooja\TrustGuard\LA\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt"

AUDIO_DIR = r"C:\Users\Pooja\TrustGuard\LA\ASVspoof2019_LA_train\flac"

REAL_DIR = r"C:\Users\Pooja\TrustGuard\data\audio\real"
FAKE_DIR = r"C:\Users\Pooja\TrustGuard\data\audio\fake"

os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(FAKE_DIR, exist_ok=True)

real = 0
fake = 0
missing = 0

with open(PROTOCOL, "r") as f:
    for line in f:
        parts = line.strip().split()

        if len(parts) < 5:
            continue

        filename = parts[1] + ".flac"
        label = parts[4]

        source = os.path.join(AUDIO_DIR, filename)

        if not os.path.exists(source):
            missing += 1
            continue

        if label == "bonafide":
            shutil.copy2(source, os.path.join(REAL_DIR, filename))
            real += 1

        elif label == "spoof":
            shutil.copy2(source, os.path.join(FAKE_DIR, filename))
            fake += 1

print("Done!")
print("Real:", real)
print("Fake:", fake)
print("Missing:", missing)