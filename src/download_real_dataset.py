import csv
import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

CSV_PATH = r"C:\Users\Pooja\Downloads\archive (1)\FINAL_DATASET.csv"
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

REAL = os.path.join(BASE, "data", "real")
FAKE = os.path.join(BASE, "data", "fake")

os.makedirs(REAL, exist_ok=True)
os.makedirs(FAKE, exist_ok=True)

headers = {
    "User-Agent": "Mozilla/5.0"
}

def download(row):
    label = row["label"].lower()
    image_id = int(row["image_id"])
    url = row["image_url"]

    folder = REAL if label == "real" else FAKE
    path = os.path.join(folder, f"{label}_{image_id:04d}.jpg")

    if os.path.exists(path):
        return "exists"

    for attempt in range(3):
        try:
            r = requests.get(
                url,
                headers=headers,
                timeout=30
            )

            if r.status_code == 200 and len(r.content) > 1000:
                with open(path, "wb") as f:
                    f.write(r.content)
                return "success"

        except Exception:
            pass

        time.sleep(1)

    return "failed"

with open(CSV_PATH, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

print("Total:", len(rows))
print("Retrying missing images...\n")

success = exists = failed = 0

with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(download, r) for r in rows]

    for i, future in enumerate(as_completed(futures), 1):
        result = future.result()

        if result == "success":
            success += 1
        elif result == "exists":
            exists += 1
        else:
            failed += 1

        if i % 100 == 0:
            print(
                f"{i}/{len(rows)} | "
                f"New: {success} | Existing: {exists} | Failed: {failed}"
            )

print("\n===== RETRY COMPLETE =====")
print("New downloads:", success)
print("Already existed:", exists)
print("Still failed:", failed)