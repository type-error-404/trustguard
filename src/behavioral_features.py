import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_DIR = os.path.join(BASE, "data", "raw_videos")
OUT_DIR = os.path.join(BASE, "data", "behavioral_features")
MODEL = os.path.join(BASE, "models", "mediapipe", "face_landmarker.task")

os.makedirs(OUT_DIR, exist_ok=True)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def ear(p, idx):
    return (
        dist(p[idx[1]], p[idx[5]]) +
        dist(p[idx[2]], p[idx[4]])
    ) / (2 * max(dist(p[idx[0]], p[idx[3]]), 1e-6))

def frame_features(face):
    p = np.array([[x.x, x.y, x.z] for x in face], dtype=np.float32)

    left_ear = ear(p[:, :2], LEFT_EYE)
    right_ear = ear(p[:, :2], RIGHT_EYE)

    eye_center_x = (
        p[33, 0] + p[133, 0] +
        p[362, 0] + p[263, 0]
    ) / 4

    eye_center_y = (
        p[33, 1] + p[133, 1] +
        p[362, 1] + p[263, 1]
    ) / 4

    nose = p[1]

    mouth_v = dist(p[13, :2], p[14, :2])
    mouth_h = dist(p[61, :2], p[291, :2])
    mouth_ratio = mouth_v / max(mouth_h, 1e-6)

    selected = [
        1, 33, 133, 362, 263,
        61, 291, 13, 14
    ]

    landmarks = p[selected].flatten()

    features = np.array([
        left_ear,
        right_ear,
        (left_ear + right_ear) / 2,
        eye_center_x,
        eye_center_y,
        nose[0],
        nose[1],
        nose[2],
        mouth_v,
        mouth_h,
        mouth_ratio,
        *landmarks
    ], dtype=np.float32)

    return features

def process_video(path, output, landmarker):
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        print("[FAIL] Cannot open:", path)
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25

    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if count > 0:
        indexes = np.linspace(
            0,
            count - 1,
            min(30, count),
            dtype=int
        )
        indexes = set(indexes.tolist())
    else:
        indexes = None

    seq = []
    frame_no = 0
        
    while True:
        ok, frame = cap.read()

        if not ok:
            break

        if indexes is None:
            process = frame_no % max(1, int(fps / 2)) == 0
        else:
            process = frame_no in indexes

        if process:
            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            # MediaPipe requires strictly increasing timestamps.
            result = landmarker.detect(image)

            
            if result.face_landmarks:
                seq.append(
                    frame_features(
                        result.face_landmarks[0]
                    )
                )

        frame_no += 1

    cap.release()

    if len(seq) < 2:
        print("[FAIL] No usable face sequence:", path)
        return False

    seq = np.asarray(seq, dtype=np.float32)

    change = np.zeros_like(seq)
    change[1:] = seq[1:] - seq[:-1]

    final = np.concatenate([seq, change], axis=1)

    np.save(output, final)

    print(
        "[OK]",
        os.path.basename(path),
        "->",
        final.shape
    )

    return True

def main():
    if not os.path.exists(MODEL):
        print("[ERROR] Missing:", MODEL)
        return

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=MODEL
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    landmarker = vision.FaceLandmarker.create_from_options(options)

    total = 0
    success = 0

    try:
        for label in ["real", "fake"]:
            source = os.path.join(VIDEO_DIR, label)
            destination = os.path.join(OUT_DIR, label)
            os.makedirs(destination, exist_ok=True)

            files = [
                f for f in os.listdir(source)
                if f.lower().endswith(
                    (".mp4", ".mov", ".avi", ".mkv", ".webm")
                )
            ]

            print()
            print("PROCESSING:", label.upper())

            for file in files:
                total += 1

                inp = os.path.join(source, file)
                name = os.path.splitext(file)[0]
                out = os.path.join(destination, name + ".npy")

                try:
                    if process_video(inp, out, landmarker):
                        success += 1
                except Exception as e:
                    print("[FAIL]", file, "->", e)

    finally:
        landmarker.close()

    print()
    print("=" * 50)
    print("BEHAVIORAL EXTRACTION COMPLETE")
    print("=" * 50)
    print("Total:", total)
    print("Success:", success)
    print("Failed:", total - success)

if __name__ == "__main__":
    main()
