"""
frame_extraction.py
--------------------
Extracts frames from an input video at a fixed FPS and (optionally) crops
the face region using OpenCV's Haar cascade, so downstream code only ever
deals with still images -- the same input type the ResNet-18 classifier
(and the base paper's model) is trained on.

Used by: app/app.py for video uploads, and can be run standalone to build
a frame dataset out of raw video files (data/raw_videos/{real,fake}/*.mp4).
"""
import os
import cv2

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def extract_frames(video_path, out_dir, every_n_frames=15, max_frames=30, face_crop=True):
    """Extract up to `max_frames` frames from `video_path` into `out_dir`,
    sampling one frame every `every_n_frames`. Returns list of saved paths."""
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    saved = []
    idx, saved_count = 0, 0
    while cap.isOpened() and saved_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % every_n_frames == 0:
            crop = frame
            if face_crop:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5)
                if len(faces) > 0:
                    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                    pad = int(0.2 * w)
                    x0, y0 = max(0, x - pad), max(0, y - pad)
                    x1, y1 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
                    crop = frame[y0:y1, x0:x1]
            out_path = os.path.join(out_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(out_path, crop)
            saved.append(out_path)
            saved_count += 1
        idx += 1

    cap.release()
    return saved


def build_dataset_from_videos(raw_root, out_root, every_n_frames=15, max_frames=30):
    """Walk data/raw_videos/{real,fake}/*.mp4 -> data/{real,fake}/*.jpg"""
    for label in ("real", "fake"):
        video_dir = os.path.join(raw_root, label)
        out_dir = os.path.join(out_root, label)
        if not os.path.isdir(video_dir):
            continue
        for fname in os.listdir(video_dir):
            if not fname.lower().endswith((".mp4", ".mov", ".avi")):
                continue
            video_path = os.path.join(video_dir, fname)
            tmp_dir = os.path.join(out_dir, os.path.splitext(fname)[0])
            extract_frames(video_path, tmp_dir, every_n_frames, max_frames)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--raw_root", default="data/raw_videos")
    p.add_argument("--out_root", default="data")
    args = p.parse_args()
    build_dataset_from_videos(args.raw_root, args.out_root)
    print("Frame extraction complete.")
