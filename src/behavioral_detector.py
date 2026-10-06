import os
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn as nn

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE, "models", "behavioral_gru_v1.pt")
FEATURE_DIR = os.path.join(
    BASE,
    "data",
    "large_dataset",
    "behavioral_features_large"
)


class BehavioralGRU(nn.Module):

    def __init__(self, input_size=76, hidden_size=128, num_layers=2):

        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x, lengths):

        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, hidden = self.gru(packed)

        last_hidden = hidden[-1]

        return self.classifier(last_hidden)


def load_model():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = BehavioralGRU().to(device)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    return model, device


def load_features(path):

    x = np.load(path).astype(np.float32)

    if x.ndim != 2:
        raise ValueError("Behavioral feature file must be 2D")

    if x.shape[1] != 76:
        raise ValueError(
            f"Expected 76 features, got {x.shape[1]}"
        )

    return x


def calculate_metrics(x):

    base = x[:, :38]

    eye_open = base[:, 2]
    eye_center = base[:, 3:5]
    nose = base[:, 5:8]
    mouth = base[:, 8:11]
    landmarks = base[:, 11:]

    eye_motion = np.linalg.norm(
        np.diff(eye_center, axis=0),
        axis=1
    )

    head_motion = np.linalg.norm(
        np.diff(nose[:, :2], axis=0),
        axis=1
    )

    mouth_motion = np.linalg.norm(
        np.diff(mouth, axis=0),
        axis=1
    )

    landmark_motion = np.linalg.norm(
        np.diff(landmarks, axis=0),
        axis=1
    )

    temporal_change = np.linalg.norm(
        x[1:] - x[:-1],
        axis=1
    )

    eye_threshold = np.median(eye_open) * 0.7

    eye_states = eye_open < eye_threshold

    transitions = int(
        np.sum(
            eye_states[1:] != eye_states[:-1]
        )
    )

    return {
        "eye_openness_mean": float(np.mean(eye_open)),
        "eye_openness_std": float(np.std(eye_open)),
        "eye_motion_mean": float(np.mean(eye_motion)),
        "eye_motion_std": float(np.std(eye_motion)),
        "head_motion_mean": float(np.mean(head_motion)),
        "head_motion_std": float(np.std(head_motion)),
        "mouth_motion_mean": float(np.mean(mouth_motion)),
        "mouth_motion_std": float(np.std(mouth_motion)),
        "landmark_motion_mean": float(np.mean(landmark_motion)),
        "landmark_motion_std": float(np.std(landmark_motion)),
        "temporal_irregularity": float(np.std(temporal_change)),
        "eye_state_transitions": transitions
    }



def build_real_reference(feature_dir):

    reference = {}

    files = list(
        Path(feature_dir).glob("*.npy")
    )

    if not files:
        print("[WARNING] No REAL feature files found:", feature_dir)
        return reference

    collected = {}

    for file in files:
        try:
            x = load_features(str(file))
            metrics = calculate_metrics(x)

            for key, value in metrics.items():

                value = float(value)

                if not np.isfinite(value):
                    continue

                collected.setdefault(
                    key, []
                ).append(value)

        except Exception as e:
            print(
                "[REFERENCE SKIP]",
                file.name,
                "->",
                e
            )

    for key, values in collected.items():

        if not values:
            continue

        values = np.asarray(
            values,
            dtype=np.float32
        )

        reference[key] = (
            float(np.percentile(values, 5)),
            float(np.percentile(values, 95))
        )

    print(
        "[REFERENCE] REAL files:",
        len(files),
        "| metrics:",
        len(reference)
    )

    return reference


def analyze_against_real(metrics, reference):
    anomalies = []

    checks = {
        "eye_openness_mean": "Eye openness pattern differs from the normal real-video range.",
        "eye_openness_std": "Eye openness variation differs from the normal real-video range.",
        "eye_motion_mean": "Eye movement pattern differs from the normal real-video range.",
        "eye_motion_std": "Eye movement variation differs from the normal real-video range.",
        "head_motion_mean": "Head movement pattern differs from the normal real-video range.",
        "head_motion_std": "Head movement variation differs from the normal real-video range.",
        "mouth_motion_mean": "Mouth-motion pattern is outside the normal real-video range.",
        "mouth_motion_std": "Mouth-motion variation is outside the normal real-video range.",
        "landmark_motion_mean": "Facial landmark movement differs from the normal real-video range.",
        "landmark_motion_std": "Facial landmark movement variation differs from the normal real-video range.",
        "temporal_irregularity": "Temporal facial movement irregularity is outside the normal real-video range.",
        "eye_state_transitions": "Eye-state transition frequency differs from the normal real-video range."
    }

    for key, message in checks.items():
        if key not in metrics or key not in reference:
            continue

        value = float(metrics[key])
        low, high = reference[key]

        if value < low or value > high:
            anomalies.append({
                "metric": key,
                "message": message,
                "value": value,
                "normal_range": (
                    float(low),
                    float(high)
                )
            })

    return anomalies


# ---- TG: cached reference + magnitude-aware, noise-resistant evidence ----
TG_STRONG_RATIO = 0.25   # one metric this far beyond its range is strong evidence
TG_MIN_CLUSTER = 4       # or this many metrics outside their range together

_TG_REF_CACHE = {}


def _tg_cached_reference(feature_dir):
    key = str(feature_dir)
    if key not in _TG_REF_CACHE:
        _TG_REF_CACHE[key] = build_real_reference(feature_dir)
    return _TG_REF_CACHE[key]


_tg_raw_analyze_against_real = analyze_against_real


def analyze_against_real(metrics, reference):
    raw = _tg_raw_analyze_against_real(metrics, reference)
    enriched = []
    for a in raw:
        low, high = a["normal_range"]
        width = max(high - low, 1e-9)
        v = a["value"]
        if v > high:
            beyond, side = (v - high) / width, "above"
        else:
            beyond, side = (low - v) / width, "below"
        item = dict(a)
        item["deviation_ratio"] = round(beyond, 3)
        item["direction"] = side
        base = str(a["message"]).rstrip(".")
        item["message"] = (
            f"{base} (measured {v:.4g}, {side} the {low:.4g} to {high:.4g} "
            f"range by {beyond * 100:.0f}% of the range width)."
        )
        enriched.append(item)

    enriched.sort(key=lambda x: x["deviation_ratio"], reverse=True)
    strong = [x for x in enriched if x["deviation_ratio"] >= TG_STRONG_RATIO]
    if strong or len(enriched) >= TG_MIN_CLUSTER:
        return enriched
    return []


def predict(feature_path):

    model, device = load_model()

    x = load_features(feature_path)

    tensor = torch.from_numpy(
        x
    ).unsqueeze(0).to(device)

    lengths = torch.tensor(
        [len(x)],
        dtype=torch.long
    )

    with torch.no_grad():

        logits = model(
            tensor,
            lengths
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

    real_probability = float(
        probabilities[0].item()
    )

    fake_probability = float(
        probabilities[1].item()
    )

    metrics = calculate_metrics(x)

    reference = _tg_cached_reference(os.path.join(BASE, "data", "large_dataset", "behavioral_features_large", "real"))

    evidence = analyze_against_real(
        metrics,
        reference
    )

    return {
        "verdict":
            "FAKE"
            if fake_probability >= real_probability
            else "REAL",

        "fake_probability":
            fake_probability,

        "real_probability":
            real_probability,

        "device":
            str(device),

        "evidence_count":
            len(evidence),

        "evidence":
            evidence,

        "metrics":
            metrics
    }


def print_result(result):

    print()
    print("=" * 60)
    print("TRUSTGUARD BEHAVIORAL VERIFICATION")
    print("=" * 60)

    print()
    print("VERDICT:", result["verdict"])

    print(
        "Fake probability:",
        f'{result["fake_probability"] * 100:.2f}%'
    )

    print(
        "Real probability:",
        f'{result["real_probability"] * 100:.2f}%'
    )

    print(
        "Device:",
        result["device"]
    )

    print()
    print("WHY FLAGGED")

    if not result["evidence"]:

        print(
            "No measured behavioral anomaly detected."
        )

    else:

        for item in result["evidence"]:

            print(
                "[ANOMALY]",
                item["message"]
            )

            print(
                "  value =",
                round(item["value"], 6)
            )

            print(
                "  normal =",
                [
                    round(v, 6)
                    for v in item["normal_range"]
                ]
            )

    print()
    print("=" * 60)


if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "Usage: python src/behavioral_detector.py <feature.npy>"
        )

        sys.exit(1)

    result = predict(sys.argv[1])

    print_result(result)







# ---- TG: per-metric breakdown vs the real-video reference ----
_tg_raw_predict = predict


def predict(*args, **kwargs):
    result = _tg_raw_predict(*args, **kwargs)
    try:
        ref = _tg_cached_reference(os.path.join(BASE, "data", "large_dataset", "behavioral_features_large", "real"))
        rows = []
        for key, val in (result.get("metrics") or {}).items():
            if key not in ref:
                continue
            low, high = ref[key]
            v = float(val)
            width = max(high - low, 1e-9)
            if v > high:
                status, beyond = "above", (v - high) / width
            elif v < low:
                status, beyond = "below", (low - v) / width
            else:
                status, beyond = "inside", 0.0
            rows.append({"metric": key, "value": v, "low": float(low), "high": float(high),
                         "status": status, "beyond_pct": round(beyond * 100, 1)})
        result["metric_breakdown"] = rows
    except Exception:
        pass
    return result
