import os
import json
import hashlib
import subprocess
from datetime import datetime, timezone


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_json(command):
    try:
        r = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30
        )

        if r.returncode != 0:
            return None

        return json.loads(r.stdout)

    except Exception:
        return None


def _run_exiftool(path):
    return _run_json(["exiftool", "-j", "-n", path])


def _run_ffprobe(path):
    try:
        r = subprocess.run(
            [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                path
            ],
            capture_output=True,
            text=True,
            timeout=30
        )

        if r.returncode != 0:
            return None

        return json.loads(r.stdout)

    except Exception:
        return None


def _get(data, *keys):
    for key in keys:
        if isinstance(data, dict) and data.get(key) not in (None, "", "N/A"):
            return data.get(key)
    return None


def _parse_time(value):
    if not value:
        return None

    value = str(value).strip()

    try:
        return datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
    except Exception:
        return None


def _time_analysis(metadata):
    candidates = {}

    possible = [
        "DateTimeOriginal",
        "CreateDate",
        "CreationDate",
        "MediaCreateDate",
        "TrackCreateDate",
        "ModifyDate",
        "MediaModifyDate",
        "TrackModifyDate",
        "FileModifyDate"
    ]

    for key in possible:
        value = _get(metadata, key)
        if value:
            candidates[key] = str(value)

    parsed = []

    for key, value in candidates.items():
        dt = _parse_time(value)
        if dt:
            parsed.append((key, dt))

    inconsistency = False
    reason = None

    if len(parsed) >= 2:
        timestamps = [x[1] for x in parsed]
        earliest = min(timestamps)
        latest = max(timestamps)

        # Large gaps are reported as an inconsistency,
        # not automatically as proof of manipulation.
        gap_hours = abs(
            (latest - earliest).total_seconds()
        ) / 3600

        if gap_hours > 24:
            inconsistency = True
            reason = (
                "Multiple embedded timestamps differ by more than "
                "24 hours."
            )

    return {
        "timestamps_found": candidates,
        "consistency": (
            "INCONSISTENT"
            if inconsistency
            else "CONSISTENT_OR_INSUFFICIENT_EVIDENCE"
        ),
        "manipulation_proven": False,
        "note": (
            reason
            if reason
            else
            "No definitive timestamp manipulation can be established "
            "from metadata alone."
        )
    }


def _location(metadata):
    lat = _get(
        metadata,
        "GPSLatitude",
        "GPSLatitudeRef"
    )

    lon = _get(
        metadata,
        "GPSLongitude",
        "GPSLongitudeRef"
    )

    location = _get(
        metadata,
        "Location",
        "City",
        "State",
        "Country"
    )

    if lat is None and lon is None and location is None:
        return {
            "available": False,
            "latitude": None,
            "longitude": None,
            "location": None,
            "message": "No location metadata found."
        }

    return {
        "available": True,
        "latitude": lat,
        "longitude": lon,
        "location": location
    }


def analyze_media_metadata(path):
    ext = os.path.splitext(path)[1].lower()

    exif = _run_exiftool(path)
    metadata = {}

    if exif and isinstance(exif, list) and exif:
        metadata = exif[0]

    ffprobe = None

    if ext in (
        ".mp4", ".mov", ".avi", ".mkv", ".webm",
        ".mp3", ".wav", ".flac", ".ogg", ".m4a"
    ):
        ffprobe = _run_ffprobe(path)

    streams = []
    format_data = {}

    if ffprobe:
        streams = ffprobe.get("streams", [])
        format_data = ffprobe.get("format", {})

    video_stream = next(
        (s for s in streams if s.get("codec_type") == "video"),
        None
    )

    audio_stream = next(
        (s for s in streams if s.get("codec_type") == "audio"),
        None
    )

    source = {
        "device": _get(
            metadata,
            "Make",
            "Model",
            "CameraModelName"
        ),
        "software": _get(
            metadata,
            "Software",
            "Encoder",
            "WritingApplication",
            "EncodedApplicationName"
        ),
        "location": _location(metadata),
    }

    timestamps = _time_analysis(metadata)

    codec = {
        "video_codec": (
            video_stream.get("codec_name")
            if video_stream else None
        ),
        "audio_codec": (
            audio_stream.get("codec_name")
            if audio_stream else None
        ),
        "width": (
            video_stream.get("width")
            if video_stream else None
        ),
        "height": (
            video_stream.get("height")
            if video_stream else None
        ),
        "frame_rate": (
            video_stream.get("r_frame_rate")
            if video_stream else None
        ),
        "duration": _get(
            format_data,
            "duration"
        )
    }

    editing = []

    software = source["software"]

    if software:
        software_text = str(software).lower()

        known_editors = [
            "adobe",
            "premiere",
            "after effects",
            "davinci",
            "resolve",
            "final cut",
            "capcut",
            "ffmpeg",
            "handbrake"
        ]

        for editor in known_editors:
            if editor in software_text:
                editing.append(
                    f"Editing/encoding software metadata contains '{editor}'."
                )

    if timestamps["consistency"] == "INCONSISTENT":
        editing.append(
            "Embedded timestamps show a significant inconsistency."
        )

    return {
        "status": "FORENSIC_ANALYSIS_COMPLETE",
        "file_name": os.path.basename(path),
        "file_size": os.path.getsize(path),
        "sha256": _sha256(path),
        "source": source,
        "timestamps": timestamps,
        "codec": codec,
        "editing_indicators": editing,
        "forensic_conclusion": (
            "FORENSIC_INDICATORS_FOUND"
            if editing
            else "NO_CLEAR_METADATA_INDICATORS"
        ),
        "disclaimer": (
            "Metadata inconsistencies and editing indicators are "
            "forensic signals, not proof of manipulation by themselves."
        )
    }


# ---- TG_VIDEO_PROPS: technical video properties added to the forensic result ----
_TG_VIDEO_EXT = (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm")
_TG_CODECS = {
    "hevc": "HEVC (H.265)", "h264": "H.264 (AVC)", "vp9": "VP9",
    "vp8": "VP8", "av1": "AV1", "mpeg4": "MPEG-4",
}


def _tg_rate(txt):
    try:
        a, b = str(txt).split("/")
        a, b = float(a), float(b)
        return a / b if b else None
    except Exception:
        return None


def _tg_video_properties(path):
    import json, shutil, subprocess
    p = str(path)
    if not p.lower().endswith(_TG_VIDEO_EXT):
        return None
    info = {}
    exe = shutil.which("ffprobe")
    if exe:
        try:
            r = subprocess.run(
                [exe, "-v", "error", "-select_streams", "v:0",
                 "-show_entries",
                 "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames,duration:format=duration",
                 "-of", "json", p],
                capture_output=True, text=True, timeout=30)
            d = json.loads(r.stdout or "{}")
            s = (d.get("streams") or [{}])[0]
            fmt = d.get("format") or {}
            dur = s.get("duration") or fmt.get("duration")
            avg = _tg_rate(s.get("avg_frame_rate"))
            nom = _tg_rate(s.get("r_frame_rate"))
            nb = str(s.get("nb_frames", ""))
            info = {
                "duration_s": float(dur) if dur not in (None, "N/A") else None,
                "width": s.get("width"),
                "height": s.get("height"),
                "codec": s.get("codec_name"),
                "fps": avg or nom,
                "fps_variable": bool(avg and nom and abs(avg - nom) / nom > 0.03),
                "total_frames": int(nb) if nb.isdigit() else None,
            }
        except Exception:
            info = {}
    if not info or not info.get("width"):
        try:
            import cv2
            c = cv2.VideoCapture(p)
            fps = c.get(cv2.CAP_PROP_FPS) or None
            n = int(c.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fc = int(c.get(cv2.CAP_PROP_FOURCC) or 0)
            codec = "".join(chr((fc >> (8 * i)) & 0xFF) for i in range(4)).strip() if fc else None
            info = {
                "duration_s": (n / fps) if (fps and n) else None,
                "width": int(c.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None,
                "height": int(c.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None,
                "codec": codec,
                "fps": fps,
                "fps_variable": False,
                "total_frames": n or None,
            }
            c.release()
        except Exception:
            return None
    codec = info.get("codec")
    if codec:
        info["codec"] = _TG_CODECS.get(str(codec).lower(), str(codec).upper())
    return info


_tg_base_analyze_media_metadata = analyze_media_metadata


def analyze_media_metadata(*args, **kwargs):
    result = _tg_base_analyze_media_metadata(*args, **kwargs)
    try:
        path = args[0] if args else next(iter(kwargs.values()), None)
        info = _tg_video_properties(path) if path else None
        if info and isinstance(result, dict):
            result["video_info"] = info
    except Exception:
        pass
    return result
