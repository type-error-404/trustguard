"""
document_forensics.py
---------------------
Document forensic analysis.

Uses:
- Tesseract OCR
- SHA-256 integrity hash
- PDF/image metadata
- ExifTool when installed
- OpenSSL when available
"""

import os
import hashlib
import json
import subprocess

from PIL import Image


def calculate_sha256(
    file_path
):

    sha256 = hashlib.sha256()

    with open(
        file_path,
        "rb"
    ) as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            sha256.update(
                chunk
            )

    return sha256.hexdigest()


def run_exiftool(
    file_path
):

    try:

        result = subprocess.run(
            [
                "exiftool",
                "-j",
                file_path
            ],
            capture_output=True,
            text=True,
            timeout=20
        )

        if result.returncode != 0:

            return {
                "available": False,
                "error":
                    result.stderr.strip()
            }

        data = json.loads(
            result.stdout
        )

        return {
            "available": True,
            "data":
                data[0]
                if data
                else {}
        }

    except FileNotFoundError:

        return {
            "available": False,
            "error":
                "ExifTool is not installed or not available in PATH."
        }

    except Exception as e:

        return {
            "available": False,
            "error": str(e)
        }


def run_openssl(
    file_path
):

    try:

        result = subprocess.run(
            [
                "openssl",
                "dgst",
                "-sha256",
                file_path
            ],
            capture_output=True,
            text=True,
            timeout=20
        )

        if result.returncode != 0:

            return {
                "available": False,
                "error":
                    result.stderr.strip()
            }

        return {
            "available": True,
            "output":
                result.stdout.strip()
        }

    except FileNotFoundError:

        return {
            "available": False,
            "error":
                "OpenSSL is not installed or not available in PATH."
        }

    except Exception as e:

        return {
            "available": False,
            "error": str(e)
        }


def perform_ocr(
    file_path
):

    try:

        import pytesseract

        text = ""

        ext = os.path.splitext(
            file_path
        )[1].lower()

        if ext in (
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tiff"
        ):

            image = Image.open(
                file_path
            )

            text = pytesseract.image_to_string(
                image
            )

        elif ext == ".pdf":

            try:

                import fitz

                pdf = fitz.open(
                    file_path
                )

                pages = []

                for page in pdf:

                    pages.append(
                        page.get_text()
                    )

                text = "\n".join(
                    pages
                )

            except Exception:
                text = ""

        return {
            "available": True,
            "text": text[:10000],
            "characters":
                len(text)
        }

    except Exception as e:

        return {
            "available": False,
            "error": str(e)
        }


def analyze_document(
    file_path
):

    sha256 = calculate_sha256(
        file_path
    )

    ocr = perform_ocr(
        file_path
    )

    exif = run_exiftool(
        file_path
    )

    openssl = run_openssl(
        file_path
    )

    return {

        "modality":
            "document",

        "status":
            "FORENSIC_ANALYSIS_COMPLETE",

        "file_name":
            os.path.basename(
                file_path
            ),

        "file_size":
            os.path.getsize(
                file_path
            ),

        "sha256":
            sha256,

        "ocr":
            ocr,

        "exiftool":
            exif,

        "openssl":
            openssl,

        "verdict":
            "FORENSIC_REVIEW_REQUIRED",

        "message":
            "Metadata, OCR and integrity information were collected. A dedicated trained document-forgery classifier is required for a definitive AI forgery verdict."
    }