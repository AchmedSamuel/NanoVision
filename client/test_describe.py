from __future__ import annotations

import argparse
import json
import mimetypes
import os
from pathlib import Path

import requests

BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload an image to /describe-sample.")
    parser.add_argument("image", help="Path to a JPEG, PNG, or WEBP image.")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        raise SystemExit(f"Image not found: {image_path}")

    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"

    with image_path.open("rb") as image_file:
        response = requests.post(
            f"{BASE_URL}/describe-sample",
            files={"file": (image_path.name, image_file, mime_type)},
            timeout=120,
        )

    response.raise_for_status()
    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    main()