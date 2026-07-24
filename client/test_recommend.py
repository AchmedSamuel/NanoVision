from __future__ import annotations

import argparse
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

import requests

BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


def post_image(endpoint: str, image_path: Path) -> dict[str, Any]:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"

    with image_path.open("rb") as image_file:
        response = requests.post(
            f"{BASE_URL}{endpoint}",
            files={"file": (image_path.name, image_file, mime_type)},
            timeout=120,
        )

    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run image detection, image captioning, then request a recommendation."
    )
    parser.add_argument("question", help="Question for the research assistant.")
    parser.add_argument("--image", help="Optional image path.")
    parser.add_argument("--context", default="", help="Optional experiment context.")
    args = parser.parse_args()

    payload: dict[str, Any] = {
        "user_question": args.question,
        "context": {"notes": args.context} if args.context else {},
    }

    if args.image:
        image_path = Path(args.image)
        if not image_path.is_file():
            raise SystemExit(f"Image not found: {image_path}")

        detection_result = post_image("/detect-objects", image_path)
        caption_result = post_image("/describe-sample", image_path)

        payload["detections"] = detection_result["detections"]
        payload["sample_caption"] = caption_result["caption"]

    response = requests.post(
        f"{BASE_URL}/recommend-action",
        json=payload,
        timeout=120,
    )

    response.raise_for_status()

    print("Request payload:")
    print(json.dumps(payload, indent=2))
    print("\nRecommendation response:")
    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    main()