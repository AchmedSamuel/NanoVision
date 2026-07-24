from __future__ import annotations

import io
import json
import os
import time
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from google import genai
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field
from ultralytics import YOLO

load_dotenv()

APP_TITLE = "NanoVision"
APP_VERSION = "1.0.0"

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolo11n.pt")
MAX_IMAGE_MB = float(os.getenv("MAX_IMAGE_MB", "8"))
MAX_IMAGE_BYTES = int(MAX_IMAGE_MB * 1024 * 1024)

START_TIME = time.perf_counter()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

gemini_client = (
    genai.Client(api_key=GEMINI_API_KEY)
    if GEMINI_API_KEY
    else None
)

# Load YOLO once at startup.
yolo_model = YOLO(YOLO_MODEL)

app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=(
        "A robotics-ready FastAPI vision service for materials-science "
        "laboratories using Gemini Vision and local YOLO detection."
    ),
)


class Detection(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    class_name: str = Field(alias="class")
    confidence: float
    box: list[float]


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    uptime_seconds: float
    endpoints: list[str]
    components: dict[str, Any]


class CaptionResponse(BaseModel):
    caption: str
    model: str
    processing_ms: float


class DetectionResponse(BaseModel):
    detections: list[Detection]
    model: str
    image_width: int
    image_height: int
    processing_ms: float


class RecommendationRequest(BaseModel):
    user_question: str = Field(min_length=3, max_length=2000)
    sample_caption: str | None = Field(default=None, max_length=2000)
    detections: list[Detection] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)


class RecommendationResponse(BaseModel):
    recommendation: str
    model: str
    used_detection_count: int
    processing_ms: float


@app.middleware("http")
async def add_process_time_header(request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    return response


def require_gemini_client():
    if gemini_client is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Gemini is not configured. Add GEMINI_API_KEY to .env "
                "and restart the server."
            ),
        )
    return gemini_client


async def load_uploaded_image(file: UploadFile) -> Image.Image:
    allowed_types = {"image/jpeg", "image/png", "image/webp"}

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail="Supported image types are JPEG, PNG, and WEBP.",
        )

    data = await file.read()

    if not data:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")

    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds the {MAX_IMAGE_MB:g} MB limit.",
        )

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid readable image.",
        ) from exc


@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "ok",
        "service": APP_TITLE,
        "version": APP_VERSION,
        "uptime_seconds": round(time.perf_counter() - START_TIME, 2),
        "endpoints": [
            "GET /health",
            "POST /describe-sample",
            "POST /detect-objects",
            "POST /recommend-action",
        ],
        "components": {
            "remote_ai": {
                "provider": "Gemini",
                "model": GEMINI_MODEL,
                "configured": gemini_client is not None,
            },
            "local_vision": {
                "provider": "Ultralytics YOLO",
                "model": YOLO_MODEL,
                "configured": True,
            },
        },
    }


@app.post("/describe-sample", response_model=CaptionResponse)
async def describe_sample(file: UploadFile = File(...)):
    start = time.perf_counter()
    image = await load_uploaded_image(file)
    client = require_gemini_client()

    prompt = (
        "You are assisting a materials-science researcher. "
        "Describe only the visible features of this laboratory or sample image "
        "in one concise sentence. Do not infer chemical composition, identity, "
        "or material properties that are not visually supported."
    )

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[prompt, image],
        )
        caption = (response.text or "").strip()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Gemini vision request failed: {exc}",
        ) from exc

    if not caption:
        raise HTTPException(
            status_code=502,
            detail="Gemini returned an empty caption.",
        )

    return {
        "caption": caption,
        "model": GEMINI_MODEL,
        "processing_ms": round((time.perf_counter() - start) * 1000, 2),
    }


@app.post("/detect-objects", response_model=DetectionResponse)
async def detect_objects(file: UploadFile = File(...)):
    start = time.perf_counter()
    image = await load_uploaded_image(file)

    try:
        result = yolo_model(image, verbose=False)[0]
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Local YOLO detection failed: {exc}",
        ) from exc

    detections: list[dict[str, Any]] = []

    for box in result.boxes:
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())
        coordinates = [
            round(float(value), 2)
            for value in box.xyxy[0].tolist()
        ]

        detections.append(
            {
                "class": str(yolo_model.names[class_id]),
                "confidence": round(confidence, 4),
                "box": coordinates,
            }
        )

    return {
        "detections": detections,
        "model": YOLO_MODEL,
        "image_width": image.width,
        "image_height": image.height,
        "processing_ms": round((time.perf_counter() - start) * 1000, 2),
    }


@app.post("/recommend-action", response_model=RecommendationResponse)
def recommend_action(request: RecommendationRequest):
    start = time.perf_counter()
    client = require_gemini_client()

    supplied_data = request.model_dump(by_alias=True)

    prompt = (
        "You are a cautious AI assistant for robotics-enabled materials-science "
        "and chemistry laboratory workflows. Provide one short, practical "
        "recommendation based only on the supplied question, caption, detections, "
        "and context. Do not invent observations. If the evidence is insufficient, "
        "state what additional image, measurement, or metadata is needed. "
        "For hazard-related decisions, advise the researcher to follow the "
        "relevant SOP and SDS.\n\n"
        f"Supplied data:\n{json.dumps(supplied_data, indent=2)}"
    )

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        recommendation = (response.text or "").strip()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Gemini recommendation request failed: {exc}",
        ) from exc

    if not recommendation:
        raise HTTPException(
            status_code=502,
            detail="Gemini returned an empty recommendation.",
        )

    return {
        "recommendation": recommendation,
        "model": GEMINI_MODEL,
        "used_detection_count": len(request.detections),
        "processing_ms": round((time.perf_counter() - start) * 1000, 2),
    }