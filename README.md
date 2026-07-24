# NanoVision

**NanoVision** is a robotics-ready RESTful AI vision service for materials-science laboratories. It combines **Gemini Vision** as a remote cloud AI service with **local YOLO11n object detection** to help a laboratory robot, automation system, or researcher describe sample images, detect visible objects, and generate cautious research-action recommendations.


## 1. Why this project exists

In nanomaterials research, many early experimental decisions are still made by visual inspection. Researchers often look at vials, suspensions, precipitates, liquid layers, phase separation, turbidity, colour changes, or workspace context and decide what to check next.

This manual process is:
- subjective.
- difficult to reproduce.
- hard to integrate into robotic workflows.
- not directly machine-readable.
- slow for closed-loop or high-throughput experimentation.

NanoVision addresses this problem by exposing a **cloud–local AI service** through FastAPI. The service can be called by a Python client, curl, Swagger UI, or later by a robotic/laboratory automation controller.

The guiding idea is:

```text
lab/sample image + context
        ↓
REST API
        ↓
Gemini Vision + local YOLO
        ↓
structured observations + recommendation
        ↓
robotics-ready JSON response
```


## 2. Research problem solved

**Problem:** Laboratory image interpretation is usually performed manually and is not directly available to robotic automation systems.

**NanoVision solution:** Convert laboratory/sample images and user context into structured API responses that can be consumed by downstream automation, monitoring, and decision-support workflows.

The current version focuses on:

- image captioning for sample/lab scenes.
- object detection using a local model.
- short AI-assisted recommendations for materials-science workflows.
- reproducible HTTP endpoints suitable for robotic service integration.


## 3. Scientific and robotics relevance

This project is relevant to materials science and robotics because it treats laboratory vision as a REST service. It aligned with the idea that a robot or automated synthesis platform should not only execute commands, but also perceive, interpret, and report the state of the laboratory environment.

Potential use cases include:
- preliminary monitoring of synthesis samples.
- visual logging of vial/sample states.
- robotic inspection of workspaces.
- image-driven experiment annotation.
- future closed-loop experimentation pipelines.

In its current form, NanoVision is intentionally simple and reproducible. It is designed to run on one laptop.



## 4. Component responsibilities

| Component | Role |
|---|---|
| FastAPI | Provides REST endpoints and OpenAPI documentation |
| Uvicorn | Runs the ASGI server |
| Gemini | Remote AI service for image captioning and recommendation generation |
| YOLO11n | Local computer-vision component for object detection |
| Pillow | Loads and validates uploaded images |
| Pydantic | Defines and validates JSON request/response schemas |
| python-dotenv | Loads environment variables from `.env` |
| requests | Used by client scripts to call the API |


## 5. Repository structure

```text
NanoVision/
├── client/
│   ├── test_describe.py
│   ├── test_detect.py
│   └── test_recommend.py
├── docs/
│   └── screenshots/
├── server/
│   └── main.py
├── .env.samples
├── .gitignore
├── README.md
├── image.jpg
└── requirements.txt
```

| Path | Purpose |
|---|---|
| `server/main.py` | Main FastAPI application, endpoints, validation, Gemini calls, YOLO inference, timing |
| `client/test_describe.py` | Client script for `/describe-sample` |
| `client/test_detect.py` | Client script for `/detect-objects` |
| `client/test_recommend.py` | Client script for the full vision-to-decision pipeline |
| `.env.samples` | Public template for environment variables |
| `.gitignore` | Prevents committing secrets, virtual environments, model files, and generated outputs |
| `requirements.txt` | Pinned Python dependencies |
| `README.md` | Project documentation |
| `image.jpg` | Optional sample image used for demonstration |



## 6. API endpoints

| Method | Endpoint | Input | Output | AI component |
|---|---|---|---|---|
| `GET` | `/health` | None | API status, uptime, endpoints, components | FastAPI |
| `POST` | `/describe-sample` | Multipart image file | Caption, model name, processing time | Gemini Vision |
| `POST` | `/detect-objects` | Multipart image file | Detection list with class, confidence, bounding box | Local YOLO11n |
| `POST` | `/recommend-action` | JSON question, optional caption, detections, context | Recommendation, model name, detection count, processing time | Gemini |


## 7. Data flow

### 7.1 `/describe-sample`

```text
image.jpg
  → FastAPI upload
  → MIME/size validation
  → Pillow image loading
  → Gemini Vision
  → JSON caption response
```

### 7.2 `/detect-objects`

```text
image.jpg
  → FastAPI upload
  → MIME/size validation
  → Pillow image loading
  → YOLO11n local inference
  → JSON detections response
```

### 7.3 `/recommend-action`

```text
user question + optional caption + optional detections + context
  → FastAPI JSON request
  → Gemini recommendation prompt
  → short cautious recommendation
  → JSON response
```

### 7.4 Client pipeline

`client/test_recommend.py` demonstrates the full pipeline:

```text
image.jpg
  → /detect-objects
  → /describe-sample
  → /recommend-action
  → final recommendation
```

## 8. Installation

### 8.1 Clone the repository

### 8.2 Create the virtual environment

### 8.3 Install dependencies

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
The first YOLO call will download `yolo11n.pt` automatically.



## 9. Run the server

```bash
cd ~/projects/NanoVision
source .NanoVision_Venv/bin/activate
uvicorn server.main:app --reload --port 8000
```

Expected terminal output:

```text
Uvicorn running on http://127.0.0.1:8000
Application startup complete.
```

Open:

```text
http://127.0.0.1:8000/docs
```

Expected endpoints:

```text
GET  /health
POST /describe-sample
POST /detect-objects
POST /recommend-action
```

## 10. Test the API with curl

Run these commands in a second WSL/Ubuntu terminal while the server is running.

### 10.1 Health check

```bash
curl "http://127.0.0.1:8000/health"
```

### 10.2 Detect objects locally

```bash
curl -X POST "http://127.0.0.1:8000/detect-objects"   -F "file=@image.jpg"
```


An empty detection list is not an error. It means YOLO ran correctly but did not detect a COCO-class object in that specific image. YOLO11n is a general detector and is not trained to identify perovskite phases, precipitation, turbidity, or chemical species.

### 10.3 Describe a sample image

```bash
curl -X POST "http://127.0.0.1:8000/describe-sample"   -F "file=@image.jpg"
```


### 10.4 Request a recommendation

```bash
curl -X POST "http://127.0.0.1:8000/recommend-action"   -H "Content-Type: application/json"   -d '{
    "user_question": "What should I check before continuing this experiment?",
    "sample_caption": "Five black-capped glass vials labeled from 0 mM to 4 mM are lined up against a blue background, each containing a two-phase mixture with a clear upper layer and an opaque yellowish-white lower layer.",
    "detections": [],
    "context": {
      "notes": "Perovskite precursor preparation with concentration series from 0 mM to 4 mM."
    }
  }'
```

## 11. Test with Python clients

Run these commands while the server is active.

### 11.1 Object detection client

```bash
python client/test_detect.py image.jpg
```

### 11.2 Image caption client

```bash
python client/test_describe.py image.jpg
```

### 11.3 Full vision-to-decision client

```bash
python client/test_recommend.py   "What should I check before continuing this experiment?"   --image image.jpg   --context "Perovskite precursor preparation with concentration series from 0 mM to 4 mM."
```
## 12. Limitations
NanoVision is a prototype. YOLO11n is trained on general COCO object classes. Hence it cannot identify crystal phases, precipitates, or degradation mechanisms.
Gemini can describe visible image features and provide general recommendations, but the output must be checked by the researcher. For safety-critical workflows, users must follow laboratory SOPs, SDS documents, and supervisor-approved procedures.

## 13. Future improvements
i. Train a custom YOLO model for lab-specific objects.
ii. Add turbidity, phase-separation, or precipitate classification.
iii. Integrate Ollama for local text reasoning.
iv. Add voice input/output.
v. Add a database for experiment history.
vi. Add latency logging to a CSV file.

## 14. License and academic note
This repository is submitted as a final course project for **Cloud Architectures & RESTful Services for Robotics**. The implementation is intended for academic demonstration and research prototyping.

