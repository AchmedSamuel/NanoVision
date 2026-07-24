# NanoVision

NanoVision is a robotics-ready FastAPI vision service for materials-science laboratories, combining Gemini Vision and local YOLO detection to help lab automation systems analyze sample images, detect objects, and recommend research actions.

## Why this exists

Materials-science and nanomaterials experiments often involve visual observations of samples, vials, workspaces, and intermediate experimental states. In a robotics or laboratory automation context, a service should not only respond to text; it should also see the laboratory scene, detect relevant objects, and support decision-making.

NanoVision addresses this by exposing a RESTful API that can:

- describe laboratory/sample images using Gemini Vision or any other suitable AI.
- detect visible objects locally using YOLO11n.
- combine image descriptions, detections, and user questions to generate research or laboratory recommendation.

This project is designed as a research-assistance and robotics-service prototype. It does not replace laboratory SOPs, SDS documents, instrument manuals, or expert supervision.

## NanoVision Targets:
-- FastAPI service: implemented in `server/main.py.
-- Has 3 endpoints: `/health`, `/describe-sample`, `/detect-objects`, `/recommend-action.
-- JSON input/output: All endpoints return JSON file and recommend-action` accepts JSON file.
-- Remote AI service: Gemini is used for image captioning and recommendations.
-- Local component: YOLO11n is used locally for object detection.

## NanoVision structure
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
└── requirements.txt

## ENDPOINTS

| Method | Endpoint            | Input                                                | Output                                             | AI component  |
| ------ | ------------------- | ---------------------------------------------------- | -------------------------------------------------- | ------------- |
| `GET`  | `/health`           | None                                                 | Service status, endpoints, components, uptime      | FastAPI       |
| `POST` | `/describe-sample`  | Image file                                           | JSON caption and processing time                   | Gemini Vision |
| `POST` | `/detect-objects`   | Image file                                           | JSON object detections, confidence, bounding boxes | Local YOLO11n |
| `POST` | `/recommend-action` | JSON question, optional caption, detections, context | JSON recommendation                                | Gemini        |


## SETUP
WSL/Ubuntu used in windows laptop
-- After cloning, create and activate a virtual environment

--Install Dependecies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt


## Environment variables
Create a private .env file from the sample file:
         cp .env.samples .env
         nano .env

### Edit file:
GEMINI_API_KEY=your_real_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash
YOLO_MODEL=yolo11n.pt
MAX_IMAGE_MB=8
API_BASE_URL=http://xxx.0.0.1:8000
### Save and exit:
        Ctrl + O
        Enter
        Ctrl + X

## Run the server
    source .NanoVision_Venv/bin/activate
    uvicorn server.main:app --reload --port 8000
    copy and paste http://xxx.0.0.1:8000 in your browser. add /docs

## Test with curl
        Use a test image named image.jpg in the project folder.
        i. do health check: curl "http://127.0.0.1:8000/health"
        ii. Describe a sample image: Describe a sample image
        iii.Test to detect objects locally: curl -X POST "http://127.0.0.1:8000/detect-objects" \
                                                   -F "file=@image.jpg"  
    An empty detection will be seen. It is not an error. Note: It happens because YOLO11n detects general COCO classes such as people, bottles, cups, chairs, and laptops. It is not trained to detect perovskite phases or chemical identities. I am working it to make the detection possible for this project.

## Test with client scripts
        i. detect objects: python client/test_detect.py image.jpg
        ii. Describe image: python client/test_describe.py image.jpg
        iii. Full vision to decision pipeline:
        python client/test_recommend.py \
        "What should I check before continuing this experiment?" \
        --image image.jpg \
        --context "Perovskite precursor preparation with concentration series from 0 mM to 4 mM."

## Limitations
NanoVision is a prototype. YOLO11n is trained on general COCO object classes. Hence it cannot identify crystal phases, precipitates, or degradation mechanisms.
Gemini can describe visible image features and provide general recommendations, but the output must be checked by the researcher. For safety-critical workflows, users must follow laboratory SOPs, SDS documents, and supervisor-approved procedures.

## Future improvements
i. Train a custom YOLO model for lab-specific objects.
ii. Add turbidity, phase-separation, or precipitate classification.
iii. Integrate Ollama for local text reasoning.
iv. Add voice input/output.
v. Add a database for experiment history.
vi. Add latency logging to a CSV file.