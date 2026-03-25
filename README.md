# PredictSport

PredictSport is a local prototype game where the user watches a live sports video, predicts a scoring event, and receives points when the backend detector confirms the event.

The repository is prepared in a clone-and-run format:
- runtime model weights are already included in the repository
- trained target models are already included in the repository
- the user does not need to train models or download weights to start the app

## What The App Does

- shows a sports stream in the browser
- lets the player submit a prediction for the next scoring event
- detects events on the backend from the video stream
- scores the prediction and updates the UI leaderboard

Supported modes right now:
- `football`
- `basketball`

## Stack

- Frontend: plain HTML/CSS/JS in [frontend/index.html](/Users/alekseidiakonov/Documents/Projects/ai-detection-prediction-game/frontend/index.html)
- Backend API: FastAPI in [backend/main.py](/Users/alekseidiakonov/Documents/Projects/ai-detection-prediction-game/backend/main.py)
- Video ingestion: `yt-dlp` + OpenCV in [backend/stream.py](/Users/alekseidiakonov/Documents/Projects/ai-detection-prediction-game/backend/stream.py)
- Detection:
  - balls are detected with YOLO / YOLOE
  - targets (football goal and basketball hoop/backboard) are detected with custom trained target models
  in [backend/detector.py](/Users/alekseidiakonov/Documents/Projects/ai-detection-prediction-game/backend/detector.py)
- State / event cache: Redis via [docker-compose.yml](/Users/alekseidiakonov/Documents/Projects/ai-detection-prediction-game/docker-compose.yml)

## Repository Layout

```text
frontend/
  index.html              # UI

backend/
  main.py                 # FastAPI app, websocket endpoints, static serving
  stream.py               # YouTube stream access and frame generator
  detector.py             # YOLO ball detection + custom target-model detection + event logic
  scoring.py              # Prediction scoring logic
  requirements.txt        # Python dependencies
  debug_frames/           # Saved debug frames with detections
  models/trained/
    football_target.pt    # Included runtime target model
    basketball_target.pt  # Included runtime target model

yoloe-11s-seg.pt          # Included runtime base detector
docker-compose.yml        # Redis
```

## Included Weights

The repository already contains the files required to run:

- [yoloe-11s-seg.pt](/Users/alekseidiakonov/Documents/Projects/ai-detection-prediction-game/yoloe-11s-seg.pt)
- [football_target.pt](/Users/alekseidiakonov/Documents/Projects/ai-detection-prediction-game/backend/models/trained/football_target.pt)
- [basketball_target.pt](/Users/alekseidiakonov/Documents/Projects/ai-detection-prediction-game/backend/models/trained/basketball_target.pt)

No extra model training is required for normal use. Some Ultralytics open-vocabulary support files may still be fetched automatically on first run if they are not present locally.

## Detection Pipeline

- Ball detection is handled by YOLO / YOLOE.
- Target detection is handled by custom trained models included in the repository:
  - `football_target.pt` for football goals
  - `basketball_target.pt` for basketball rim / backboard targets
- A scoring event is emitted when the detected ball enters the detected target area.
- Debug frames that contain detections are saved to [backend/debug_frames](/Users/alekseidiakonov/Documents/Projects/ai-detection-prediction-game/backend/debug_frames).

## Architecture

1. The browser opens the UI from FastAPI.
2. The frontend requests match metadata from `/matches`.
3. The backend opens the configured YouTube stream through `yt-dlp`.
4. OpenCV reads frames from the direct stream URL.
5. YOLO / YOLOE detects the ball, while the football and basketball targets are detected by the included custom trained target models.
6. The detector emits a `goal` event when the ball enters the target area.
7. Events and scoring results are pushed to the browser over WebSocket.
8. Redis is used for lightweight shared state and event delivery.

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/lexaisnotdead/prediction-game-ai-detection
cd prediction-game-ai-detection
```

### 2. Start Redis

```bash
docker-compose up -d
```

### 3. Create a Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

`backend/requirements.txt` already includes the dependency required by the Ultralytics open-vocabulary stack, so an extra `CLIP` auto-install on first run should usually not be needed.

### 4. Run the backend

```bash
cd backend
python -m uvicorn main:app --port 8000
```

### 5. Open the app

Open:

```text
http://localhost:8000
```

## Runtime Notes

- The app serves the frontend directly from FastAPI.
- Only the currently active match is processed.
- Football and basketball streams are configured in [backend/stream.py](/Users/alekseidiakonov/Documents/Projects/ai-detection-prediction-game/backend/stream.py).
- Debug frames that contain detections are saved to [backend/debug_frames](/Users/alekseidiakonov/Documents/Projects/ai-detection-prediction-game/backend/debug_frames).
- The repository contains training utilities, but they are optional and not needed to run the app.

## Optional Training Utilities

These files remain in the repository as developer tools:

- [backend/train_targets.py](/Users/alekseidiakonov/Documents/Projects/ai-detection-prediction-game/backend/train_targets.py)
- [backend/prepare_target_dataset.py](/Users/alekseidiakonov/Documents/Projects/ai-detection-prediction-game/backend/prepare_target_dataset.py)
- [backend/import_label_studio_yolo.py](/Users/alekseidiakonov/Documents/Projects/ai-detection-prediction-game/backend/import_label_studio_yolo.py)

They are not required for end users.

## Known Limitations

- Detection quality still depends on the visual quality and camera angle of the broadcast.
- Stream timestamps come from the video stream and can drift slightly from what the user perceives in the player.
- This is a prototype: scoring, anti-cheat logic, and event detection are intentionally simple.
