# PredictSport

PredictSport is a local prototype game where the user watches a live sports video, predicts a scoring event, and receives points when the AI detector confirms the event.

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

## Game Mechanics

The game loop is simple:

1. Open one of the available sports streams.
2. Click `Predict Now` before the scoring moment happens.
3. The frontend sends the current video timestamp, selected event type, measured stream delay, and client id to the backend.
4. The backend waits for the next detected event for that match.
5. When an event is detected, the prediction is scored and the result is pushed back to the browser over WebSocket.
6. If the prediction earns points, they are added to the player total and reflected in the leaderboard.

Current gameplay details:

- The supported event type is currently `goal` for both football and basketball.
- The predict button is locked for 10 seconds after a successful click on the frontend.
- The backend also enforces a 10 second server-side rate limit per client.
- The leaderboard is local to the running app session and includes a few seeded fake players plus the current user.

## Scoring

Scoring is based on how early the prediction was made relative to the detected event timestamp.

- `0-2s` before the event: `1000` points, `Perfect`
- `2-5s` before the event: `500` points, `Great`
- `5-10s` before the event: `100` points, `Good`
- `>10s` before the event: `0` points, `Too Early`


Predictions can also be rejected. The current backend rejects a prediction when:

- the click timestamp is already after the detected event timestamp
- the delay-normalized prediction timing is not positive, which protects against stream-delay abuse
- the client hits the server-side rate limit

## Stack

- Frontend: plain HTML/CSS/JS in [frontend/index.html](/frontend/index.html)
- Backend API: FastAPI in [backend/main.py](/backend/main.py)
- Video ingestion: `yt-dlp` + OpenCV in [backend/stream.py](/backend/stream.py)
- Detection:
  - balls are detected with YOLO / YOLOE
  - targets (football goal and basketball hoop/backboard) are detected with custom trained target models
  in [backend/detector.py](/backend/detector.py)
- State / event cache: Redis via [docker-compose.yml](/docker-compose.yml)

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

- yoloe-11s-seg.pt
- football_target.pt
- basketball_target.pt

No extra model training is required for normal use. Some Ultralytics open-vocabulary support files may still be fetched automatically on first run if they are not present locally.

## Detection Pipeline

- Ball detection is handled by YOLO / YOLOE.
- Target detection is handled by custom trained models included in the repository:
  - `football_target.pt` for football goals
  - `basketball_target.pt` for basketball rim / backboard targets
- A scoring event is emitted when the detected ball enters the detected target area.
- Debug frames that contain detections are saved to [backend/debug_frames](/backend/debug_frames).

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

### 5. Optional debug mode

Debug mode is disabled by default.

To run the backend with debug enabled:

```bash
cd backend
DEBUG_MODE=1 python -m uvicorn main:app --port 8000
```

When debug mode is enabled:

- frames with detected balls, targets, and goals are saved to [backend/debug_frames](/backend/debug_frames)
- debug overlay drawing is enabled
- detector debug logs are printed to the console

### 6. Open the app

Open:

```text
http://localhost:8000
```

## Runtime Notes

- The app serves the frontend directly from FastAPI.
- Only the currently active match is processed.
- Football and basketball streams are configured in [backend/stream.py](/backend/stream.py).
- Debug frame saving is controlled by `DEBUG_MODE`; when it is enabled, frames with detected objects and goals are saved to [backend/debug_frames](/backend/debug_frames).
- The repository contains training utilities, but they are optional and not needed to run the app.

## Optional Training Utilities

These files remain in the repository as developer tools:

- backend/train_targets.py
- backend/prepare_target_dataset.py
- backend/import_label_studio_yolo.py

They are not required for end users.

## Known Limitations

- Detection quality still depends on the visual quality and camera angle of the broadcast.
- Stream timestamps come from the video stream and can drift slightly from what the user perceives in the player.
- This is a prototype: scoring, anti-cheat logic, and event detection are intentionally simple.
