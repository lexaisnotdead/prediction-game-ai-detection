# PredictSport

PredictSport is a local prototype game where the user watches a live sports video, predicts a scoring event, and receives points when the AI detector confirms the event.

The repository is prepared in a clone-and-run format:
- runtime model weights are already included in the repository
- the unified runtime detector is already included in the repository
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
3. The frontend sends the current video timestamp, selected event type, measured stream delay, and the active prediction session token to the backend.
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
  - a single fine-tuned YOLO detector finds balls and scoring targets for both sports
  in [backend/detector.py](/backend/detector.py)
- State / event cache: Redis via [docker-compose.yml](/docker-compose.yml)

## Repository Layout

```text
frontend/
  index.html              # UI

backend/
  main.py                        # FastAPI app, websocket endpoints, static serving
  stream.py                      # YouTube stream access and frame generator
  detector.py                    # Unified YOLO detection + event logic
  scoring.py                     # Prediction scoring logic
  game_session.py                # WebSocket session registry + pending predictions
  import_unified_dataset.py      # Import Label Studio train/val export into backend/data/unified
  strip_dataset_hash_prefixes.py # Remove hash prefixes from imported dataset files
  train_unified_model.py         # Fine-tune the unified detector on backend/data/unified
  requirements.txt               # Python dependencies
  data/unified/
    dataset.yaml                 # Main training dataset config
    classes.txt                  # Unified class list
    images/train/                # Unified training images
    images/val/                  # Unified validation images
    labels/train/                # Unified training labels
    labels/val/                  # Unified validation labels
  models/trained/
    unified_detector.pt           # Included runtime unified detector
    unified_detector.json         # Metadata for the trained unified detector
  tests/
    test_detector.py              # Detector helper regression tests
    test_game_session.py          # Session/pending prediction tests
    test_scoring.py               # Scoring and anti-cheat tests
  runs/unified_training/          # Local training runs and metrics (generated)

docker-compose.yml        # Redis
.gitignore                # Ignore local env, generated data, and training artifacts
```

## Included Weights

The repository already contains the files required to run:

- unified_detector.pt

No extra model training is required for normal use.

## Detection Pipeline

- Ball and target detection are handled by one fine-tuned YOLO model with four classes:
  - `basketball_ball`
  - `basketball_rim`
  - `football_ball`
  - `football_goal`
- A scoring event is emitted when the detected ball enters the detected target area.
- Debug frames that contain detections are saved to [backend/debug_frames](/backend/debug_frames).

## Architecture

1. The browser opens the UI from FastAPI.
2. The frontend requests match metadata from `/matches`.
3. The backend opens the configured YouTube stream through `yt-dlp`.
4. OpenCV reads frames from the direct stream URL.
5. One unified YOLO detector finds both the ball and the scoring target for the active sport.
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
- This repository is currently runtime-focused: included weights are meant to be used as-is, and the old dataset/training helper scripts are no longer part of the repo.

## Known Limitations

- Detection quality still depends on the visual quality and camera angle of the broadcast.
- Stream timestamps come from the video stream and can drift slightly from what the user perceives in the player.
- This is a prototype: scoring, anti-cheat logic, and event detection are intentionally simple.
