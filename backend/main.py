"""
main.py — FastAPI + WebSocket

Endpoints:
  GET  /matches                    — список матчей и их доступность
  POST /predict                    — принять предсказание
  WS   /ws/{match_id}/{client_id}  — получать события в реальном времени

Фоновая задача на каждый матч:
  stream.frame_generator → detector.process → при событии:
    1. Пишем event_ts в Redis
    2. Бродкастим {event_type, event_ts} всем WS-клиентам этого матча
    3. Скорим все ожидающие предсказания

Redis ключи:
  match:{id}:last_event_ts    — float, timestamp последнего события
  match:{id}:last_event_type  — str
"""

import asyncio
import math
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import redis.asyncio as aioredis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.responses import FileResponse
from pydantic import BaseModel

from stream import MATCHES, is_stream_available, frame_generator
from detector import make_detector
from game_session import ConnectionRegistry, PendingPredictions
from scoring import Prediction, score, check_rate_limit, clear_rate_limit, clamp_stream_delay


# ── Redis ──────────────────────────────────────────────────────────────
redis_client: Optional[aioredis.Redis] = None

async def get_redis() -> aioredis.Redis:
    global redis_client
    if redis_client is None:
        redis_client = aioredis.from_url("redis://localhost:6379", decode_responses=True)
    return redis_client


manager = ConnectionRegistry()
pending = PendingPredictions()


# ── Detector background loop ───────────────────────────────────────────
available_matches: dict[str, bool] = {}
detector_tasks: dict[str, asyncio.Task] = {}
detector_stop_events: dict[str, threading.Event] = {}


async def ensure_detector(match_id: str) -> bool:
    if not available_matches.get(match_id, False):
        return False
    task = detector_tasks.get(match_id)
    if task and not task.done():
        return True
    stop_event = threading.Event()
    detector_stop_events[match_id] = stop_event
    detector_tasks[match_id] = asyncio.create_task(run_detector(match_id, stop_event))
    print(f"[runtime] {match_id} — detector started on demand")
    return True


async def stop_detector_if_unused(match_id: str) -> None:
    if manager.count(match_id) > 0:
        return
    task = detector_tasks.get(match_id)
    if not task or task.done():
        return
    stop_event = detector_stop_events.get(match_id)
    if stop_event is not None:
        stop_event.set()
    try:
        await asyncio.wait_for(task, timeout=5)
    except asyncio.TimeoutError:
        print(f"[runtime] {match_id} — detector stop timeout, waiting for stream loop to exit")


async def run_detector(match_id: str, stop_event: threading.Event):
    """
    Запускаем в asyncio executor чтобы не блокировать event loop.
    YOLO и cv2 — синхронные, поэтому thread pool.
    """
    sport    = MATCHES[match_id]["sport"]
    r        = await get_redis()

    print(f"[{match_id}] detector started, sport={sport}")

    loop = asyncio.get_event_loop()

    # Запускаем blocking loop в thread executor
    # Используем queue для передачи событий в async контекст
    event_queue: asyncio.Queue = asyncio.Queue()

    def blocking_generator():
        sport_det = make_detector(sport)
        try:
            for frame, ts in frame_generator(match_id, stop_event=stop_event):
                ev = sport_det.process(frame, ts)
                if ev:
                    # Кладём в queue через thread-safe вызов
                    loop.call_soon_threadsafe(event_queue.put_nowait, ev)
        except Exception as e:
            print(f"[{match_id}] generator error: {e}")
        finally:
            loop.call_soon_threadsafe(event_queue.put_nowait, None)  # sentinel

    # Запускаем blocking часть в executor
    executor_task = loop.run_in_executor(None, blocking_generator)

    # Читаем события из очереди
    while True:
        ev = await event_queue.get()
        if ev is None:
            break

        print(f"[{match_id}] EVENT: {ev.event_type} at {ev.timestamp:.2f}s (conf={ev.confidence:.2f})")

        # Пишем в Redis одним пайплайном, чтобы не делать лишние round-trips.
        async with r.pipeline() as pipe:
            await pipe.set(f"match:{match_id}:last_event_ts", str(ev.timestamp))
            await pipe.set(f"match:{match_id}:last_event_type", ev.event_type)
            await pipe.expire(f"match:{match_id}:last_event_ts", 60)
            await pipe.expire(f"match:{match_id}:last_event_type", 60)
            await pipe.execute()

        # Бродкастим клиентам
        await manager.broadcast(match_id, {
            "type":       "event",
            "event_type": ev.event_type,
            "event_ts":   round(ev.timestamp, 2),
        })

        # Скорим ожидающие предсказания
        preds = pending.pop_match(match_id)
        event_received_time = time.time()
        for pred in preds:
            clear_rate_limit(pred.client_id)
            result = score(pred, ev.timestamp, ev.event_type, event_received_time=event_received_time)
            await manager.send_to_client(match_id, pred.client_id, {
                "type":        "score_result",
                "client_id":   pred.client_id,
                "pts":         result.pts,
                "quality":     result.quality,
                "delta_raw":   round(result.delta_raw,  3),
                "delta_norm":  round(result.delta_norm, 3),
                "delta_server": round(result.delta_server, 3),
                "type_match":  result.type_match,
                "rejected":    result.rejected,
                "reject_reason": result.reject_reason,
            })

    await executor_task
    detector_tasks.pop(match_id, None)
    detector_stop_events.pop(match_id, None)
    print(f"[{match_id}] detector finished")


# ── App lifecycle ──────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    availability = await asyncio.gather(
        *(asyncio.to_thread(is_stream_available, match_id) for match_id in MATCHES)
    )
    for match_id, available in zip(MATCHES, availability):
        available_matches[match_id] = available
        print(f"[startup] {match_id} — {'available' if available else 'unavailable'}")
    yield
    for task in detector_tasks.values():
        task.cancel()


app = FastAPI(title="PredictSport API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_INDEX = ROOT_DIR / "frontend" / "index.html"


# ── REST endpoints ─────────────────────────────────────────────────────

@app.get("/")
async def serve_frontend():
    return FileResponse(FRONTEND_INDEX)

@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

@app.get("/matches")
async def get_matches():
    """Возвращает список матчей с флагом доступности."""
    result = {}
    for mid, info in MATCHES.items():
        result[mid] = {
            **info,
            "available": available_matches.get(mid, False),
            "active": mid in detector_tasks and not detector_tasks[mid].done(),
        }
    return result


class PredictRequest(BaseModel):
    match_id:         str
    event_type:       str   # "goal"
    predicted_offset: float # player.getCurrentTime()
    stream_delay:     float # измеренная задержка на клиенте (секунды)
    prediction_token: str


@app.post("/predict")
async def predict(req: PredictRequest):
    if req.match_id not in MATCHES:
        raise HTTPException(status_code=404, detail="match not found")

    if not math.isfinite(req.predicted_offset) or req.predicted_offset < 0:
        raise HTTPException(status_code=422, detail="predicted_offset must be a finite positive number")

    if not math.isfinite(req.stream_delay):
        raise HTTPException(status_code=422, detail="stream_delay must be finite")

    session = manager.get_session(req.prediction_token)
    if session is None or session.match_id != req.match_id:
        raise HTTPException(status_code=403, detail="prediction session is invalid or expired")

    # Rate limit
    err = check_rate_limit(session.client_id)
    if err:
        raise HTTPException(status_code=429, detail=err)

    if not await ensure_detector(req.match_id):
        raise HTTPException(status_code=503, detail="stream not available")

    pred = Prediction(
        client_id        = session.client_id,
        match_id         = req.match_id,
        event_type       = req.event_type,
        predicted_offset = req.predicted_offset,
        stream_delay     = clamp_stream_delay(req.stream_delay),
        server_recv_time = time.time(),
    )

    pending.add(req.match_id, session.client_id, pred)
    return {"status": "ok", "message": "prediction received"}


# ── WebSocket ──────────────────────────────────────────────────────────

@app.websocket("/ws/{match_id}/{client_id}")
async def websocket_endpoint(ws: WebSocket, match_id: str, client_id: str):
    if match_id not in MATCHES or not available_matches.get(match_id, False):
        await ws.close(code=1013)
        return

    prediction_token = await manager.connect(match_id, client_id, ws)
    try:
        await ensure_detector(match_id)
        # Отправляем текущее состояние матча сразу после подключения
        r = await get_redis()
        last_ts   = await r.get(f"match:{match_id}:last_event_ts")
        last_type = await r.get(f"match:{match_id}:last_event_type")
        await ws.send_json({
            "type":       "connected",
            "match_id":   match_id,
            "prediction_token": prediction_token,
            "last_event_ts":   float(last_ts)   if last_ts   else None,
            "last_event_type": last_type if last_type else None,
        })

        # Держим соединение живым — клиент шлёт ping
        while True:
            data = await ws.receive_text()
            if data == "ping":
                await ws.send_text("pong")

    except WebSocketDisconnect:
        manager.disconnect(match_id, ws)
        await stop_detector_if_unused(match_id)
