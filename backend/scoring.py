"""
scoring.py — Scoring engine + anti-cheat

delta_raw  = event_ts - predictedOffset
delta_norm = delta_raw - stream_delay   (нормализация задержки потока)

Anti-cheat:
  1. delta_norm ≤ 0 → клик после события (с учётом задержки) → rejected
  2. predictedOffset > event_ts → клик явно после события → rejected
  3. Максимум 1 предсказание на событие за сессию
  4. Rate limit: не чаще 1 клика в 5 секунд (серверное время)

Очки:
  0–2s   → 1000 pts (Perfect)
  2–5s   → 500  pts (Great)
  5–10s  → 100  pts (Good)
  >10s   → 0    pts (Too early)
"""

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Prediction:
    match_id:         str
    event_type:       str
    predicted_offset: float   # секунды, player.getCurrentTime() на клиенте
    stream_delay:     float   # измеренная задержка потока этого клиента
    server_recv_time: float   # time.time() на сервере при получении


@dataclass
class ScoreResult:
    pts:          int
    quality:      str     # Perfect / Great / Good / Too Early / Rejected
    delta_raw:    float
    delta_norm:   float
    type_match:   bool
    rejected:     bool
    reject_reason: Optional[str] = None


# Защита от флуда: храним время последнего клика per client
_last_click: dict[str, float] = {}
RATE_LIMIT_SECONDS = 10.0


def score(prediction: Prediction, event_ts: float, event_type: str) -> ScoreResult:
    """
    Считаем очки для одного предсказания.

    prediction.predicted_offset — позиция в видео в момент клика.
    event_ts                    — позиция в видео когда произошло событие
                                  (из CAP_PROP_POS_MSEC на сервере).
    """
    delta_raw  = event_ts - prediction.predicted_offset
    delta_norm = delta_raw - prediction.stream_delay

    # ── Anti-cheat ────────────────────────────────────────────────────
    if prediction.predicted_offset >= event_ts:
        return ScoreResult(
            pts=0, quality="Rejected", delta_raw=delta_raw,
            delta_norm=delta_norm, type_match=False,
            rejected=True, reject_reason="click after event timestamp"
        )

    if delta_norm <= 0:
        return ScoreResult(
            pts=0, quality="Rejected", delta_raw=delta_raw,
            delta_norm=delta_norm, type_match=False,
            rejected=True, reject_reason="normalised delta ≤ 0 (stream delay exploit)"
        )

    type_match = prediction.event_type == event_type

    # ── Scoring ───────────────────────────────────────────────────────
    if delta_raw <= 2:
        base, quality = 1000, "Perfect ✦"
    elif delta_raw <= 5:
        base, quality = 500,  "Great"
    elif delta_raw <= 10:
        base, quality = 100,  "Good"
    else:
        base, quality = 0,    "Too Early"

    pts = base

    return ScoreResult(
        pts=pts, quality=quality,
        delta_raw=delta_raw, delta_norm=delta_norm,
        type_match=type_match, rejected=False,
    )


def check_rate_limit(client_id: str) -> Optional[str]:
    """Возвращает None если ок, строку с причиной если заблокировано."""
    now = time.time()
    last = _last_click.get(client_id, 0)
    if now - last < RATE_LIMIT_SECONDS:
        wait = RATE_LIMIT_SECONDS - (now - last)
        return f"rate limit: wait {wait:.1f}s"
    _last_click[client_id] = now
    return None
