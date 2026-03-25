"""
stream.py — yt-dlp + OpenCV frame generator

Получает прямой URL потока через yt-dlp,
затем читает кадры через OpenCV с нужным fps.

ВАЖНО: для live-стримов YouTube getCurrentTime() на клиенте
и timestamp кадра на сервере оба отсчитываются от начала
видео/стрима — они на одной шкале времени.
"""

import time
import threading
import yt_dlp
import cv2
import numpy as np
from typing import Generator, Optional, Tuple


# Захардкоженные матчи
MATCHES = {
    "football": {
        "url":   "https://youtu.be/hzR3qPMTQzQ",
        "label": "Football · Premier League",
        "sport": "football",
        "start_offset": 300.0,
    },
    "basketball": {
        "url":   "https://youtu.be/LPDnemFoqVk",
        "label": "Basketball · NBA Highlights",
        "sport": "basketball",
        "start_offset": 20.0,
    },
}

# 6 fps — разумный баланс: мяч в броске пролетает зону за ~0.1-0.2s,
# при 6fps это 1-2 кадра покрытия. При старом YOLO_IMGSZ=768 + апскейле
# мы не успевали даже 4fps; теперь с imgsz=480 без апскейла — успеваем.
TARGET_FPS = 6


def get_stream_url(youtube_url: str) -> str:
    """Получаем прямой URL медиапотока через yt-dlp."""
    ydl_opts = {
        # Забираем metadata всех форматов и сами выбираем лучший одиночный video URL для OpenCV.
        "format": "best",
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        # Для live-стримов info может быть playlist с одним entry
        if "entries" in info:
            info = info["entries"][0]

        formats = info.get("formats") or []
        candidates = []
        for fmt in formats:
            url = fmt.get("url")
            if not url:
                continue
            if fmt.get("vcodec") == "none":
                continue
            protocol = (fmt.get("protocol") or "").lower()
            ext = (fmt.get("ext") or "").lower()
            height = int(fmt.get("height") or 0)
            width = int(fmt.get("width") or 0)
            fps = float(fmt.get("fps") or 0)

            # OpenCV обычно лучше всего дружит с mp4/https и HLS-потоками.
            protocol_score = 2 if protocol in {"https", "http", "m3u8_native", "m3u8"} else 0
            ext_score = 1 if ext in {"mp4", "ts"} else 0
            candidates.append(((height, width, fps, protocol_score, ext_score), fmt))

        if candidates:
            candidates.sort(key=lambda item: item[0], reverse=True)
            best = candidates[0][1]
            print(
                "[stream] selected format:",
                f"id={best.get('format_id')}",
                f"ext={best.get('ext')}",
                f"protocol={best.get('protocol')}",
                f"size={best.get('width')}x{best.get('height')}",
                f"fps={best.get('fps')}",
            )
            return best["url"]

        if "url" in info:
            return info["url"]

        raise KeyError("url")


def is_stream_available(match_id: str) -> bool:
    """Проверяем доступность видео/стрима перед запуском."""
    url = MATCHES[match_id]["url"]
    ydl_opts = {"quiet": True, "no_warnings": True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if "entries" in info:
                info = info["entries"][0]
            # live_status: "is_live", "was_live", "not_live", None
            live_status = info.get("live_status")
            is_live = live_status in ("is_live", "was_live", "not_live", None)
            return is_live
    except Exception:
        return False


def frame_generator(
    match_id: str,
    target_fps: int = TARGET_FPS,
    stop_event: Optional[threading.Event] = None,
) -> Generator[Tuple[np.ndarray, float], None, None]:
    """
    Генератор кадров: yield (frame_bgr, timestamp_seconds)

    timestamp — позиция в видео в секундах, совпадает с тем,
    что YouTube IFrame player.getCurrentTime() вернёт клиенту.
    """
    url = MATCHES[match_id]["url"]
    start_offset = float(MATCHES[match_id].get("start_offset", 0.0))
    stream_url = get_stream_url(url)

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open stream for match {match_id}")

    if start_offset > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_offset * 1000.0)

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    skip = max(1, int(native_fps / target_fps))

    frame_idx = 0
    while True:
        if stop_event is not None and stop_event.is_set():
            break
        ret, frame = cap.read()
        if not ret:
            break

        # Обрабатываем только каждый N-й кадр
        if frame_idx % skip == 0:
            # Timestamp в секундах от начала видео
            ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if ts < start_offset:
                frame_idx += 1
                continue
            yield frame, ts

        frame_idx += 1

    cap.release()
