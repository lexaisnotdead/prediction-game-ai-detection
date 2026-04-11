"""
detector.py — unified-model rule-based event detection

Футбол/Баскетбол:
  одна fine-tuned YOLO-модель ищет и мяч, и цель.
  Событие "goal" фиксируется, когда центр мяча попадает в bbox цели.
  Для отладки на кадры рисуется overlay с bbox мяча и ворот/кольца.
"""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional

from ultralytics import YOLO
import cv2
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent
DEBUG_FRAMES_DIR = ROOT_DIR / "debug_frames"
TRAINED_MODELS_DIR = ROOT_DIR / "models" / "trained"
UNIFIED_MODEL_PATH = TRAINED_MODELS_DIR / "unified_detector.pt"

BALL_LABELS = {
    "football": {"football_ball"},
    "basketball": {"basketball_ball"},
}
TARGET_LABELS = {
    "football": {"football_goal"},
    "basketball": {"basketball_rim"},
}

# Минимальные confidence по типам объектов
BALL_CONFIDENCE = {
    "football": 0.10,
    "basketball": 0.18,
}
TARGET_CONFIDENCE = {
    "football": 0.10,
    "basketball": 0.10,
}

# Насколько расширяем bbox цели, чтобы учесть шум детекции.
TARGET_PADDING = {
    "football": 0.10,
    "basketball": 0.22,
}

TARGET_STICKY_FRAMES = {
    "football": 2,
    "basketball": 2,
}

TARGET_SCAN_INTERVAL = {
    "football": 6,
    # Кольцо не движется — сканируем реже, кэш держит позицию между сканами.
    "basketball": 10,
}

TARGET_TILE_LAYOUTS = {
    "football": [
        (0.00, 0.18, 0.32, 0.82),
        (0.68, 0.18, 1.00, 0.82),
    ],
    "basketball": [
        (0.18, 0.00, 0.82, 0.42),
        (0.28, 0.05, 0.72, 0.34),
    ],
}
def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


DEBUG_MODE = _env_flag("DEBUG_MODE", default=False)
DRAW_DEBUG_OVERLAY = DEBUG_MODE
DEBUG_CONSOLE_LOGS = DEBUG_MODE
DEBUG_SAVE_FRAMES = DEBUG_MODE
DEBUG_HEARTBEAT_EVERY_N_FRAMES = 50
UPSCALE_FACTOR = 1.5
DEBUG_LOG_SAVED_FRAMES = DEBUG_MODE
YOLO_IMGSZ = 768
BALL_TILE_SCAN_INTERVAL = {
    "football": 3,
    # Тайлы каждые 4 кадра вместо каждого — главный источник speedup.
    "basketball": 4,
}
BALL_STICKY_FRAMES = {
    "football": 1,
    "basketball": 2,
}
GOAL_CONFIRM_FRAMES = {
    "football": 2,
    "basketball": 2,
}
# При 6fps баскетбольный мяч смещается на 100-200px между кадрами —
# требование 2 совпадений подряд через _same_object(max_dist=0.20) слишком жёсткое.
# 1 = засчитываем мяч сразу при первом надёжном обнаружении.
OBJECT_CONFIRM_FRAMES = 1
TILE_LAYOUTS = {
    "football": [(0.15, 0.10, 0.85, 0.90), (0.00, 0.20, 1.00, 0.85)],
    "basketball": [
        (0.08, 0.00, 0.92, 0.72),
        (0.22, 0.08, 0.78, 0.62),
        (0.00, 0.00, 0.60, 0.70),
        (0.40, 0.00, 1.00, 0.70),
    ],
}


_unified_model: YOLO | None = None


def get_unified_model() -> YOLO:
    global _unified_model
    if _unified_model is None:
        if not UNIFIED_MODEL_PATH.exists():
            raise FileNotFoundError(f"Unified detector weights not found: {UNIFIED_MODEL_PATH}")
        _unified_model = YOLO(str(UNIFIED_MODEL_PATH))
    return _unified_model


@dataclass
class DetectedEvent:
    event_type: str   # "goal"
    timestamp: float  # секунды от начала видео (из CAP_PROP_POS_MSEC)
    confidence: float


@dataclass
class DetectionBox:
    label: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


def _draw_box(frame, box: DetectionBox, color: tuple[int, int, int], text: str) -> None:
    x1, y1, x2, y2 = map(int, [box.x1, box.y1, box.x2, box.y2])
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        text,
        (x1, max(18, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
        cv2.LINE_AA,
    )


def _preprocess_frame(frame: np.ndarray) -> np.ndarray:
    if UPSCALE_FACTOR == 1.0:
        enlarged = frame
    else:
        enlarged = cv2.resize(frame, None, fx=UPSCALE_FACTOR, fy=UPSCALE_FACTOR, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(enlarged, (0, 0), 0.8)
    return cv2.addWeighted(enlarged, 1.20, blur, -0.20, 0)


def _should_prepare_debug_frame() -> bool:
    return DRAW_DEBUG_OVERLAY or DEBUG_SAVE_FRAMES


def _prepare_debug_frame(frame: np.ndarray) -> np.ndarray:
    if UPSCALE_FACTOR == 1.0:
        return frame.copy()
    enlarged = cv2.resize(frame, None, fx=UPSCALE_FACTOR, fy=UPSCALE_FACTOR, interpolation=cv2.INTER_CUBIC)
    return enlarged


def _iou(a: DetectionBox, b: DetectionBox) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    union = a.width * a.height + b.width * b.height - inter
    return inter / max(union, 1e-6)


def _dedupe_boxes(boxes: list[DetectionBox], iou_threshold: float = 0.45) -> list[DetectionBox]:
    kept: list[DetectionBox] = []
    for box in sorted(boxes, key=lambda b: b.confidence, reverse=True):
        if all(_iou(box, existing) < iou_threshold for existing in kept):
            kept.append(box)
    return kept


def _distance(a: DetectionBox, b: DetectionBox) -> float:
    return ((a.cx - b.cx) ** 2 + (a.cy - b.cy) ** 2) ** 0.5


def _same_object(
    current: DetectionBox,
    previous: Optional[DetectionBox],
    frame_shape: tuple[int, int, int],
    max_dist_ratio: float = 0.18,
) -> bool:
    if previous is None:
        return False
    h, w = frame_shape[:2]
    return _distance(current, previous) <= max_dist_ratio * max(w, h)

def _crop_box(frame: np.ndarray, box: DetectionBox, inset_ratio: float = 0.18) -> np.ndarray:
    h, w = frame.shape[:2]
    pad_x = box.width * inset_ratio
    pad_y = box.height * inset_ratio
    x1 = max(0, int(box.x1 + pad_x))
    y1 = max(0, int(box.y1 + pad_y))
    x2 = min(w, int(box.x2 - pad_x))
    y2 = min(h, int(box.y2 - pad_y))
    if x2 <= x1 or y2 <= y1:
        x1 = max(0, int(box.x1))
        y1 = max(0, int(box.y1))
        x2 = min(w, int(box.x2))
        y2 = min(h, int(box.y2))
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=frame.dtype)
    return frame[y1:y2, x1:x2]


def _basketball_color_ratios(frame: np.ndarray, box: DetectionBox) -> tuple[float, float, float, float, float]:
    crop = _crop_box(frame, box)
    if crop.size == 0:
        return 0.0, 1.0, 1.0, 1.0, 1.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Для этой трансляции считаем только коричневый и темно-оранжевый диапазон,
    # без ярко-оранжевых тонов.
    orange = cv2.inRange(hsv, (6, 55, 20), (18, 255, 190))
    # Белые/серые яркие объекты вроде табличек, кресел, ламп и бликов.
    neutral = cv2.inRange(hsv, (0, 0, 90), (180, 70, 255))
    # Светлый бежевый/деревянный паркет.
    beige = cv2.inRange(hsv, (10, 20, 120), (32, 135, 255))
    # Сужаем до H=0-7 (был 0-10): тёмно-оранжевый мяч имеет H≈8-18 и теперь не попадает сюда.
    red1 = cv2.inRange(hsv, (0,  70, 45), (7,  255, 255))
    red2 = cv2.inRange(hsv, (170, 70, 45), (180, 255, 255))
    red = cv2.bitwise_or(red1, red2)
    purple = cv2.inRange(hsv, (118, 35, 35), (165, 255, 255))
    return (
        float(orange.mean() / 255.0),
        float(neutral.mean() / 255.0),
        float(beige.mean() / 255.0),
        float(red.mean() / 255.0),
        float(purple.mean() / 255.0),
    )


def _box_area(box: DetectionBox) -> float:
    return max(box.width, 0.0) * max(box.height, 0.0)


def _format_box(box: DetectionBox) -> str:
    return (
        f"{box.label} conf={box.confidence:.2f} "
        f"xyxy=({int(box.x1)},{int(box.y1)},{int(box.x2)},{int(box.y2)}) "
        f"center=({int(box.cx)},{int(box.cy)})"
    )


def _draw_debug_overlay(
    frame,
    sport: str,
    balls: list[DetectionBox],
    targets: list[DetectionBox],
    using_cached_target: bool,
    ball_in_target: bool,
    timestamp: float,
) -> None:
    if not DRAW_DEBUG_OVERLAY:
        return

    ball_color = (0, 255, 255)
    target_color = (0, 200, 0) if not using_cached_target else (255, 200, 0)
    active_color = (0, 255, 0) if ball_in_target else (180, 180, 180)

    for ball in balls:
        _draw_box(frame, ball, ball_color, f"ball {ball.confidence:.2f}")

    for target in targets:
        label = f"{target.label} {target.confidence:.2f}"
        if using_cached_target:
            label = f"cached {label}"
        _draw_box(frame, target, target_color, label)

    cv2.putText(
        frame,
        f"{sport} | t={timestamp:.1f}s | in_target={ball_in_target}",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        active_color,
        2,
        cv2.LINE_AA,
    )

def _extract_ball_boxes(result, sport: str) -> list[DetectionBox]:
    names = result.names
    balls: list[DetectionBox] = []
    ball_labels = BALL_LABELS[sport]

    for box in result.boxes:
        cls_id = int(box.cls[0])
        label = names[cls_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        det = DetectionBox(label=label, confidence=conf, x1=x1, y1=y1, x2=x2, y2=y2)
        aspect = det.width / max(det.height, 1e-6)
        area = det.width * det.height

        if label not in ball_labels or conf < BALL_CONFIDENCE[sport]:
            continue
        # Фильтруем явные ложные срабатывания вроде руки/торса.
        if not (0.65 <= aspect <= 1.55):
            continue
        if area > 0.012 * (result.orig_shape[0] * result.orig_shape[1]):
            continue
        if sport == "basketball":
            # Нижние 35% только на полном кадре.
            frame_h = result.orig_shape[0]
            if frame_h > 300 and det.cy > frame_h * 0.80:
                continue
            # Мяч NBA на broadcast никогда не превышает ~80px.
            # 134x134 — это игрок или часть тела, не мяч.
            if det.width > 80 or det.height > 80:
                continue
            if area < 12:
                continue
        balls.append(det)

    return balls


def _extract_target_boxes(result, sport: str) -> list[DetectionBox]:
    names = result.names
    targets: list[DetectionBox] = []
    target_labels = TARGET_LABELS[sport]

    for box in result.boxes:
        cls_id = int(box.cls[0])
        label = names[cls_id]
        conf = float(box.conf[0])
        if label not in target_labels or conf < TARGET_CONFIDENCE[sport]:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        targets.append(DetectionBox(label=label, confidence=conf, x1=x1, y1=y1, x2=x2, y2=y2))

    return targets


def _normalize_basketball_targets(targets: list[DetectionBox], frame_shape: tuple[int, int, int]) -> list[DetectionBox]:
    if not targets:
        return []

    h, w = frame_shape[:2]
    rims = [
        t for t in targets
        if t.label in TARGET_LABELS["basketball"]
        and t.cy <= 0.45 * h
        and 0.0002 <= (t.width * t.height) / max(w * h, 1) <= 0.03
    ]
    if not rims:
        return []
    return [max(rims, key=lambda t: t.confidence)]


def _normalize_football_targets(targets: list[DetectionBox], frame_shape: tuple[int, int, int]) -> list[DetectionBox]:
    if not targets:
        return []
    h, w = frame_shape[:2]
    plausible = [
        t for t in targets
        if (t.cx <= 0.3 * w or t.cx >= 0.7 * w)
        and 0.18 * h <= t.cy <= 0.86 * h
        and 0.002 <= (t.width * t.height) / max(w * h, 1) <= 0.18
    ]
    if not plausible:
        return []
    goal_like = [t for t in plausible if t.label in TARGET_LABELS["football"]]
    pool = goal_like or plausible
    return [max(pool, key=lambda t: t.confidence)]


def _extract_ball_boxes_from_result(
    result,
    sport: str,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
) -> list[DetectionBox]:
    balls = _extract_ball_boxes(result, sport)
    if x_offset == 0.0 and y_offset == 0.0:
        return balls

    adjusted_balls = [
        DetectionBox(
            label=box.label,
            confidence=box.confidence,
            x1=box.x1 + x_offset,
            y1=box.y1 + y_offset,
            x2=box.x2 + x_offset,
            y2=box.y2 + y_offset,
        )
        for box in balls
    ]
    return adjusted_balls


def _extract_target_boxes_from_result(
    result,
    sport: str,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
) -> list[DetectionBox]:
    targets = _extract_target_boxes(result, sport)
    if x_offset == 0.0 and y_offset == 0.0:
        return targets

    adjusted_targets = [
        DetectionBox(
            label=box.label,
            confidence=box.confidence,
            x1=box.x1 + x_offset,
            y1=box.y1 + y_offset,
            x2=box.x2 + x_offset,
            y2=box.y2 + y_offset,
        )
        for box in targets
    ]
    return adjusted_targets

def _ball_inside_target(ball: DetectionBox, target: DetectionBox, padding_ratio: float) -> bool:
    width = max(target.x2 - target.x1, 1.0)
    height = max(target.y2 - target.y1, 1.0)
    pad_x = width * padding_ratio
    pad_y = height * padding_ratio

    return (
        target.x1 - pad_x <= ball.cx <= target.x2 + pad_x and
        target.y1 - pad_y <= ball.cy <= target.y2 + pad_y
    )


def _save_debug_frame(
    frame: np.ndarray,
    sport: str,
    timestamp: float,
    frame_idx: int,
    buckets: list[str],
) -> None:
    if not DEBUG_SAVE_FRAMES:
        return
    saved_paths: list[str] = []
    for bucket in buckets:
        out_dir = DEBUG_FRAMES_DIR / sport / bucket
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = out_dir / f"{sport}_{frame_idx:06d}_{timestamp:.2f}.png"
        cv2.imwrite(str(filename), frame)
        saved_paths.append(str(filename.relative_to(DEBUG_FRAMES_DIR)))
    if DEBUG_LOG_SAVED_FRAMES and saved_paths:
        print(f"[{sport}] saved debug frame -> {', '.join(saved_paths)}")


class BallIntoTargetDetector:
    """Срабатывает, когда YOLO видит мяч внутри bbox цели."""

    # Сколько последних позиций мяча храним для анализа направления.
    TRAJECTORY_HISTORY = 6

    def __init__(self, sport: str, debounce_frames: int):
        self.sport = sport
        self.debounce_frames = debounce_frames
        self._frames_since_event = debounce_frames + 1
        self._ball_was_in_target = False
        self._ball_in_target_streak = 0
        self._last_ball: Optional[DetectionBox] = None
        self._last_target: Optional[DetectionBox] = None
        self._ball_confirm_streak = 0
        self._target_confirm_streak = 0
        self._cached_balls: list[DetectionBox] = []
        self._ball_frames_left = 0
        self._cached_targets: list[DetectionBox] = []
        self._target_frames_left = 0
        self._frame_idx = 0
        # Очередь (cx, cy) последних подтверждённых позиций мяча.
        self._ball_trajectory: list[tuple[float, float]] = []

    def _update_trajectory(self, ball: Optional[DetectionBox]) -> None:
        """Обновляем историю позиций мяча."""
        if ball is None:
            return
        self._ball_trajectory.append((ball.cx, ball.cy))
        if len(self._ball_trajectory) > self.TRAJECTORY_HISTORY:
            self._ball_trajectory.pop(0)

    def _ball_approaching_target(self, target: DetectionBox) -> bool:
        """
        Проверяем направление движения мяча перед засчитыванием гола.

        Баскетбол: мяч должен двигаться ВНИЗ (cy растёт) — бросок сверху в кольцо.
        Футбол: мяч должен двигаться ГОРИЗОНТАЛЬНО к воротам (cx меняется в сторону цели).

        Если истории недостаточно — пропускаем проверку (не блокируем детекцию).
        """
        if len(self._ball_trajectory) < 3:
            return True

        first_cx, first_cy = self._ball_trajectory[0]
        last_cx,  last_cy  = self._ball_trajectory[-1]

        if self.sport == "basketball":
            # Мяч должен лететь вниз: cy последней точки больше первой.
            dy = last_cy - first_cy
            return dy > 0

        if self.sport == "football":
            # Мяч летит к воротам: cx движется в сторону цели.
            # Ворота либо левый край (target.cx < 0.3w), либо правый (target.cx > 0.7w).
            # Направление движения должно совпадать.
            dx = last_cx - first_cx
            if target.cx < first_cx:   # ворота левее мяча
                return dx < 0
            else:                       # ворота правее мяча
                return dx > 0

        return True

    def _run_detection(
        self,
        model: YOLO,
        frame: np.ndarray,
        use_tiles: bool,
        use_target_tiles: bool,
    ) -> tuple[list[DetectionBox], list[DetectionBox]]:
        balls: list[DetectionBox] = []
        targets: list[DetectionBox] = []

        full_results = model(frame, verbose=False, imgsz=YOLO_IMGSZ, conf=0.02)
        full_balls = _extract_ball_boxes_from_result(full_results[0], self.sport)
        full_targets = _extract_target_boxes_from_result(full_results[0], self.sport)
        balls.extend(full_balls)
        targets.extend(full_targets)

        if use_tiles:
            h, w = frame.shape[:2]
            for x0n, y0n, x1n, y1n in TILE_LAYOUTS[self.sport]:
                x0 = max(0, int(w * x0n))
                y0 = max(0, int(h * y0n))
                x1 = min(w, int(w * x1n))
                y1 = min(h, int(h * y1n))
                crop = frame[y0:y1, x0:x1]
                if crop.size == 0:
                    continue
                tile_results = model(crop, verbose=False, imgsz=YOLO_IMGSZ, conf=0.02)
                tile_balls = _extract_ball_boxes_from_result(tile_results[0], self.sport, x0, y0)
                balls.extend(tile_balls)

        if use_target_tiles:
            h, w = frame.shape[:2]
            for x0n, y0n, x1n, y1n in TARGET_TILE_LAYOUTS[self.sport]:
                x0 = max(0, int(w * x0n))
                y0 = max(0, int(h * y0n))
                x1 = min(w, int(w * x1n))
                y1 = min(h, int(h * y1n))
                crop = frame[y0:y1, x0:x1]
                if crop.size == 0:
                    continue
                tile_results = model(crop, verbose=False, imgsz=YOLO_IMGSZ, conf=0.02)
                tile_targets = _extract_target_boxes_from_result(tile_results[0], self.sport, x0, y0)
                targets.extend(tile_targets)

        return _dedupe_boxes(balls), _dedupe_boxes(targets)

    def _pick_best_ball(self, balls: list[DetectionBox], frame: np.ndarray) -> list[DetectionBox]:
        frame_shape = frame.shape
        if not balls:
            self._last_ball = None
            self._ball_confirm_streak = 0
            return []

        h, w = frame_shape[:2]
        if self._last_ball is not None:
            max_dist = 0.20 * max(w, h)
            nearby = [ball for ball in balls if _distance(ball, self._last_ball) <= max_dist]
            if nearby:
                chosen = max(nearby, key=lambda b: (b.confidence, -_distance(b, self._last_ball)))
                same_as_previous = _same_object(chosen, self._last_ball, frame_shape, max_dist_ratio=0.20)
                self._ball_confirm_streak = self._ball_confirm_streak + 1 if same_as_previous else 1
                self._last_ball = chosen
                return [chosen] if self._ball_confirm_streak >= OBJECT_CONFIRM_FRAMES else []

        if self.sport == "basketball":
            ranked: list[tuple[float, DetectionBox]] = []
            for ball in balls:
                if ball.cy < frame_shape[0] * 0.22 and ball.width < 26 and ball.height < 26:
                    continue
                if ball.width < 10 or ball.height < 10:
                    continue

                orange_ratio, neutral_ratio, beige_ratio, red_ratio, purple_ratio = _basketball_color_ratios(frame, ball)

                # Жесткие отрицательные фильтры. Если объект выглядит как крепеж, паркет,
                # светлая табличка или фиолетовый элемент арены, он не может быть мячом
                # независимо от confidence модели.
                if red_ratio > 0.18:
                    # Тёмно-оранжевый мяч имеет пересечение с красным диапазоном HSV.
                    # Если orange высокий — это мяч, а не красная форма игрока.
                    if orange_ratio < 0.3:
                        continue
                if beige_ratio > 0.36:
                    continue
                if neutral_ratio > 0.46:
                    continue
                if purple_ratio > 0.22:
                    continue

                # Без апскейла HSV-сигнал слабее: реальный мяч даёт ~0.007.
                # Порог снижен 0.035 → 0.005.
                if orange_ratio < 0.005:
                    continue
                if self._last_ball is not None:
                    prev_area = max(_box_area(self._last_ball), 1.0)
                    curr_area = _box_area(ball)
                    area_ratio = curr_area / prev_area
                    if prev_area >= 420 and curr_area < 120:
                        continue
                    if prev_area >= 280 and area_ratio < 0.30:
                        continue
                    if prev_area >= 180 and area_ratio < 0.22 and orange_ratio < 0.12:
                        continue
                score = ball.confidence + min(orange_ratio, 0.32) * 1.4 - abs(ball.width - ball.height) * 0.0007
                score -= neutral_ratio * 0.45
                score -= beige_ratio * 0.42
                score -= red_ratio * 0.50
                score -= purple_ratio * 0.50
                if self._last_ball is not None and _same_object(ball, self._last_ball, frame_shape, max_dist_ratio=0.24):
                    score += 0.20
                if 16 <= ball.width <= 140 and 16 <= ball.height <= 140:
                    score += 0.06
                ranked.append((score, ball))
            if ranked:
                if self._last_ball is not None:
                    nearby_ranked = [
                        item for item in ranked
                        if _same_object(item[1], self._last_ball, frame_shape, max_dist_ratio=0.30)
                    ]
                    if nearby_ranked:
                        chosen = max(nearby_ranked, key=lambda item: item[0])[1]
                    else:
                        best_score, best_ball = max(ranked, key=lambda item: item[0])
                        if best_score < 0.34:
                            self._ball_confirm_streak = 0
                            return []
                        chosen = best_ball
                else:
                    chosen = max(ranked, key=lambda item: item[0])[1]
            else:
                chosen = max(balls, key=lambda b: (b.confidence, -abs(b.width - b.height)))
        else:
            chosen = max(balls, key=lambda b: b.confidence)
        same_as_previous = _same_object(chosen, self._last_ball, frame_shape, max_dist_ratio=0.20)
        self._ball_confirm_streak = self._ball_confirm_streak + 1 if same_as_previous else 1
        self._last_ball = chosen
        return [chosen] if self._ball_confirm_streak >= OBJECT_CONFIRM_FRAMES else []

    def _pick_best_target(self, targets: list[DetectionBox], frame_shape: tuple[int, int, int]) -> list[DetectionBox]:
        if self.sport == "basketball":
            targets = _normalize_basketball_targets(targets, frame_shape)
        elif self.sport == "football":
            targets = _normalize_football_targets(targets, frame_shape)

        if not targets:
            self._last_target = None
            self._target_confirm_streak = 0
            return []

        if self._last_target is not None:
            nearby = [target for target in targets if _same_object(target, self._last_target, frame_shape, 0.25)]
            if nearby:
                chosen = max(nearby, key=lambda t: t.confidence)
                same_as_previous = _same_object(chosen, self._last_target, frame_shape, 0.25)
                self._target_confirm_streak = self._target_confirm_streak + 1 if same_as_previous else 1
                self._last_target = chosen
                return [chosen] if self._target_confirm_streak >= OBJECT_CONFIRM_FRAMES else []

        chosen = max(targets, key=lambda t: t.confidence)
        same_as_previous = _same_object(chosen, self._last_target, frame_shape, 0.25)
        self._target_confirm_streak = self._target_confirm_streak + 1 if same_as_previous else 1
        self._last_target = chosen
        return [chosen] if self._target_confirm_streak >= OBJECT_CONFIRM_FRAMES else []

    def process(self, frame: np.ndarray, timestamp: float) -> Optional[DetectedEvent]:
        model = get_unified_model()
        work_frame = _preprocess_frame(frame)
        debug_frame: Optional[np.ndarray] = None
        self._frames_since_event += 1
        self._frame_idx += 1

        use_tiles = (self._frame_idx % BALL_TILE_SCAN_INTERVAL[self.sport] == 0)
        use_target_tiles = (self._frame_idx % TARGET_SCAN_INTERVAL[self.sport] == 0)
        balls, targets = self._run_detection(
            model,
            work_frame,
            use_tiles=use_tiles,
            use_target_tiles=use_target_tiles,
        )

        balls = self._pick_best_ball(balls, work_frame)
        real_balls = list(balls)
        using_cached_ball = False
        if real_balls:
            self._cached_balls = real_balls
            self._ball_frames_left = BALL_STICKY_FRAMES[self.sport]
            # Обновляем траекторию только по реально детектированным позициям.
            self._update_trajectory(real_balls[0])
        elif self._ball_frames_left > 0:
            self._ball_frames_left -= 1
            balls = self._cached_balls
            using_cached_ball = True
        else:
            self._cached_balls = []
        targets = self._pick_best_target(targets, work_frame.shape)
        real_targets = list(targets)
        real_target_count = len(real_targets)
        using_cached_target = False
        scoring_targets = real_targets

        if real_targets:
            self._cached_targets = real_targets
            self._target_frames_left = TARGET_STICKY_FRAMES[self.sport]
        elif self._target_frames_left > 0:
            self._target_frames_left -= 1
            scoring_targets = self._cached_targets
            using_cached_target = True
        else:
            self._cached_targets = []

        if DEBUG_CONSOLE_LOGS:
            if balls or real_target_count or using_cached_target:
                print(f"[{self.sport}] frame={self._frame_idx} ts={timestamp:.2f}s")
                for ball in real_balls:
                    print(f"  ball   -> {_format_box(ball)}")
                if using_cached_ball:
                    for ball in balls:
                        print(f"  ball   -> cached {_format_box(ball)}")
                for target in real_targets:
                    print(f"  target -> {_format_box(target)}")
                if using_cached_target:
                    for target in scoring_targets:
                        cached = "cached "
                        print(f"  target -> {cached}{_format_box(target)}")
            elif self._frame_idx % DEBUG_HEARTBEAT_EVERY_N_FRAMES == 0:
                print(f"[{self.sport}] frame={self._frame_idx} ts={timestamp:.2f}s no detections")

        if not balls or not scoring_targets:
            self._ball_was_in_target = False
            self._ball_in_target_streak = 0
            buckets: list[str] = []
            if balls:
                buckets.append("ball")
            if real_target_count:
                buckets.append("target")
            if buckets:
                if _should_prepare_debug_frame():
                    debug_frame = _prepare_debug_frame(frame)
                    _draw_debug_overlay(
                        debug_frame,
                        self.sport,
                        balls,
                        scoring_targets,
                        using_cached_target,
                        False,
                        timestamp,
                    )
                _save_debug_frame(debug_frame, self.sport, timestamp, self._frame_idx, buckets)
            return None

        best_pair_conf = 0.0
        ball_in_target = False
        padding = TARGET_PADDING[self.sport]

        for ball in balls:
            for target in scoring_targets:
                if _ball_inside_target(ball, target, padding):
                    ball_in_target = True
                    best_pair_conf = max(best_pair_conf, min(ball.confidence, target.confidence))

        self._ball_in_target_streak = self._ball_in_target_streak + 1 if ball_in_target else 0

        buckets = []
        if balls:
            buckets.append("ball")
        if real_target_count:
            buckets.append("target")
        if ball_in_target:
            buckets.append("goal")
        if buckets:
            if _should_prepare_debug_frame():
                debug_frame = _prepare_debug_frame(frame)
                _draw_debug_overlay(
                    debug_frame,
                    self.sport,
                    balls,
                    scoring_targets,
                    using_cached_target,
                    ball_in_target,
                    timestamp,
                )
            _save_debug_frame(debug_frame, self.sport, timestamp, self._frame_idx, buckets)

        if (
            ball_in_target and
            self._ball_in_target_streak >= GOAL_CONFIRM_FRAMES[self.sport] and
            not self._ball_was_in_target and
            self._frames_since_event > self.debounce_frames and
            self._ball_approaching_target(scoring_targets[0])
        ):
            self._frames_since_event = 0
            self._ball_was_in_target = True
            return DetectedEvent("goal", timestamp, best_pair_conf)

        self._ball_was_in_target = ball_in_target
        return None


class FootballDetector(BallIntoTargetDetector):
    def __init__(self):
        super().__init__(sport="football", debounce_frames=15)


class BasketballDetector(BallIntoTargetDetector):
    def __init__(self):
        super().__init__(sport="basketball", debounce_frames=12)


def make_detector(sport: str):
    """Фабрика детекторов по виду спорта."""
    if sport == "football":
        return FootballDetector()
    if sport == "basketball":
        return BasketballDetector()
    raise ValueError(f"Unknown sport: {sport}")
