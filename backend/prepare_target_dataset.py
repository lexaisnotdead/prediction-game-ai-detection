from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from stream import MATCHES, get_stream_url


ROOT_DIR = Path(__file__).resolve().parent
DATA_ROOT = ROOT_DIR / "data"
DATASETS = {
    "football": DATA_ROOT / "football_target",
    "basketball": DATA_ROOT / "basketball_target",
}


def ensure_layout(root: Path) -> None:
    for rel in ("images/train", "images/val", "labels/train", "labels/val"):
        (root / rel).mkdir(parents=True, exist_ok=True)


def extract_frames(
    sport: str,
    every_seconds: float,
    limit: int,
    val_ratio: float,
    width: int,
) -> tuple[int, Path]:
    dataset_root = DATASETS[sport]
    ensure_layout(dataset_root)

    match = MATCHES[sport]
    stream_url = get_stream_url(match["url"])
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open stream for {sport}")

    start_offset = float(match.get("start_offset", 0.0))
    if start_offset > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_offset * 1000.0)

    saved = 0
    next_capture_ts = start_offset
    sample_idx = 0

    while saved < limit:
        ret, frame = cap.read()
        if not ret:
            break

        ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if ts < next_capture_ts:
            continue

        h, w = frame.shape[:2]
        if width > 0 and w > width:
            scale = width / w
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        split = "val" if (sample_idx % max(1, round(1 / max(val_ratio, 1e-6)))) == 0 else "train"
        img_name = f"{sport}_{sample_idx:05d}_{ts:.2f}.jpg"
        label_name = f"{sport}_{sample_idx:05d}_{ts:.2f}.txt"

        img_path = dataset_root / "images" / split / img_name
        label_path = dataset_root / "labels" / split / label_name

        cv2.imwrite(str(img_path), frame)
        label_path.touch(exist_ok=True)

        saved += 1
        sample_idx += 1
        next_capture_ts = ts + every_seconds

    cap.release()
    return saved, dataset_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare target dataset frames for PredictSport")
    parser.add_argument("--sport", choices=["football", "basketball"], required=True)
    parser.add_argument("--every-seconds", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--width", type=int, default=1280)
    args = parser.parse_args()

    saved, root = extract_frames(
        sport=args.sport,
        every_seconds=args.every_seconds,
        limit=args.limit,
        val_ratio=args.val_ratio,
        width=args.width,
    )
    print(f"Prepared {saved} frames in {root}")


if __name__ == "__main__":
    main()
