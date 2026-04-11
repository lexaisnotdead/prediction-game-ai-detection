from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from ultralytics import YOLO
import yaml


ROOT_DIR = Path(__file__).resolve().parent
DATASET_YAML = ROOT_DIR / "data" / "unified" / "dataset.yaml"
RUNS_DIR = ROOT_DIR / "runs" / "unified_training"
DEFAULT_OUTPUT = ROOT_DIR / "models" / "trained" / "unified_detector.pt"


def _count_files(path: Path, pattern: str) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.glob(pattern))


def _validate_dataset(dataset_yaml: Path) -> dict[str, int]:
    dataset_root = dataset_yaml.parent
    counts = {
        "train_images": _count_files(dataset_root / "images" / "train", "*"),
        "train_labels": _count_files(dataset_root / "labels" / "train", "*.txt"),
        "val_images": _count_files(dataset_root / "images" / "val", "*"),
        "val_labels": _count_files(dataset_root / "labels" / "val", "*.txt"),
    }

    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {dataset_yaml}")
    if counts["train_images"] == 0 or counts["train_labels"] == 0:
        raise ValueError(f"Train split is empty: {counts}")
    if counts["val_images"] == 0 or counts["val_labels"] == 0:
        raise ValueError(f"Validation split is empty: {counts}")
    return counts


def _materialize_absolute_dataset_yaml(dataset_yaml: Path) -> Path:
    dataset_yaml = dataset_yaml.resolve()
    dataset_root = dataset_yaml.parent.resolve()
    data = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8")) or {}

    absolute_data = {
        **data,
        "path": str(dataset_root),
        "train": "images/train",
        "val": "images/val",
    }

    absolute_yaml = dataset_root / "dataset.absolute.yaml"
    absolute_yaml.write_text(
        yaml.safe_dump(absolute_data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return absolute_yaml


def train_unified_model(
    model_name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    workers: int,
    patience: int,
    cache: bool,
    run_name: str,
    output_path: Path,
    dataset_yaml: Path = DATASET_YAML,
) -> Path:
    dataset_yaml = dataset_yaml.resolve()
    counts = _validate_dataset(dataset_yaml)
    training_dataset_yaml = _materialize_absolute_dataset_yaml(dataset_yaml)

    model = YOLO(model_name)
    results = model.train(
        data=str(training_dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        patience=patience,
        cache=cache,
        project=str(RUNS_DIR),
        name=run_name,
        exist_ok=True,
    )

    save_dir = Path(results.save_dir)
    best = save_dir / "weights" / "best.pt"
    last = save_dir / "weights" / "last.pt"
    source = best if best.exists() else last
    if not source.exists():
        raise FileNotFoundError(f"Training finished but no weights were found in {save_dir / 'weights'}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output_path)

    metadata = {
        "dataset_yaml": str(dataset_yaml),
        "training_dataset_yaml": str(training_dataset_yaml),
        "output_path": str(output_path.resolve()),
        "source_weights": str(source.resolve()),
        "counts": counts,
        "train_args": {
            "model_name": model_name,
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "device": device,
            "workers": workers,
            "patience": patience,
            "cache": cache,
            "run_name": run_name,
        },
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune one unified Ultralytics detection model on backend/data/unified"
    )
    parser.add_argument(
        "--model",
        default="yolo11s.pt",
        help="Base Ultralytics detection checkpoint or local .pt file",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--run-name", default="predictsport_unified_train")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to copy the final trained checkpoint",
    )
    parser.add_argument(
        "--dataset-yaml",
        type=Path,
        default=DATASET_YAML,
        help="Dataset yaml to train on",
    )
    args = parser.parse_args()

    output = train_unified_model(
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        cache=args.cache,
        run_name=args.run_name,
        output_path=args.output.resolve(),
        dataset_yaml=args.dataset_yaml.resolve(),
    )
    print(f"Saved unified model to: {output}")


if __name__ == "__main__":
    main()
