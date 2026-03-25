from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parent
DATASETS = {
    "football": ROOT_DIR / "data" / "football_target" / "dataset.yaml",
    "basketball": ROOT_DIR / "data" / "basketball_target" / "dataset.yaml",
}
OUTPUTS = {
    "football": ROOT_DIR / "models" / "trained" / "football_target.pt",
    "basketball": ROOT_DIR / "models" / "trained" / "basketball_target.pt",
}


def train_target_model(
    sport: str,
    model_name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
) -> Path:
    data = DATASETS[sport]
    if not data.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data}")

    model = YOLO(model_name)
    project_dir = ROOT_DIR / "runs" / "target_training"
    run_name = f"{sport}_target"
    results = model.train(
        data=str(data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"Training finished but best.pt not found: {best}")

    out = OUTPUTS[sport]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(best.read_bytes())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train custom target detector for PredictSport")
    parser.add_argument("--sport", choices=["football", "basketball"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    out = train_target_model(
        sport=args.sport,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )
    print(f"Saved trained target model to: {out}")


if __name__ == "__main__":
    main()
