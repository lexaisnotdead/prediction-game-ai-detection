from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def read_classes(dataset_root: Path) -> list[str]:
    classes_path = dataset_root / "classes.txt"
    if not classes_path.exists():
        raise FileNotFoundError(f"classes.txt not found in {dataset_root}")
    return [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def copy_split(source_root: Path, split: str, output_root: Path) -> dict[str, int]:
    source_images = source_root / "images"
    source_labels = source_root / "labels"
    target_images = output_root / "images" / split
    target_labels = output_root / "labels" / split

    target_images.mkdir(parents=True, exist_ok=True)
    target_labels.mkdir(parents=True, exist_ok=True)

    copied_images = 0
    copied_labels = 0

    if source_images.exists():
        for path in sorted(p for p in source_images.iterdir() if p.is_file() and not p.name.startswith(".")):
            shutil.copy2(path, target_images / path.name)
            copied_images += 1

    if source_labels.exists():
        for path in sorted(p for p in source_labels.iterdir() if p.is_file() and not p.name.startswith(".")):
            shutil.copy2(path, target_labels / path.name)
            copied_labels += 1

    return {"images": copied_images, "labels": copied_labels}


def write_dataset_yaml(output_root: Path, class_names: list[str]) -> None:
    lines = [
        "path: .",
        "train: images/train",
        "val: images/val",
        "",
        f"nc: {len(class_names)}",
        "names:",
    ]
    lines.extend(f"  {idx}: {name}" for idx, name in enumerate(class_names))
    (output_root / "dataset.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def import_unified_dataset(train_source: Path, val_source: Path, output_root: Path) -> dict[str, object]:
    train_classes = read_classes(train_source)
    val_classes = read_classes(val_source)
    if train_classes != val_classes:
        raise ValueError("Train/val classes.txt do not match")

    output_root.mkdir(parents=True, exist_ok=True)
    train_stats = copy_split(train_source, "train", output_root)
    val_stats = copy_split(val_source, "val", output_root)

    (output_root / "classes.txt").write_text("\n".join(train_classes) + "\n", encoding="utf-8")
    write_dataset_yaml(output_root, train_classes)

    summary = {
        "output_root": str(output_root),
        "classes": train_classes,
        "train": train_stats,
        "val": val_stats,
    }
    (output_root / "import_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Import Label Studio train/val exports into backend/data")
    parser.add_argument("--train-source", type=Path, required=True)
    parser.add_argument("--val-source", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "unified",
    )
    args = parser.parse_args()

    summary = import_unified_dataset(
        train_source=args.train_source.expanduser().resolve(),
        val_source=args.val_source.expanduser().resolve(),
        output_root=args.output_root.resolve(),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
