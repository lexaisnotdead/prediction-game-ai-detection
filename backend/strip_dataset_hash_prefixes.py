from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


HASH_PREFIX_RE = re.compile(
    r"^(?:[0-9a-f]{8}|[0-9a-f]{32}|[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})-(.+)$",
    re.IGNORECASE,
)


def strip_hash_prefixes(dataset_root: Path) -> dict[str, object]:
    renamed: list[dict[str, str]] = []

    for relative_dir in ("images/train", "images/val", "labels/train", "labels/val"):
        directory = dataset_root / relative_dir
        if not directory.exists():
            continue

        for path in sorted(p for p in directory.iterdir() if p.is_file() and not p.name.startswith(".")):
            match = HASH_PREFIX_RE.match(path.name)
            if not match:
                continue

            target = path.with_name(match.group(1))
            if target.exists():
                raise FileExistsError(f"Cannot rename {path} to {target}: target already exists")

            path.rename(target)
            renamed.append(
                {
                    "from": str(path.relative_to(dataset_root)),
                    "to": str(target.relative_to(dataset_root)),
                }
            )

    summary = {
        "dataset_root": str(dataset_root),
        "renamed_count": len(renamed),
        "renamed": renamed,
    }
    (dataset_root / "strip_hash_prefixes_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Strip hash prefixes from dataset image/label filenames")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "unified",
    )
    args = parser.parse_args()

    summary = strip_hash_prefixes(args.dataset_root.expanduser().resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
