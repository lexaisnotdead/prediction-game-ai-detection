# Football Target Dataset

Expected structure:

```text
backend/data/football_target/
  images/
    train/
    val/
  labels/
    train/
    val/
```

Classes:

- `goal`

How to prepare frames:

```bash
cd backend
source venv/bin/activate
python3 prepare_target_dataset.py --sport football --every-seconds 0.5 --limit 400
```

Use YOLO txt labels:

```text
<class_id> <x_center> <y_center> <width> <height>
```

Normalized to image width/height.

How to annotate:

1. Open images from `images/train` and `images/val` in CVAT, Label Studio, or LabelImg.
2. Create one class: `goal`.
3. Draw one box around the full visible goal structure.
4. If goal is partly visible, still annotate the visible goal area.
5. If no goal is visible, leave the `.txt` file empty.

Tips:

- Include hard negatives where stands, ad boards, and posts are visible but no real goal.
- Keep labels tight around the goal, not the whole penalty area.
