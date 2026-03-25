# Basketball Target Dataset

Expected structure:

```text
backend/data/basketball_target/
  images/
    train/
    val/
  labels/
    train/
    val/
```

Recommended classes:

- `rim`
- `backboard`
- `stanchion`

Optional:

- `target`

Use YOLO txt labels:

```text
<class_id> <x_center> <y_center> <width> <height>
```

Normalized to image width/height.

How to prepare frames:

```bash
cd backend
source venv/bin/activate
python3 prepare_target_dataset.py --sport basketball --every-seconds 0.5 --limit 500
```

How to annotate:

1. Open images from `images/train` and `images/val` in CVAT, Label Studio, or LabelImg.
2. Create classes in this exact order:
   `rim`, `backboard`, `stanchion`
3. Draw separate boxes:
   `rim` -> only the ring/net area
   `backboard` -> the transparent/white board
   `stanchion` -> the support arm / crane / base structure
4. If one part is not visible, do not invent it; just leave that class absent in the frame.
5. If basket is not visible at all, leave the `.txt` file empty.

Tips:

- Add many hard negatives with stands, scoreboards, benches, Gatorade coolers, and white signs.
- Tight boxes are important. Do not box the whole basket construction as `rim`.
