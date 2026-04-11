from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

if "ultralytics" not in sys.modules:
    fake_ultralytics = types.ModuleType("ultralytics")

    class _FakeYOLO:
        def __init__(self, *args, **kwargs) -> None:
            pass

    fake_ultralytics.YOLO = _FakeYOLO
    sys.modules["ultralytics"] = fake_ultralytics

if "cv2" not in sys.modules:
    sys.modules["cv2"] = types.ModuleType("cv2")

if "numpy" not in sys.modules:
    fake_numpy = types.ModuleType("numpy")
    fake_numpy.ndarray = object
    fake_numpy.empty = lambda *args, **kwargs: []  # type: ignore[attr-defined]
    sys.modules["numpy"] = fake_numpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detector import (  # noqa: E402
    _extract_ball_boxes,
    _extract_target_boxes,
    _normalize_basketball_targets,
    _normalize_football_targets,
    make_detector,
)


class _FakeBox:
    class _Vector:
        def __init__(self, values: tuple[float, ...]) -> None:
            self._values = values

        def tolist(self) -> list[float]:
            return list(self._values)

    def __init__(self, cls_id: int, conf: float, xyxy: tuple[float, float, float, float]) -> None:
        self.cls = [float(cls_id)]
        self.conf = [float(conf)]
        self.xyxy = [self._Vector(xyxy)]


class _FakeResult:
    def __init__(self, names: dict[int, str], boxes: list[_FakeBox], orig_shape: tuple[int, int]) -> None:
        self.names = names
        self.boxes = boxes
        self.orig_shape = orig_shape


class DetectorHelpersTests(unittest.TestCase):
    def test_extract_ball_boxes_uses_unified_labels(self) -> None:
        result = _FakeResult(
            names={0: "basketball_ball", 1: "football_ball", 2: "basketball_rim"},
            boxes=[
                _FakeBox(0, 0.9, (100, 100, 130, 130)),
                _FakeBox(1, 0.9, (120, 100, 150, 130)),
                _FakeBox(2, 0.9, (200, 50, 260, 90)),
            ],
            orig_shape=(720, 1280),
        )

        basketball_balls = _extract_ball_boxes(result, "basketball")
        football_balls = _extract_ball_boxes(result, "football")

        self.assertEqual([box.label for box in basketball_balls], ["basketball_ball"])
        self.assertEqual([box.label for box in football_balls], ["football_ball"])

    def test_extract_target_boxes_uses_unified_labels(self) -> None:
        result = _FakeResult(
            names={0: "basketball_rim", 1: "football_goal", 2: "football_ball"},
            boxes=[
                _FakeBox(0, 0.8, (200, 80, 260, 120)),
                _FakeBox(1, 0.8, (900, 180, 1100, 500)),
                _FakeBox(2, 0.8, (500, 300, 520, 320)),
            ],
            orig_shape=(720, 1280),
        )

        basketball_targets = _extract_target_boxes(result, "basketball")
        football_targets = _extract_target_boxes(result, "football")

        self.assertEqual([box.label for box in basketball_targets], ["basketball_rim"])
        self.assertEqual([box.label for box in football_targets], ["football_goal"])

    def test_normalize_basketball_targets_filters_implausible_rims(self) -> None:
        result = _FakeResult(
            names={0: "basketball_rim"},
            boxes=[
                _FakeBox(0, 0.7, (300, 120, 360, 160)),
                _FakeBox(0, 0.9, (300, 500, 360, 540)),
            ],
            orig_shape=(720, 1280),
        )

        normalized = _normalize_basketball_targets(_extract_target_boxes(result, "basketball"), (720, 1280, 3))

        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0].label, "basketball_rim")
        self.assertLess(normalized[0].cy, 720 * 0.45)

    def test_normalize_football_targets_prefers_plausible_goal_edges(self) -> None:
        result = _FakeResult(
            names={0: "football_goal"},
            boxes=[
                _FakeBox(0, 0.6, (500, 150, 800, 450)),
                _FakeBox(0, 0.9, (930, 170, 1180, 500)),
            ],
            orig_shape=(720, 1280),
        )

        normalized = _normalize_football_targets(_extract_target_boxes(result, "football"), (720, 1280, 3))

        self.assertEqual(len(normalized), 1)
        self.assertGreater(normalized[0].cx, 1280 * 0.7)

    def test_make_detector_rejects_unknown_sport(self) -> None:
        with self.assertRaises(ValueError):
            make_detector("tennis")


if __name__ == "__main__":
    unittest.main()
