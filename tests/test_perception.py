from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from digital_hippocampus.perception import YoloObjectDetector


class FakeTensor:
    def __init__(self, values: list):
        self.values = values

    def cpu(self) -> "FakeTensor":
        return self

    def tolist(self) -> list:
        return self.values

    def int(self) -> "FakeTensor":
        return self


class FakeModel:
    names = {0: "person"}

    def track(self, _frame: np.ndarray, **_kwargs: object) -> list:
        boxes = SimpleNamespace(
            cls=FakeTensor([0.0]),
            conf=FakeTensor([0.91234]),
            xyxy=FakeTensor([[10.0, 20.0, 50.0, 80.0]]),
            id=FakeTensor([7]),
        )
        return [SimpleNamespace(boxes=boxes, names=self.names)]


class YoloObjectDetectorTest(unittest.TestCase):
    def test_detection_contains_pixel_and_normalized_bbox(self) -> None:
        detector = YoloObjectDetector.__new__(YoloObjectDetector)
        detector.model = FakeModel()
        detector.confidence_threshold = 0.5

        observations = detector.observe(np.zeros((100, 200, 3), dtype=np.uint8))

        self.assertEqual(detector.class_names, ["person"])
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].label, "person")
        self.assertEqual(observations[0].confidence, 0.9123)
        self.assertEqual(
            observations[0].details["bbox_xywh"],
            {"x": 10.0, "y": 20.0, "width": 40.0, "height": 60.0},
        )
        self.assertEqual(
            observations[0].details["bbox_normalized_xywh"],
            {"x": 0.05, "y": 0.2, "width": 0.2, "height": 0.6},
        )
        self.assertEqual(observations[0].details["track_id"], 7)


if __name__ == "__main__":
    unittest.main()
