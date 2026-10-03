"""Perception and observation logic for individual video frames."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_YOLO_MODEL = "yolo11n.pt"


@dataclass(frozen=True)
class Observation:
    kind: str
    label: str
    confidence: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


class YoloObjectDetector:
    """Optional object detector backed by a local Ultralytics model."""

    def __init__(
        self,
        model_path: Path | str,
        *,
        confidence_threshold: float = 0.5,
    ):
        from ultralytics import YOLO

        self.model_path = str(model_path)
        self.model = YOLO(str(model_path))
        self.confidence_threshold = confidence_threshold
        self._embedding_model: Any | None = None

    @property
    def class_names(self) -> list[str]:
        names = self.model.names
        values = names.values() if isinstance(names, dict) else names
        return [str(name) for name in values]

    def observe(self, frame: np.ndarray) -> list[Observation]:
        result = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.confidence_threshold,
            verbose=False,
        )[0]
        if result.boxes is None:
            return []

        frame_height, frame_width = frame.shape[:2]
        observations: list[Observation] = []
        classes = result.boxes.cls.cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()
        boxes_xyxy = result.boxes.xyxy.cpu().tolist()
        track_ids = (
            result.boxes.id.int().cpu().tolist()
            if result.boxes.id is not None
            else [None] * len(classes)
        )
        for class_id, confidence, (x1, y1, x2, y2), track_id in zip(
            classes, confidences, boxes_xyxy, track_ids
        ):
            if track_id is None:
                continue
            label = str(result.names[int(class_id)])
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            observations.append(
                Observation(
                    "object",
                    label,
                    round(float(confidence), 4),
                    details={
                        "bbox_xywh": {
                            "x": round(float(x1), 2),
                            "y": round(float(y1), 2),
                            "width": round(float(width), 2),
                            "height": round(float(height), 2),
                        },
                        "bbox_normalized_xywh": {
                            "x": round(float(x1 / frame_width), 6),
                            "y": round(float(y1 / frame_height), 6),
                            "width": round(float(width / frame_width), 6),
                            "height": round(float(height / frame_height), 6),
                        },
                        "coordinate_space": "pixels; top-left origin",
                        "frame_size": {
                            "width": frame_width,
                            "height": frame_height,
                        },
                        "track_id": int(track_id),
                    },
                )
            )
        return observations

    def embed_detection(
        self, frame: np.ndarray, bbox: tuple[float, float, float, float]
    ) -> tuple[float, ...] | None:
        """Extract a normalized YOLO appearance embedding for one object crop."""

        x, y, width, height = bbox
        frame_height, frame_width = frame.shape[:2]
        left = max(0, min(frame_width, int(x)))
        top = max(0, min(frame_height, int(y)))
        right = max(left + 1, min(frame_width, int(x + width)))
        bottom = max(top + 1, min(frame_height, int(y + height)))
        crop = frame[top:bottom, left:right]
        if crop.size == 0:
            return None

        if self._embedding_model is None:
            from ultralytics import YOLO

            self._embedding_model = YOLO(self.model_path)
        embeddings = self._embedding_model.embed(crop, verbose=False)
        if not embeddings:
            return None
        vector = embeddings[0].detach().cpu().float().flatten().numpy()
        magnitude = float(np.linalg.norm(vector))
        if magnitude == 0:
            return None
        return tuple(float(value) for value in vector / magnitude)

    def reset(self) -> None:
        """Start ByteTrack with fresh state for a new video."""

        self.model.predictor = None


class PerceptionLayer:
    """Turns pixels into small, explicit observations.

    The built-in signals are intentionally factual and local. Object recognition is
    enabled only when the caller supplies a local YOLO model.
    """

    COLOR_PALETTE = {
        "black": (20, 20, 20),
        "white": (235, 235, 235),
        "gray": (128, 128, 128),
        "red": (200, 55, 55),
        "green": (55, 170, 75),
        "blue": (55, 95, 200),
        "yellow": (210, 200, 55),
        "orange": (220, 130, 45),
        "purple": (140, 70, 170),
        "brown": (120, 80, 45),
    }

    def __init__(self, yolo_model: Path | str | None = DEFAULT_YOLO_MODEL):
        self.previous_gray: np.ndarray | None = None
        self.object_detector = YoloObjectDetector(yolo_model) if yolo_model else None

    def reset(self) -> None:
        self.previous_gray = None
        if self.object_detector is not None:
            self.object_detector.reset()

    def observe(
        self,
        frame: np.ndarray,
        *,
        object_observations: list[Observation] | None = None,
    ) -> list[Observation]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness_value = float(gray.mean())
        brightness = (
            "dark" if brightness_value < 70 else "bright" if brightness_value > 185 else "normal"
        )

        sharpness_value = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        focus = "blurry" if sharpness_value < 75 else "sharp"

        mean_bgr = frame.reshape(-1, 3).mean(axis=0)
        mean_rgb = np.array([mean_bgr[2], mean_bgr[1], mean_bgr[0]])
        dominant_color = min(
            self.COLOR_PALETTE,
            key=lambda name: np.linalg.norm(
                mean_rgb - np.array(self.COLOR_PALETTE[name])
            ),
        )

        observations = [
            Observation(
                "lighting", brightness, details={"mean_brightness": round(brightness_value, 2)}
            ),
            Observation(
                "focus", focus, details={"laplacian_variance": round(sharpness_value, 2)}
            ),
            Observation(
                "dominant_color",
                dominant_color,
                details={"mean_rgb": [round(float(value), 2) for value in mean_rgb]},
            ),
        ]

        motion_label = "baseline"
        motion_value = 0.0
        if self.previous_gray is not None:
            resized_previous = cv2.resize(
                self.previous_gray, (gray.shape[1], gray.shape[0])
            )
            motion_value = float(
                cv2.absdiff(gray, resized_previous).mean() / 255.0
            )
            motion_label = (
                "high" if motion_value >= 0.18 else "moderate" if motion_value >= 0.06 else "low"
            )
        observations.append(
            Observation(
                "motion",
                motion_label,
                details={"normalized_frame_difference": round(motion_value, 4)},
            )
        )
        self.previous_gray = gray

        if object_observations is not None:
            observations.extend(object_observations)
        elif self.object_detector:
            observations.extend(self.object_detector.observe(frame))

        object_labels = list(
            dict.fromkeys(item.label for item in observations if item.kind == "object")
        )
        summary = f"{brightness.capitalize()} lighting; {focus}; mostly {dominant_color}; {motion_label} motion"
        if object_labels:
            summary += "; objects: " + ", ".join(object_labels)
        observations.append(Observation("summary", summary))
        return observations
