from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from digital_hippocampus.pipeline import VideoPipeline
from digital_hippocampus.perception import Observation
from digital_hippocampus.database import MemoryDatabase
from digital_hippocampus.symbolic_events import (
    EventEvidence,
    EventExtractionResult,
    SymbolicEvent,
)


class FakeEventExtractor:
    provider = "gemini"
    model = "fake-gemini"
    schema_version = "test-schema"

    def __init__(self, *, fails: bool = False):
        self.fails = fails

    def extract(self, video: dict) -> EventExtractionResult:
        if self.fails:
            raise RuntimeError("simulated Gemini failure")
        frame = video["frames"][0]
        return EventExtractionResult(
            events=[
                SymbolicEvent(
                    event_id="event-1",
                    video_id=video["id"],
                    event_type="moved",
                    start_time=0.1,
                    end_time=0.4,
                    actor="person",
                    object="box",
                    source_location="floor",
                    destination_location="table",
                    confidence=0.82,
                    evidence=[
                        EventEvidence(
                            timestamp_seconds=0.3,
                            frame_id=frame["id"],
                            frame_number=frame["frame_number"],
                            frame_timestamp_seconds=frame["timestamp_seconds"],
                        )
                    ],
                )
            ],
            rejected_event_count=0,
        )


class VideoPipelineTest(unittest.TestCase):
    @staticmethod
    def make_video(path: Path) -> None:
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            5.0,
            (64, 48),
        )
        if not writer.isOpened():
            raise RuntimeError("Could not create test video")
        for index in range(10):
            frame = np.full((48, 64, 3), index * 20, dtype=np.uint8)
            writer.write(frame)
        writer.release()

    def test_video_becomes_frames_and_database_observations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            video_path = root / "sample.avi"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                5.0,
                (64, 48),
            )
            self.assertTrue(writer.isOpened())
            for index in range(10):
                frame = np.full((48, 64, 3), index * 20, dtype=np.uint8)
                cv2.rectangle(frame, (index * 2, 10), (index * 2 + 12, 24), (0, 255, 0), -1)
                writer.write(frame)
            writer.release()

            pipeline = VideoPipeline(root / "data", sample_interval=0.5, yolo_model=None)
            result = pipeline.process(video_path)

            self.assertEqual(result["status"], "complete")
            self.assertGreaterEqual(result["sampled_frames"], 4)
            self.assertEqual(len(result["frames"]), result["sampled_frames"])
            kinds = {item["kind"] for item in result["frames"][0]["observations"]}
            self.assertTrue({"lighting", "focus", "dominant_color", "motion", "summary"} <= kinds)
            self.assertTrue(Path(result["frames"][0]["image_path"]).is_file())

    def test_rejects_unsupported_file_type(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            bad_file = root / "notes.txt"
            bad_file.write_text("not a video")
            pipeline = VideoPipeline(root / "data", yolo_model=None)
            with self.assertRaisesRegex(ValueError, "Unsupported video type"):
                pipeline.process(bad_file)

    def test_question_returns_latest_matching_object_with_bbox(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pipeline = VideoPipeline(root / "data", yolo_model=None)
            video_path = root / "data" / "uploads" / "video.mp4"
            video_path.touch()
            frame_path = root / "data" / "frames" / "frame.jpg"
            frame_path.parent.mkdir(parents=True, exist_ok=True)
            frame_path.touch()
            pipeline.database.create_video("video-1", "kitchen.mp4", video_path)
            pipeline.database.add_frame_with_observations(
                "video-1",
                45,
                1.5,
                frame_path,
                [
                    Observation(
                        "object",
                        "chair",
                        0.8732,
                        details={
                            "bbox_xywh": {
                                "x": 10.0,
                                "y": 20.0,
                                "width": 30.0,
                                "height": 40.0,
                            },
                            "coordinate_space": "pixels; top-left origin",
                        },
                    )
                ],
            )

            result = pipeline.answer_question("Where was the chair last seen?")

            self.assertTrue(result["found"])
            self.assertEqual(result["requested_class"], "chair")
            self.assertEqual(result["detections"][0]["class"], "chair")
            self.assertEqual(result["detections"][0]["confidence"], 0.8732)
            self.assertEqual(result["detections"][0]["bbox_xywh"]["width"], 30.0)
            self.assertIn("1.5s in kitchen.mp4", result["answer"])

    def test_object_observations_receive_code_generated_entity_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            video_path = root / "sample.avi"
            self.make_video(video_path)
            pipeline = VideoPipeline(
                root / "data",
                sample_interval=1.0,
                yolo_model=None,
            )

            class FakeTrackedDetector:
                class_names = ["chair"]

                def reset(self) -> None:
                    pass

                def observe(self, _frame: np.ndarray) -> list[Observation]:
                    return [
                        Observation(
                            "object",
                            "chair",
                            0.9,
                            details={
                                "track_id": 7,
                                "bbox_xywh": {
                                    "x": 10.0,
                                    "y": 10.0,
                                    "width": 20.0,
                                    "height": 20.0,
                                },
                            },
                        )
                    ]

                def embed_detection(
                    self, _frame: np.ndarray, _bbox: tuple[float, ...]
                ) -> tuple[float, ...]:
                    return (1.0, 0.0)

            pipeline.perception.object_detector = FakeTrackedDetector()

            result = pipeline.process(video_path)

            self.assertEqual(len(result["entities"]), 1)
            self.assertEqual(result["entities"][0]["label"], "chair")
            entity_ids = {
                observation["details"]["entity_id"]
                for frame in result["frames"]
                for observation in frame["observations"]
                if observation["kind"] == "object"
            }
            self.assertEqual(entity_ids, {result["entities"][0]["entity_id"]})

            reopened_database = MemoryDatabase(root / "data" / "memory.db")
            stored_entities = reopened_database.get_video_entities(result["id"])
            self.assertEqual(len(stored_entities), 1)
            self.assertEqual(
                stored_entities[0]["entity_id"],
                result["entities"][0]["entity_id"],
            )
            self.assertEqual(stored_entities[0]["last_seen"], 1.8)
            self.assertEqual(
                [event["event_type"] for event in result["entity_events"]],
                ["APPEARED", "DISAPPEARED"],
            )
            appeared_evidence = result["entity_events"][0]["evidence"][0]
            self.assertEqual(appeared_evidence["frame_number"], 2)
            self.assertEqual(appeared_evidence["timestamp_seconds"], 0.4)
            self.assertIsNone(appeared_evidence["frame_id"])
            self.assertIsNone(appeared_evidence["image_path"])
            self.assertEqual(result["tracking_summary"]["entity_count"], 1)
            self.assertEqual(result["tracking_summary"]["event_count"], 2)

    def test_tracked_movement_keeps_exact_frame_and_timestamp_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            video_path = root / "sample.avi"
            self.make_video(video_path)
            pipeline = VideoPipeline(
                root / "data",
                sample_interval=0.2,
                yolo_model=None,
            )

            class FakeMovingDetector:
                class_names = ["chair"]

                def __init__(self) -> None:
                    self.frame_number = 0

                def reset(self) -> None:
                    self.frame_number = 0

                def observe(self, _frame: np.ndarray) -> list[Observation]:
                    x = 5.0 if self.frame_number < 3 else 35.0
                    self.frame_number += 1
                    return [
                        Observation(
                            "object",
                            "chair",
                            0.9,
                            details={
                                "track_id": 7,
                                "bbox_xywh": {
                                    "x": x,
                                    "y": 10.0,
                                    "width": 10.0,
                                    "height": 10.0,
                                },
                            },
                        )
                    ]

                def embed_detection(
                    self, _frame: np.ndarray, _bbox: tuple[float, ...]
                ) -> tuple[float, ...]:
                    return (1.0, 0.0)

            pipeline.perception.object_detector = FakeMovingDetector()

            result = pipeline.process(video_path)

            moved_events = [
                event
                for event in result["entity_events"]
                if event["event_type"] == "MOVED"
            ]
            self.assertEqual(len(moved_events), 1)
            moved = moved_events[0]
            self.assertEqual(moved["source_zone"], "left")
            self.assertEqual(moved["destination_zone"], "center")
            self.assertEqual(moved["timestamp_seconds"], 0.6)
            self.assertEqual(
                moved["evidence"],
                [
                    {
                        "frame_id": result["frames"][3]["id"],
                        "frame_number": 3,
                        "timestamp_seconds": 0.6,
                        "image_path": result["frames"][3]["image_path"],
                    }
                ],
            )

            reopened_database = MemoryDatabase(root / "data" / "memory.db")
            self.assertEqual(
                reopened_database.get_entity_state_events(result["id"]),
                result["entity_events"],
            )

    def test_symbolic_events_are_persisted_after_perception(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            video_path = root / "sample.avi"
            self.make_video(video_path)
            pipeline = VideoPipeline(
                root / "data",
                sample_interval=1.0,
                yolo_model=None,
                event_extractor=FakeEventExtractor(),
            )

            result = pipeline.process(video_path)

            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["event_extraction"]["status"], "complete")
            self.assertEqual(result["symbolic_events"][0]["event_type"], "moved")
            self.assertEqual(
                result["symbolic_events"][0]["evidence"][0]["frame_id"],
                result["frames"][0]["id"],
            )

    def test_event_failure_does_not_fail_perception(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            video_path = root / "sample.avi"
            self.make_video(video_path)
            pipeline = VideoPipeline(
                root / "data",
                yolo_model=None,
                event_extractor=FakeEventExtractor(fails=True),
            )

            result = pipeline.process(video_path)

            self.assertEqual(result["status"], "complete")
            self.assertGreater(len(result["frames"]), 0)
            self.assertEqual(result["event_extraction"]["status"], "failed")
            self.assertIn("simulated Gemini failure", result["event_extraction"]["error"])
            self.assertEqual(result["symbolic_events"], [])


if __name__ == "__main__":
    unittest.main()
