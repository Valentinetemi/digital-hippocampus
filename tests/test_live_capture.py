from __future__ import annotations

import queue
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from digital_hippocampus.database import MemoryDatabase
from digital_hippocampus.entities import EntityResolver
from digital_hippocampus.live_capture import (
    CapturedFrame,
    LiveCameraService,
    parse_camera_source,
)
from digital_hippocampus.live_memory import CameraState
from digital_hippocampus.perception import Observation


class FakeCapture:
    def __init__(self, _source: str | int) -> None:
        self.frames = [
            np.full((48, 64, 3), value, dtype=np.uint8)
            for value in (20, 50, 80, 110, 140, 170)
        ]
        self.released = False

    def isOpened(self) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        time.sleep(0.01)
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self) -> None:
        self.released = True


class FakePerception:
    def reset(self) -> None:
        pass

    def observe(self, _frame: np.ndarray) -> list[Observation]:
        return [
            Observation(
                "motion",
                "moderate",
                details={"normalized_frame_difference": 0.08},
            ),
            Observation(
                "object",
                "person",
                0.9,
                {"bbox_normalized_xywh": {"x": 0, "y": 0, "width": 1, "height": 1}},
            ),
            Observation(
                "object",
                "cup",
                0.88,
                {"bbox_normalized_xywh": {"x": 0.1, "y": 0.5, "width": 0.1, "height": 0.1}},
            ),
            Observation(
                "object",
                "bottle",
                0.86,
                {"bbox_normalized_xywh": {"x": 0.3, "y": 0.3, "width": 0.1, "height": 0.3}},
            ),
        ]


class LiveCameraServiceTest(unittest.TestCase):
    def make_service(self, root: Path, **kwargs: object) -> LiveCameraService:
        database = MemoryDatabase(root / "memory.db")
        return LiveCameraService(
            database,
            root,
            yolo_model=None,
            capture_factory=FakeCapture,
            perception_factory=FakePerception,
            **kwargs,
        )

    def test_camera_source_parses_device_index_or_keeps_url(self) -> None:
        self.assertEqual(parse_camera_source("0"), 0)
        self.assertEqual(parse_camera_source(2), 2)
        self.assertEqual(
            parse_camera_source("rtsp://camera.local/live"),
            "rtsp://camera.local/live",
        )

    def test_bounded_queue_discards_the_oldest_stale_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            service = self.make_service(Path(temporary_directory), queue_size=2)
            now = datetime.now(timezone.utc)
            for sequence in (1, 2, 3):
                service._offer_frame(
                    CapturedFrame(sequence, now, np.zeros((2, 2, 3), dtype=np.uint8))
                )

            self.assertEqual(service._frames.qsize(), 2)
            self.assertEqual(service._dropped_frames, 1)
            self.assertEqual(service._frames.get_nowait().sequence, 2)
            self.assertEqual(service._frames.get_nowait().sequence, 3)
            with self.assertRaises(queue.Empty):
                service._frames.get_nowait()

    def test_occluded_frames_are_not_treated_as_usable_camera_evidence(self) -> None:
        black = np.zeros((40, 40, 3), dtype=np.uint8)
        checkerboard = np.indices((40, 40)).sum(axis=0) % 2 * 255
        visible = np.repeat(checkerboard[:, :, None], 3, axis=2).astype(np.uint8)

        self.assertEqual(
            LiveCameraService.classify_camera_state(black),
            (CameraState.OCCLUDED, False),
        )
        self.assertEqual(
            LiveCameraService.classify_camera_state(visible),
            (CameraState.AVAILABLE, True),
        )

    def test_observations_become_timestamped_facts(self) -> None:
        observed_at = datetime.now(timezone.utc)
        evidence = LiveCameraService.observations_to_evidence(
            observed_at,
            CameraState.AVAILABLE,
            True,
            FakePerception().observe(np.zeros((10, 10, 3), dtype=np.uint8)),
            Path("evidence.jpg"),
        )

        self.assertTrue(evidence.person_present)
        self.assertEqual(evidence.motion_score, 0.08)
        self.assertEqual(
            {item.label for item in evidence.objects},
            {"person", "cup", "bottle"},
        )

    def test_live_tracks_receive_persistent_entity_ids(self) -> None:
        resolver = EntityResolver("session", min_consecutive_frames=2)
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        observation = Observation(
            "object",
            "cup",
            0.9,
            {
                "track_id": 7,
                "bbox_xywh": {"x": 10, "y": 20, "width": 30, "height": 40},
                "bbox_normalized_xywh": {
                    "x": 0.05,
                    "y": 0.2,
                    "width": 0.15,
                    "height": 0.4,
                },
            },
        )

        first, _ = LiveCameraService._resolve_live_entities(
            [observation], frame, 1, 0.0, resolver
        )
        second, _ = LiveCameraService._resolve_live_entities(
            [observation], frame, 2, 0.75, resolver
        )
        evidence = LiveCameraService.observations_to_evidence(
            datetime.now(timezone.utc),
            CameraState.AVAILABLE,
            True,
            second,
            Path("evidence.jpg"),
        )

        self.assertEqual(first, [])
        self.assertEqual(second[0].details["entity_id"], "session:cup-1")
        self.assertEqual(evidence.objects[0].entity_id, "session:cup-1")
        self.assertEqual(evidence.objects[0].detector_track_id, 7)

    def test_capture_and_perception_run_on_separate_workers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            service = self.make_service(
                Path(temporary_directory),
                perception_interval_seconds=0.01,
                capture_fps=100,
            )
            started = service.start(
                checkin_after_seconds=10,
                departure_confirm_seconds=2,
            )
            self.assertTrue(started["running"])

            deadline = time.monotonic() + 2
            while service.status()["processed_frames"] < 2 and time.monotonic() < deadline:
                time.sleep(0.02)
            stopped = service.stop()

            self.assertFalse(stopped["running"])
            self.assertGreaterEqual(stopped["captured_frames"], 2)
            self.assertGreaterEqual(stopped["processed_frames"], 2)
            session_id = stopped["session_id"]
            with service.database.connect() as connection:
                session = connection.execute(
                    "SELECT status FROM live_sessions WHERE id = ?", (session_id,)
                ).fetchone()
                observations = connection.execute(
                    "SELECT COUNT(*) FROM live_observations WHERE session_id = ?",
                    (session_id,),
                ).fetchone()[0]
            self.assertEqual(session["status"], "stopped")
            self.assertGreaterEqual(observations, 2)


if __name__ == "__main__":
    unittest.main()
