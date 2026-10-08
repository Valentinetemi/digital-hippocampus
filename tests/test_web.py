from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from digital_hippocampus.pipeline import VideoPipeline
from digital_hippocampus.perception import Observation
from digital_hippocampus.server import VideoRequestHandler


class FakeLiveService:
    def __init__(self) -> None:
        self.start_arguments: dict | None = None

    def status(self) -> dict:
        return {"running": False, "alexa_simulation": True}

    def start(self, **kwargs: object) -> dict:
        self.start_arguments = dict(kwargs)
        return {"running": True, "alexa_simulation": True}

    def stop(self) -> dict:
        return {"running": False, "alexa_simulation": True}

    def respond(self, text: str) -> dict:
        return {"handled": True, "intent": "completed", "text": text}


class UploadHandlerTest(unittest.TestCase):
    def test_raw_video_upload_runs_the_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            video_path = root / "upload.avi"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                4.0,
                (48, 32),
            )
            self.assertTrue(writer.isOpened())
            for value in (20, 80, 140, 200):
                writer.write(np.full((32, 48, 3), value, dtype=np.uint8))
            writer.release()
            video_bytes = video_path.read_bytes()

            handler = VideoRequestHandler.__new__(VideoRequestHandler)
            handler.pipeline = VideoPipeline(
                root / "data", sample_interval=0.5, yolo_model=None
            )
            handler.path = "/api/videos"
            handler.headers = {
                "Content-Length": str(len(video_bytes)),
                "X-Filename": "upload.avi",
            }
            handler.rfile = io.BytesIO(video_bytes)
            handler.wfile = io.BytesIO()
            statuses: list[int] = []
            handler.send_response = lambda status: statuses.append(int(status))
            handler.send_header = lambda *_args: None
            handler.end_headers = lambda: None

            handler.do_POST()

            payload = json.loads(handler.wfile.getvalue())
            self.assertEqual(statuses, [201])
            self.assertEqual(payload["status"], "complete")
            self.assertEqual(payload["original_name"], "upload.avi")
            self.assertGreaterEqual(payload["sampled_frames"], 2)

    def test_question_endpoint_returns_detection_details(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pipeline = VideoPipeline(root / "data", yolo_model=None)
            video_path = root / "video.mp4"
            frame_path = root / "frame.jpg"
            video_path.touch()
            frame_path.touch()
            pipeline.database.create_video("video1", "room.mp4", video_path)
            pipeline.database.add_frame_with_observations(
                "video1",
                60,
                2.0,
                frame_path,
                [
                    Observation(
                        "object",
                        "person",
                        0.91,
                        {"bbox_xywh": {"x": 5, "y": 6, "width": 70, "height": 80}},
                    )
                ],
            )
            request_body = json.dumps(
                {"question": "Where was the person last seen?"}
            ).encode()

            handler = VideoRequestHandler.__new__(VideoRequestHandler)
            handler.pipeline = pipeline
            handler.path = "/api/questions"
            handler.headers = {"Content-Length": str(len(request_body))}
            handler.rfile = io.BytesIO(request_body)
            handler.wfile = io.BytesIO()
            statuses: list[int] = []
            handler.send_response = lambda status: statuses.append(int(status))
            handler.send_header = lambda *_args: None
            handler.end_headers = lambda: None

            handler.do_POST()

            payload = json.loads(handler.wfile.getvalue())
            self.assertEqual(statuses, [200])
            self.assertTrue(payload["found"])
            self.assertEqual(payload["detections"][0]["class"], "person")
            self.assertEqual(
                payload["detections"][0]["bbox_xywh"],
                {"x": 5, "y": 6, "width": 70, "height": 80},
            )

    def test_live_start_endpoint_applies_demo_timer(self) -> None:
        payload = json.dumps(
            {
                "checkin_after_seconds": 10,
                "departure_confirm_seconds": 3,
                "snooze_seconds": 60,
                "cooldown_seconds": 60,
            }
        ).encode()
        live_service = FakeLiveService()
        handler = VideoRequestHandler.__new__(VideoRequestHandler)
        handler.live_service = live_service
        handler.path = "/api/live/start"
        handler.headers = {"Content-Length": str(len(payload))}
        handler.rfile = io.BytesIO(payload)
        handler.wfile = io.BytesIO()
        statuses: list[int] = []
        handler.send_response = lambda status: statuses.append(int(status))
        handler.send_header = lambda *_args: None
        handler.end_headers = lambda: None

        handler.do_POST()

        response = json.loads(handler.wfile.getvalue())
        self.assertEqual(statuses, [201])
        self.assertTrue(response["running"])
        assert live_service.start_arguments is not None
        self.assertEqual(
            live_service.start_arguments["checkin_after_seconds"], 10.0
        )

    def test_spoken_or_typed_response_uses_live_episode(self) -> None:
        payload = json.dumps({"text": "I'm done"}).encode()
        handler = VideoRequestHandler.__new__(VideoRequestHandler)
        handler.live_service = FakeLiveService()
        handler.path = "/api/live/respond"
        handler.headers = {"Content-Length": str(len(payload))}
        handler.rfile = io.BytesIO(payload)
        handler.wfile = io.BytesIO()
        statuses: list[int] = []
        handler.send_response = lambda status: statuses.append(int(status))
        handler.send_header = lambda *_args: None
        handler.end_headers = lambda: None

        handler.do_POST()

        response = json.loads(handler.wfile.getvalue())
        self.assertEqual(statuses, [200])
        self.assertEqual(response["intent"], "completed")
        self.assertEqual(response["text"], "I'm done")


if __name__ == "__main__":
    unittest.main()
