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


if __name__ == "__main__":
    unittest.main()
