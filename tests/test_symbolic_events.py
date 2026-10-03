from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from digital_hippocampus.symbolic_events import (
    GeminiEventBatch,
    GeminiEventCandidate,
    GeminiSymbolicEventExtractor,
    validate_and_resolve_events,
)


class FakeFiles:
    def __init__(self) -> None:
        self.deleted: list[str] = []
        self.upload_config: dict | None = None

    def upload(self, *, file: str, config: dict) -> SimpleNamespace:
        self.upload_config = config
        return SimpleNamespace(
            name="files/test-video",
            uri="https://example.test/video",
            mime_type=config["mime_type"],
            state=SimpleNamespace(name="ACTIVE"),
        )

    def get(self, *, name: str) -> SimpleNamespace:
        raise AssertionError(f"ACTIVE test file should not be polled: {name}")

    def delete(self, *, name: str) -> None:
        self.deleted.append(name)


class FakeInteractions:
    def __init__(self, output: dict) -> None:
        self.output = output
        self.request: dict | None = None

    def create(self, **request: object) -> SimpleNamespace:
        self.request = request
        return SimpleNamespace(output_text=json.dumps(self.output))


class FakeGeminiClient:
    def __init__(self, output: dict) -> None:
        self.files = FakeFiles()
        self.interactions = FakeInteractions(output)


class SymbolicEventValidationTest(unittest.TestCase):
    def test_one_evidence_timestamp_is_valid_and_maps_to_nearest_frame(self) -> None:
        batch = GeminiEventBatch(
            events=[
                GeminiEventCandidate(
                    event_type="picked_up",
                    start_time=1.1,
                    end_time=1.4,
                    actor="person",
                    object="cup",
                    source_location="table",
                    destination_location=None,
                    confidence=0.88,
                    evidence_timestamps=[1.25],
                )
            ]
        )
        frames = [
            {"id": 10, "frame_number": 30, "timestamp_seconds": 1.0},
            {"id": 11, "frame_number": 60, "timestamp_seconds": 2.0},
        ]

        result = validate_and_resolve_events(
            batch,
            video_id="video-1",
            duration_seconds=3.0,
            frames=frames,
        )

        self.assertEqual(result.rejected_event_count, 0)
        self.assertEqual(len(result.events), 1)
        self.assertEqual(len(result.events[0].evidence), 1)
        self.assertEqual(result.events[0].evidence[0].frame_id, 10)
        self.assertEqual(result.events[0].evidence[0].timestamp_seconds, 1.25)

    def test_event_with_out_of_range_evidence_is_rejected(self) -> None:
        batch = GeminiEventBatch(
            events=[
                GeminiEventCandidate(
                    event_type="moved",
                    start_time=1.0,
                    end_time=2.0,
                    actor=None,
                    object="chair",
                    source_location=None,
                    destination_location=None,
                    confidence=0.6,
                    evidence_timestamps=[4.0],
                )
            ]
        )

        result = validate_and_resolve_events(
            batch,
            video_id="video-1",
            duration_seconds=3.0,
            frames=[],
        )

        self.assertEqual(result.events, [])
        self.assertEqual(result.rejected_event_count, 1)

    def test_short_event_remains_valid_without_a_local_frame_match(self) -> None:
        batch = GeminiEventBatch(
            events=[
                GeminiEventCandidate(
                    event_type="closed",
                    start_time=0.2,
                    end_time=0.35,
                    actor="person",
                    object="drawer",
                    source_location=None,
                    destination_location=None,
                    confidence=0.8,
                    evidence_timestamps=[0.3],
                )
            ]
        )

        result = validate_and_resolve_events(
            batch,
            video_id="video-1",
            duration_seconds=2.0,
            frames=[],
        )

        self.assertEqual(result.rejected_event_count, 0)
        self.assertEqual(len(result.events), 1)
        evidence = result.events[0].evidence[0]
        self.assertEqual(evidence.timestamp_seconds, 0.3)
        self.assertIsNone(evidence.frame_id)
        self.assertIsNone(evidence.frame_timestamp_seconds)

    def test_gemini_request_uses_video_schema_and_deletes_remote_file(self) -> None:
        output = {
            "events": [
                {
                    "event_type": "opened",
                    "start_time": 0.5,
                    "end_time": 1.0,
                    "actor": "person",
                    "object": "door",
                    "source_location": None,
                    "destination_location": None,
                    "confidence": 0.9,
                    "evidence_timestamps": [0.8],
                }
            ]
        }
        client = FakeGeminiClient(output)
        extractor = GeminiSymbolicEventExtractor(client=client, model="test-model")

        with tempfile.TemporaryDirectory() as temporary_directory:
            video_path = Path(temporary_directory) / "clip.mp4"
            video_path.touch()
            batch = extractor.extract_candidates(
                {
                    "id": "video-1",
                    "stored_path": str(video_path),
                    "duration_seconds": 2.0,
                    "frames": [],
                }
            )

        self.assertEqual(batch.events[0].event_type, "opened")
        self.assertEqual(client.files.deleted, ["files/test-video"])
        request = client.interactions.request
        assert request is not None
        self.assertEqual(request["model"], "test-model")
        self.assertEqual(request["input"][0]["processing"]["fps"], 1.0)
        self.assertEqual(request["response_format"]["mime_type"], "application/json")
        self.assertFalse(request["store"])


if __name__ == "__main__":
    unittest.main()
