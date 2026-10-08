"""Video ingestion pipeline: video -> frames -> perception -> observations."""

from __future__ import annotations

import re
import shutil
import threading
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2

from .database import MemoryDatabase
from .entities import EntityResolver
from .perception import Observation, PerceptionLayer

if TYPE_CHECKING:
    from .symbolic_events import GeminiSymbolicEventExtractor


SUPPORTED_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


class VideoPipeline:
    def __init__(
        self,
        data_dir: Path | str = "data",
        sample_interval: float = 1.0,
        yolo_model: Path | str | None = None,
        event_extractor: GeminiSymbolicEventExtractor | None = None,
    ):
        if sample_interval <= 0:
            raise ValueError("sample_interval must be greater than zero")
        self.data_dir = Path(data_dir).resolve()
        self.upload_dir = self.data_dir / "uploads"
        self.frame_dir = self.data_dir / "frames"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        self.database = MemoryDatabase(self.data_dir / "memory.db")
        self.sample_interval = sample_interval
        self.perception = PerceptionLayer(yolo_model)
        self.event_extractor = event_extractor
        # Perception compares consecutive frames and optional model runtimes may not
        # be thread-safe, so concurrent uploads are processed one at a time.
        self._processing_lock = threading.Lock()

    def upload_path(self, video_id: str, original_name: str) -> Path:
        suffix = Path(original_name).suffix.lower()
        if suffix not in SUPPORTED_VIDEO_SUFFIXES:
            allowed = ", ".join(sorted(SUPPORTED_VIDEO_SUFFIXES))
            raise ValueError(f"Unsupported video type. Use one of: {allowed}")
        return self.upload_dir / f"{video_id}{suffix}"

    def answer_question(self, question: str) -> dict[str, Any]:
        question = question.strip()
        if not question:
            raise ValueError("Enter a question about the observation archive")

        with self._processing_lock:
            requested_class = self._requested_object_class(question)
            frame = self.database.find_latest_object_frame(requested_class)
            if frame is None:
                subject = requested_class or "object"
                return {
                    "found": False,
                    "question": question,
                    "requested_class": requested_class,
                    "answer": f"No {subject} observation was found in the archive.",
                    "detections": [],
                }

            detections = self.database.get_frame_objects(
                frame["frame_id"], requested_class
            )
            needs_bbox = any(
                "bbox_xywh" not in detection["details"] for detection in detections
            )
            detector = self.perception.object_detector
            if needs_bbox and detector is not None:
                image = cv2.imread(frame["image_path"])
                if image is not None:
                    self.database.replace_frame_objects(
                        frame["frame_id"], detector.observe(image)
                    )
                    detections = self.database.get_frame_objects(
                        frame["frame_id"], requested_class
                    )

            if not detections:
                subject = requested_class or "object"
                return {
                    "found": False,
                    "question": question,
                    "requested_class": requested_class,
                    "answer": f"No {subject} observation was found in the archive.",
                    "detections": [],
                }

            for detection in detections:
                detection["bbox_xywh"] = detection["details"].get("bbox_xywh")
                detection["coordinate_space"] = detection["details"].get(
                    "coordinate_space"
                )

            primary = detections[0]
            confidence = primary["confidence"]
            confidence_text = (
                f"{confidence * 100:.1f}%" if confidence is not None else "unknown"
            )
            bbox = primary["bbox_xywh"]
            if bbox:
                bbox_text = (
                    f"x={bbox['x']}, y={bbox['y']}, width={bbox['width']}, "
                    f"height={bbox['height']} pixels"
                )
            else:
                bbox_text = "bounding box unavailable"
            answer = (
                f"The latest {primary['class']} observation was at "
                f"{frame['timestamp_seconds']:.1f}s in {frame['original_name']}. "
                f"Confidence: {confidence_text}. Bounding box: {bbox_text}."
            )
            return {
                "found": True,
                "question": question,
                "requested_class": requested_class,
                "answer": answer,
                "video_id": frame["video_id"],
                "video_name": frame["original_name"],
                "timestamp_seconds": frame["timestamp_seconds"],
                "frame_number": frame["frame_number"],
                "frame_url": (
                    f"/frames/{frame['video_id']}/{Path(frame['image_path']).name}"
                ),
                "detections": detections,
            }

    def _requested_object_class(self, question: str) -> str | None:
        labels = set(self.database.list_object_labels())
        detector = self.perception.object_detector
        if detector is not None:
            labels.update(detector.class_names)

        normalized_question = question.casefold()
        special_aliases = {
            "person": {"people", "persons", "someone"},
            "cell phone": {"phone", "mobile", "mobile phone"},
            "couch": {"sofa"},
            "tv": {"television"},
            "dining table": {"table"},
            "potted plant": {"plant"},
        }
        for label in sorted(labels, key=len, reverse=True):
            aliases = {label.casefold(), f"{label.casefold()}s"}
            if " " in label:
                aliases.add(label.casefold().rsplit(" ", 1)[-1])
            aliases.update(special_aliases.get(label.casefold(), set()))
            if any(
                re.search(rf"\b{re.escape(alias)}\b", normalized_question)
                for alias in aliases
            ):
                return label
        return None

    def process(
        self,
        source: Path | str,
        *,
        original_name: str | None = None,
        video_id: str | None = None,
        source_is_staged: bool = False,
    ) -> dict[str, Any]:
        with self._processing_lock:
            result = self._process_locked(
                source,
                original_name=original_name,
                video_id=video_id,
                source_is_staged=source_is_staged,
            )
            if self.event_extractor is not None:
                result = self._extract_symbolic_events(result["id"], result)
            return result

    def _extract_symbolic_events(
        self, video_id: str, perception_result: dict[str, Any]
    ) -> dict[str, Any]:
        assert self.event_extractor is not None
        extraction_id = uuid.uuid4().hex
        extraction_created = False
        try:
            self.database.create_event_extraction(
                extraction_id,
                video_id,
                provider=self.event_extractor.provider,
                model=self.event_extractor.model,
                schema_version=self.event_extractor.schema_version,
            )
            extraction_created = True
            video = self.database.get_video(video_id)
            if video is None:
                raise RuntimeError("Video disappeared before symbolic event extraction")
            extraction = self.event_extractor.extract(video)
            self.database.complete_event_extraction(
                extraction_id,
                extraction.events,
                rejected_event_count=extraction.rejected_event_count,
            )
        except Exception as exc:
            if extraction_created:
                self.database.fail_event_extraction(extraction_id, str(exc))
            else:
                perception_result["event_extraction"] = {
                    "status": "failed",
                    "error": str(exc),
                }

        refreshed = self.database.get_video(video_id)
        if refreshed is None:
            return perception_result
        refreshed["sampled_frames"] = len(refreshed["frames"])
        refreshed["tracking_summary"] = perception_result.get(
            "tracking_summary", {}
        )
        return refreshed

    @staticmethod
    def _resolve_object_entities(
        observations: list[Observation],
        frame: Any,
        frame_number: int,
        timestamp: float,
        resolver: EntityResolver,
        detector: Any,
    ) -> tuple[list[Observation], set[int]]:
        """Debounce tracked detections and attach persistent entity IDs."""

        resolved_observations: list[Observation] = []
        visible_track_ids: set[int] = set()
        for observation in observations:
            bbox_details = observation.details.get("bbox_xywh")
            track_id_value = observation.details.get("track_id")
            if (
                observation.kind != "object"
                or observation.confidence is None
                or not isinstance(bbox_details, dict)
                or track_id_value is None
            ):
                continue

            try:
                track_id = int(track_id_value)
                bbox = (
                    float(bbox_details["x"]),
                    float(bbox_details["y"]),
                    float(bbox_details["width"]),
                    float(bbox_details["height"]),
                )
            except (KeyError, TypeError, ValueError):
                continue

            visible_track_ids.add(track_id)
            embedding = None
            if resolver.needs_embedding(track_id):
                try:
                    embedding = detector.embed_detection(frame, bbox)
                except Exception:
                    embedding = None

            entity = resolver.resolve_track(
                track_id=track_id,
                label=observation.label,
                frame_number=frame_number,
                timestamp=timestamp,
                bbox=bbox,
                frame_width=int(frame.shape[1]),
                detection_confidence=observation.confidence,
                embedding=embedding,
            )
            if entity is None:
                continue
            details = dict(observation.details)
            details["entity_id"] = entity.entity_id
            details["identity_confidence"] = entity.identity_confidence
            details["zone"] = resolver.entity_zones.get(entity.entity_id)
            resolved_observations.append(
                Observation(
                    kind=observation.kind,
                    label=observation.label,
                    confidence=observation.confidence,
                    details=details,
                )
            )

        return resolved_observations, visible_track_ids

    @staticmethod
    def _print_tracking_summary(summary: dict[str, Any]) -> None:
        print("\nEntity tracking summary")
        print(f"Entities: {summary['entity_count']}")
        for label, count in sorted(summary["entities_by_class"].items()):
            print(f"  {label}: {count}")
        print(f"Events: {summary['event_count']}")
        for event in summary["events"]:
            movement = ""
            if event["source_zone"] or event["destination_zone"]:
                movement = (
                    f" ({event['source_zone'] or 'outside'} -> "
                    f"{event['destination_zone'] or 'outside'})"
                )
            print(
                f"  {event['timestamp']:.2f}s {event['event_type']} "
                f"{event['entity_id']}{movement} "
                f"confidence={event['confidence']:.3f}"
            )

    def _process_locked(
        self,
        source: Path | str,
        *,
        original_name: str | None,
        video_id: str | None,
        source_is_staged: bool,
    ) -> dict[str, Any]:
        source_path = Path(source).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Video not found: {source_path}")

        video_id = video_id or uuid.uuid4().hex
        original_name = Path(original_name or source_path.name).name
        stored_path = self.upload_path(video_id, original_name)
        if not source_is_staged:
            shutil.copy2(source_path, stored_path)
        elif source_path != stored_path.resolve():
            raise ValueError("Staged source path does not match the pipeline upload path")

        self.database.create_video(video_id, original_name, stored_path)
        frame_output_dir = self.frame_dir / video_id
        frame_output_dir.mkdir(parents=True, exist_ok=True)
        self.perception.reset()
        entity_resolver = EntityResolver(
            video_id,
            max_time_gap=max(2.5, self.sample_interval * 1.5),
        )

        capture = cv2.VideoCapture(str(stored_path))
        if not capture.isOpened():
            error = "OpenCV could not open this video"
            self.database.fail_video(video_id, error)
            raise ValueError(error)

        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if fps <= 0:
                fps = 30.0
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if total_frames > 0 else 0.0
            sample_every = max(1, round(fps * self.sample_interval))
            frame_number = 0
            sampled_frames = 0
            detector = self.perception.object_detector

            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp = frame_number / fps
                resolved_objects: list[Observation] = []
                if detector is not None:
                    tracked_objects = detector.observe(frame)
                    resolved_objects, visible_track_ids = self._resolve_object_entities(
                        tracked_objects,
                        frame,
                        frame_number,
                        timestamp,
                        entity_resolver,
                        detector,
                    )
                    entity_resolver.advance_frame(
                        frame_number=frame_number,
                        timestamp=timestamp,
                        visible_track_ids=visible_track_ids,
                    )
                if frame_number % sample_every == 0:
                    frame_path = frame_output_dir / f"frame_{frame_number:08d}.jpg"
                    if not cv2.imwrite(str(frame_path), frame):
                        raise OSError(f"Could not write extracted frame: {frame_path}")
                    observations = self.perception.observe(
                        frame,
                        object_observations=(
                            resolved_objects if detector is not None else None
                        ),
                    )
                    self.database.add_frame_with_observations(
                        video_id,
                        frame_number,
                        timestamp,
                        frame_path,
                        observations,
                    )
                    sampled_frames += 1
                frame_number += 1

            if frame_number == 0:
                raise ValueError("The uploaded file contains no readable video frames")

            actual_total = total_frames if total_frames > 0 else frame_number
            actual_duration = duration if duration > 0 else frame_number / fps
            entity_resolver.finalize(actual_duration)
            self.database.upsert_entities(entity_resolver.entities.values())
            self.database.replace_entity_state_events(
                video_id, entity_resolver.state_events
            )
            self.database.finish_video(
                video_id,
                duration_seconds=actual_duration,
                fps=fps,
                total_frames=actual_total,
            )
            result = self.database.get_video(video_id)
            assert result is not None
            result["sampled_frames"] = sampled_frames
            result["tracking_summary"] = entity_resolver.summary()
            self._print_tracking_summary(result["tracking_summary"])
            return result
        except Exception as exc:
            self.database.fail_video(video_id, str(exc))
            raise
        finally:
            capture.release()
