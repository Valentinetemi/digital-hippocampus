"""Responsive live-camera capture and asynchronous temporal perception."""

from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .database import MemoryDatabase
from .live_memory import (
    CameraState,
    EpisodeCoordinator,
    EpisodePolicyConfig,
    FrameEvidence,
    ObjectFact,
    to_iso,
    utc_now,
)
from .perception import Observation, PerceptionLayer
from .notifications import CaregiverPolicyConfig, LocalCaregiverPreview


@dataclass(frozen=True)
class CapturedFrame:
    sequence: int
    captured_at: datetime
    frame: np.ndarray | None
    camera_state: CameraState = CameraState.AVAILABLE


def parse_camera_source(value: str | int) -> str | int:
    """Map a numeric CLI value to an OpenCV device index; keep URLs/paths."""

    if isinstance(value, int):
        return value
    normalized = value.strip()
    if normalized.lstrip("-").isdigit():
        return int(normalized)
    return normalized


class LiveCameraService:
    """Capture continuously while a bounded worker processes only recent frames."""

    def __init__(
        self,
        database: MemoryDatabase,
        data_dir: Path | str,
        *,
        camera_source: str | int = 0,
        yolo_model: Path | str | None = None,
        person_name: str = "Temi",
        perception_interval_seconds: float = 0.75,
        capture_fps: float = 12.0,
        queue_size: int = 2,
        capture_factory: Callable[[str | int], Any] | None = None,
        perception_factory: Callable[[], PerceptionLayer] | None = None,
    ) -> None:
        if perception_interval_seconds <= 0 or capture_fps <= 0:
            raise ValueError("Capture and perception timing must be greater than zero")
        if queue_size <= 0:
            raise ValueError("queue_size must be greater than zero")
        self.database = database
        self.data_dir = Path(data_dir).resolve()
        self.live_frame_dir = self.data_dir / "live_frames"
        self.live_frame_dir.mkdir(parents=True, exist_ok=True)
        self.camera_source = parse_camera_source(camera_source)
        self.yolo_model = yolo_model
        self.person_name = person_name
        self.perception_interval_seconds = perception_interval_seconds
        self.capture_fps = capture_fps
        self._capture_factory = capture_factory or cv2.VideoCapture
        self._perception_factory = perception_factory or (
            lambda: PerceptionLayer(self.yolo_model)
        )
        self._frames: queue.Queue[CapturedFrame] = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._state_lock = threading.RLock()
        self._frame_ready = threading.Condition(self._state_lock)
        self._capture_thread: threading.Thread | None = None
        self._perception_thread: threading.Thread | None = None
        self._coordinator: EpisodeCoordinator | None = None
        self._session_id: str | None = None
        self._running = False
        self._camera_state = CameraState.UNAVAILABLE
        self._error: str | None = None
        self._started_at: datetime | None = None
        self._latest_jpeg: bytes | None = None
        self._frame_version = 0
        self._dropped_frames = 0
        self._captured_frames = 0
        self._processed_frames = 0
        self._last_perception_ms: float | None = None
        self._last_result: dict[str, Any] | None = None
        self._caregiver_preview = LocalCaregiverPreview(database)

    @property
    def running(self) -> bool:
        with self._state_lock:
            return self._running

    @property
    def session_id(self) -> str | None:
        with self._state_lock:
            return self._session_id

    def start(
        self,
        *,
        checkin_after_seconds: float = 15.0,
        departure_confirm_seconds: float = 3.0,
        snooze_seconds: float = 60.0,
        cooldown_seconds: float = 60.0,
        caregiver_preview_enabled: bool = False,
        caregiver_recipient: str | None = None,
        caregiver_escalation_delay_seconds: float = 120.0,
        caregiver_notification_limit: int = 1,
    ) -> dict[str, Any]:
        config = EpisodePolicyConfig(
            departure_confirm_seconds=departure_confirm_seconds,
            checkin_after_seconds=checkin_after_seconds,
            snooze_seconds=snooze_seconds,
            cooldown_seconds=cooldown_seconds,
        )
        caregiver_config = CaregiverPolicyConfig(
            enabled=caregiver_preview_enabled,
            recipient_label=caregiver_recipient,
            escalation_delay_seconds=caregiver_escalation_delay_seconds,
            notification_limit=caregiver_notification_limit,
            external_delivery_enabled=False,
        )
        with self._state_lock:
            if self._running:
                return self.status()
            self._stop.clear()
            self._drain_queue()
            self._session_id = uuid.uuid4().hex
            self._started_at = utc_now()
            self._running = True
            self._camera_state = CameraState.UNAVAILABLE
            self._error = None
            self._latest_jpeg = None
            self._frame_version += 1
            self._dropped_frames = 0
            self._captured_frames = 0
            self._processed_frames = 0
            self._last_perception_ms = None
            self._last_result = None
            self._coordinator = EpisodeCoordinator(
                self.database,
                person_name=self.person_name,
                config=config,
            )
            self._caregiver_preview = LocalCaregiverPreview(
                self.database, caregiver_config
            )
            self.database.create_live_session(
                self._session_id,
                str(self.camera_source),
                to_iso(self._started_at),
            )
            self._capture_thread = threading.Thread(
                target=self._capture_loop,
                name="digital-hippocampus-capture",
                daemon=True,
            )
            self._perception_thread = threading.Thread(
                target=self._perception_loop,
                name="digital-hippocampus-perception",
                daemon=True,
            )
            self._capture_thread.start()
            self._perception_thread.start()
        return self.status()

    def stop(self) -> dict[str, Any]:
        with self._state_lock:
            if not self._running:
                return self.status()
            self._stop.set()
            capture_thread = self._capture_thread
            perception_thread = self._perception_thread
            session_id = self._session_id
        for thread in (capture_thread, perception_thread):
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=3.0)
        with self._state_lock:
            self._running = False
            self._camera_state = CameraState.UNAVAILABLE
            self._frame_ready.notify_all()
            if session_id is not None:
                self.database.finish_live_session(
                    session_id,
                    to_iso(utc_now()),
                    error=self._error,
                )
        return self.status()

    def respond(self, text: str) -> dict[str, Any]:
        with self._state_lock:
            coordinator = self._coordinator
        if coordinator is None:
            return {"handled": False, "intent": "not_running", "episode": None}
        result = coordinator.respond(text)
        return {**result, "status": self.status()}

    def status(self) -> dict[str, Any]:
        with self._state_lock:
            session_id = self._session_id
            result = {
                "running": self._running,
                "session_id": session_id,
                "camera_source": str(self.camera_source),
                "camera_state": self._camera_state.value,
                "error": self._error,
                "started_at": to_iso(self._started_at) if self._started_at else None,
                "queue_depth": self._frames.qsize(),
                "queue_capacity": self._frames.maxsize,
                "dropped_frames": self._dropped_frames,
                "captured_frames": self._captured_frames,
                "processed_frames": self._processed_frames,
                "last_perception_ms": self._last_perception_ms,
                "has_frame": self._latest_jpeg is not None,
            }
        episodes = self.database.list_activity_episodes(limit=1)
        episode = episodes[0] if episodes else None
        timeline: list[dict[str, Any]] = []
        evidence: list[dict[str, Any]] = []
        checkin = None
        if episode is not None:
            timeline = self.database.get_episode_timeline(episode["id"])
            evidence = self.database.get_episode_evidence(episode["id"])[-12:]
            for item in evidence:
                if item.get("frame_path"):
                    item["frame_url"] = (
                        f"/live-frames/{item['session_id']}/"
                        f"{Path(item['frame_path']).name}"
                    )
            checkin = self.database.get_latest_episode_checkin(episode["id"])
        result.update(
            {
                "episode": episode,
                "timeline": timeline,
                "evidence": evidence,
                "checkin": checkin,
                "caregiver_preview": {
                    "enabled": self._caregiver_preview.config.enabled,
                    "adapter": self._caregiver_preview.adapter_name,
                    "external_delivery": False,
                    "recipient_label": self._caregiver_preview.config.recipient_label,
                    "escalation_delay_seconds": (
                        self._caregiver_preview.config.escalation_delay_seconds
                    ),
                    "notification_limit": (
                        self._caregiver_preview.config.notification_limit
                    ),
                },
                "caregiver_outbox": (
                    self.database.list_caregiver_previews(episode["id"])
                    if episode is not None
                    else []
                ),
                "alexa_simulation": True,
            }
        )
        return result

    def wait_for_jpeg(
        self, version: int, *, timeout: float = 1.0
    ) -> tuple[int, bytes | None]:
        with self._frame_ready:
            if version == self._frame_version:
                self._frame_ready.wait(timeout)
            return self._frame_version, self._latest_jpeg

    def _capture_loop(self) -> None:
        capture = self._capture_factory(self.camera_source)
        session_id = self.session_id
        if session_id is None:
            return
        try:
            if not capture.isOpened():
                self._set_error(
                    f"OpenCV could not open camera source {self.camera_source!r}"
                )
                self._offer_frame(
                    CapturedFrame(0, utc_now(), None, CameraState.UNAVAILABLE)
                )
                return

            minimum_period = 1.0 / self.capture_fps
            sequence = 0
            last_offered = 0.0
            consecutive_failures = 0
            while not self._stop.is_set():
                ok, frame = capture.read()
                now_monotonic = time.monotonic()
                if not ok or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures == 1 or consecutive_failures % 10 == 0:
                        self._offer_frame(
                            CapturedFrame(
                                sequence,
                                utc_now(),
                                None,
                                CameraState.UNAVAILABLE,
                            )
                        )
                    self._stop.wait(0.1)
                    continue
                consecutive_failures = 0
                if now_monotonic - last_offered < minimum_period:
                    continue
                last_offered = now_monotonic
                captured_at = utc_now()
                sequence += 1
                encoded, jpeg = cv2.imencode(
                    ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82]
                )
                with self._frame_ready:
                    self._camera_state = CameraState.AVAILABLE
                    self._captured_frames += 1
                    if encoded:
                        self._latest_jpeg = jpeg.tobytes()
                        self._frame_version += 1
                        self._frame_ready.notify_all()
                self._offer_frame(CapturedFrame(sequence, captured_at, frame))
                self.database.update_live_session_frame(
                    session_id,
                    to_iso(captured_at),
                    dropped_frames=self._dropped_frames,
                )
        except Exception as exc:
            self._set_error(f"Camera capture failed: {exc}")
            self._offer_frame(
                CapturedFrame(0, utc_now(), None, CameraState.UNAVAILABLE)
            )
        finally:
            capture.release()

    def _perception_loop(self) -> None:
        session_id = self.session_id
        coordinator = self._coordinator
        if session_id is None or coordinator is None:
            return
        try:
            perception = self._perception_factory()
            perception.reset()
        except Exception as exc:
            self._set_error(f"Could not initialize perception: {exc}")
            return

        last_processed = 0.0
        output_dir = self.live_frame_dir / session_id
        output_dir.mkdir(parents=True, exist_ok=True)
        while not self._stop.is_set():
            try:
                captured = self._frames.get(timeout=0.5)
            except queue.Empty:
                continue
            captured = self._take_latest(captured)
            if captured.frame is None:
                evidence = FrameEvidence(
                    observed_at=captured.captured_at,
                    camera_state=CameraState.UNAVAILABLE,
                    scene_clear=False,
                    person_present=None,
                    facts={"capture_error": "No readable frame was available."},
                )
                with self._state_lock:
                    self._camera_state = CameraState.UNAVAILABLE
                    self._last_result = coordinator.observe(session_id, evidence)
                self._evaluate_caregiver(self._last_result, evidence)
                continue

            since_last = time.monotonic() - last_processed
            if since_last < self.perception_interval_seconds:
                if self._stop.wait(self.perception_interval_seconds - since_last):
                    break
                captured = self._take_latest(captured)
                if captured.frame is None:
                    continue

            started = time.perf_counter()
            camera_state, scene_clear = self.classify_camera_state(captured.frame)
            try:
                observations = perception.observe(captured.frame)
                frame_path = output_dir / f"frame_{captured.sequence:010d}.jpg"
                cv2.imwrite(str(frame_path), captured.frame)
                evidence = self.observations_to_evidence(
                    captured.captured_at,
                    camera_state,
                    scene_clear,
                    observations,
                    frame_path,
                )
                result = coordinator.observe(session_id, evidence)
                self._evaluate_caregiver(result, evidence)
                elapsed_ms = (time.perf_counter() - started) * 1000
                with self._state_lock:
                    self._camera_state = camera_state
                    self._processed_frames += 1
                    self._last_perception_ms = round(elapsed_ms, 1)
                    self._last_result = result
            except Exception as exc:
                self._set_error(f"Perception failed: {exc}")
            last_processed = time.monotonic()

    def _offer_frame(self, captured: CapturedFrame) -> None:
        try:
            self._frames.put_nowait(captured)
            return
        except queue.Full:
            pass
        try:
            self._frames.get_nowait()
            with self._state_lock:
                self._dropped_frames += 1
        except queue.Empty:
            pass
        try:
            self._frames.put_nowait(captured)
        except queue.Full:
            with self._state_lock:
                self._dropped_frames += 1

    def _take_latest(self, initial: CapturedFrame) -> CapturedFrame:
        latest = initial
        while True:
            try:
                latest = self._frames.get_nowait()
                with self._state_lock:
                    self._dropped_frames += 1
            except queue.Empty:
                return latest

    def _drain_queue(self) -> None:
        while True:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                return

    def _set_error(self, message: str) -> None:
        with self._state_lock:
            self._error = message
            self._camera_state = CameraState.UNAVAILABLE

    def _evaluate_caregiver(
        self, result: dict[str, Any] | None, evidence: FrameEvidence
    ) -> dict[str, Any] | None:
        if not result or result.get("episode") is None:
            return None
        episode = result["episode"]
        checkin = self.database.get_latest_episode_checkin(episode["id"])
        return self._caregiver_preview.evaluate(
            episode=episode,
            checkin=checkin,
            observed_at=evidence.observed_at,
            person_present=evidence.person_present,
            camera_state=evidence.camera_state,
            scene_clear=evidence.scene_clear,
        )

    @staticmethod
    def classify_camera_state(frame: np.ndarray) -> tuple[CameraState, bool]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean = float(gray.mean())
        standard_deviation = float(gray.std())
        occluded = mean < 7.0 or standard_deviation < 2.5
        if occluded:
            return CameraState.OCCLUDED, False
        return CameraState.AVAILABLE, True

    @staticmethod
    def observations_to_evidence(
        observed_at: datetime,
        camera_state: CameraState,
        scene_clear: bool,
        observations: list[Observation],
        frame_path: Path,
    ) -> FrameEvidence:
        objects: list[ObjectFact] = []
        motion_score = 0.0
        summaries: dict[str, Any] = {}
        for observation in observations:
            if observation.kind == "motion":
                motion_score = float(
                    observation.details.get("normalized_frame_difference", 0.0)
                )
            if observation.kind in {"lighting", "focus", "summary"}:
                summaries[observation.kind] = {
                    "label": observation.label,
                    "details": observation.details,
                }
            if observation.kind != "object" or observation.confidence is None:
                continue
            bbox = observation.details.get("bbox_normalized_xywh")
            normalized_bbox = None
            if isinstance(bbox, dict):
                try:
                    normalized_bbox = (
                        float(bbox["x"]),
                        float(bbox["y"]),
                        float(bbox["width"]),
                        float(bbox["height"]),
                    )
                except (KeyError, TypeError, ValueError):
                    normalized_bbox = None
            objects.append(
                ObjectFact(
                    observation.label,
                    observation.confidence,
                    normalized_bbox,
                )
            )

        person_present = None
        if camera_state is CameraState.AVAILABLE and scene_clear:
            person_present = any(
                item.label.casefold() == "person" for item in objects
            )
        return FrameEvidence(
            observed_at=observed_at.astimezone(timezone.utc),
            camera_state=camera_state,
            scene_clear=scene_clear,
            person_present=person_present,
            motion_score=motion_score,
            objects=tuple(objects),
            frame_path=frame_path,
            facts={"visual_signals": summaries},
        )
