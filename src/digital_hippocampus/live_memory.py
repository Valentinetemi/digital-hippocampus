"""Temporal activity inference and persistent live-episode policy.

The models in this module deliberately separate camera facts (objects, motion,
person presence, camera health) from activity inferences.  The policy only
advances absence timers while the camera is available and the scene is clear.
"""

from __future__ import annotations

import math
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from typing import Any, Iterable

from .database import MemoryDatabase


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def to_iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds")


def from_iso(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


class CameraState(StrEnum):
    AVAILABLE = "available"
    OCCLUDED = "occluded"
    UNAVAILABLE = "unavailable"


class EpisodeState(StrEnum):
    ONGOING = "ongoing"
    POSSIBLY_INTERRUPTED = "possibly_interrupted"
    AWAITING_RESPONSE = "awaiting_response"
    SNOOZED = "snoozed"
    RESUMED = "resumed"
    COMPLETED = "completed"
    DISMISSED = "dismissed"


@dataclass(frozen=True)
class ObjectFact:
    label: str
    confidence: float
    bbox_normalized_xywh: tuple[float, float, float, float] | None = None
    entity_id: str | None = None
    detector_track_id: int | None = None

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
        }
        if self.bbox_normalized_xywh is not None:
            x, y, width, height = self.bbox_normalized_xywh
            result["bbox_normalized_xywh"] = {
                "x": x,
                "y": y,
                "width": width,
                "height": height,
            }
        if self.entity_id is not None:
            result["entity_id"] = self.entity_id
        if self.detector_track_id is not None:
            result["detector_track_id"] = self.detector_track_id
        return result


@dataclass(frozen=True)
class FrameEvidence:
    observed_at: datetime
    camera_state: CameraState
    scene_clear: bool
    person_present: bool | None
    motion_score: float = 0.0
    objects: tuple[ObjectFact, ...] = ()
    frame_path: Any | None = None
    facts: dict[str, Any] = field(default_factory=dict)

    @property
    def usable_for_departure(self) -> bool:
        return (
            self.camera_state is CameraState.AVAILABLE
            and self.scene_clear
            and self.person_present is not None
        )


@dataclass(frozen=True)
class ActivityInference:
    kind: str
    confidence: float
    reason: str
    evidence_observation_ids: tuple[int, ...]


@dataclass(frozen=True)
class EpisodePolicyConfig:
    departure_confirm_seconds: float = 3.0
    checkin_after_seconds: float = 15.0
    snooze_seconds: float = 60.0
    cooldown_seconds: float = 60.0

    def __post_init__(self) -> None:
        values = (
            self.departure_confirm_seconds,
            self.checkin_after_seconds,
            self.snooze_seconds,
            self.cooldown_seconds,
        )
        if any(value <= 0 for value in values):
            raise ValueError("Episode policy timings must be greater than zero")
        if self.checkin_after_seconds < self.departure_confirm_seconds:
            raise ValueError(
                "checkin_after_seconds must be at least departure_confirm_seconds"
            )


class TemporalTeaRecognizer:
    """Infer tea preparation from a short sequence of factual detections.

    A start requires repeated, time-separated co-occurrence of a person, a cup,
    and a preparation object plus observed motion.  Visual completion is kept
    deliberately conservative: the cup must move materially and stay close to
    the detected person in more than one frame.  No temperature is inferred.
    """

    CUP_LABELS = frozenset({"cup", "mug"})
    PREPARATION_LABELS = frozenset(
        {"bottle", "kettle", "flask", "teapot", "thermos"}
    )

    def __init__(
        self,
        *,
        window_seconds: float = 12.0,
        min_sequence_seconds: float = 0.8,
        min_activity_samples: int = 3,
        completion_move_ratio: float = 0.16,
    ) -> None:
        self.window_seconds = window_seconds
        self.min_sequence_seconds = min_sequence_seconds
        self.min_activity_samples = min_activity_samples
        self.completion_move_ratio = completion_move_ratio
        self._history: deque[tuple[int, FrameEvidence]] = deque()
        self._cup_origin: tuple[float, float] | None = None
        self._completion_candidates: deque[int] = deque(maxlen=3)

    def reset(self) -> None:
        self._history.clear()
        self._cup_origin = None
        self._completion_candidates.clear()

    def observe(
        self,
        observation_id: int,
        evidence: FrameEvidence,
        active_episode: dict[str, Any] | None,
    ) -> list[ActivityInference]:
        self._history.append((observation_id, evidence))
        cutoff = evidence.observed_at - timedelta(seconds=self.window_seconds)
        while self._history and self._history[0][1].observed_at < cutoff:
            self._history.popleft()

        if active_episode is None:
            start = self._infer_start()
            if start is not None:
                self._cup_origin = self._cup_center(evidence)
                return [start]
            return []

        state = active_episode["state"]
        if state in {EpisodeState.COMPLETED, EpisodeState.DISMISSED}:
            return []
        completion = self._infer_completion(observation_id, evidence)
        return [completion] if completion is not None else []

    def _infer_start(self) -> ActivityInference | None:
        qualifying = [
            (observation_id, item)
            for observation_id, item in self._history
            if self._is_activity_sample(item)
        ]
        if len(qualifying) < self.min_activity_samples:
            return None
        selected = qualifying[-self.min_activity_samples :]
        span = (
            selected[-1][1].observed_at - selected[0][1].observed_at
        ).total_seconds()
        if span < self.min_sequence_seconds:
            return None
        if max(item.motion_score for _, item in selected) < 0.01:
            return None

        object_confidences: list[float] = []
        for _, item in selected:
            object_confidences.extend(
                fact.confidence
                for fact in item.objects
                if fact.label.casefold()
                in self.CUP_LABELS | self.PREPARATION_LABELS
            )
        average_confidence = (
            sum(object_confidences) / len(object_confidences)
            if object_confidences
            else 0.5
        )
        confidence = min(0.98, 0.55 + 0.35 * average_confidence)
        return ActivityInference(
            kind="tea_preparation_started",
            confidence=confidence,
            reason=(
                "Repeated observations showed a person, a cup, a preparation "
                "object, and motion across a temporal sequence."
            ),
            evidence_observation_ids=tuple(item[0] for item in selected),
        )

    def _infer_completion(
        self, observation_id: int, evidence: FrameEvidence
    ) -> ActivityInference | None:
        if not evidence.usable_for_departure or not evidence.person_present:
            self._completion_candidates.clear()
            return None
        cup_center = self._cup_center(evidence)
        if cup_center is None:
            self._completion_candidates.clear()
            return None
        if self._cup_origin is None:
            self._cup_origin = cup_center
            return None

        distance = math.dist(cup_center, self._cup_origin)
        if distance < self.completion_move_ratio or not self._cup_near_person(
            evidence, cup_center
        ):
            self._completion_candidates.clear()
            return None

        self._completion_candidates.append(observation_id)
        if len(self._completion_candidates) < 2:
            return None
        return ActivityInference(
            kind="tea_preparation_completed",
            confidence=0.72,
            reason=(
                "The cup moved away from its preparation position and remained "
                "close to the visible person across multiple observations."
            ),
            evidence_observation_ids=tuple(self._completion_candidates),
        )

    def _is_activity_sample(self, evidence: FrameEvidence) -> bool:
        if not evidence.usable_for_departure or not evidence.person_present:
            return False
        labels = {item.label.casefold() for item in evidence.objects}
        return bool(labels & self.CUP_LABELS) and bool(
            labels & self.PREPARATION_LABELS
        )

    def _cup_center(self, evidence: FrameEvidence) -> tuple[float, float] | None:
        for item in evidence.objects:
            if (
                item.label.casefold() in self.CUP_LABELS
                and item.bbox_normalized_xywh is not None
            ):
                x, y, width, height = item.bbox_normalized_xywh
                return x + width / 2, y + height / 2
        return None

    @staticmethod
    def _cup_near_person(
        evidence: FrameEvidence, cup_center: tuple[float, float]
    ) -> bool:
        for item in evidence.objects:
            if (
                item.label.casefold() != "person"
                or item.bbox_normalized_xywh is None
            ):
                continue
            x, y, width, height = item.bbox_normalized_xywh
            margin_x = width * 0.25
            margin_y = height * 0.2
            if (
                x - margin_x <= cup_center[0] <= x + width + margin_x
                and y - margin_y <= cup_center[1] <= y + height + margin_y
            ):
                return True
        return False


class EpisodeCoordinator:
    """Persist observations and update one live activity episode at a time."""

    def __init__(
        self,
        database: MemoryDatabase,
        *,
        person_name: str = "Temi",
        config: EpisodePolicyConfig | None = None,
        recognizer: TemporalTeaRecognizer | None = None,
    ) -> None:
        self.database = database
        self.person_name = person_name.strip() or "there"
        self.config = config or EpisodePolicyConfig()
        self.recognizer = recognizer or TemporalTeaRecognizer()

    def observe(
        self, session_id: str, evidence: FrameEvidence
    ) -> dict[str, Any]:
        observed_at = to_iso(evidence.observed_at)
        objects = [item.as_dict() for item in evidence.objects]
        facts = {
            "observed": {
                "camera_state": evidence.camera_state.value,
                "scene_clear": evidence.scene_clear,
                "person_present": evidence.person_present,
                "object_labels": [item.label for item in evidence.objects],
                "motion_score": round(float(evidence.motion_score), 4),
            },
            **evidence.facts,
        }
        observation_id = self.database.add_live_observation(
            session_id=session_id,
            observed_at=observed_at,
            frame_path=evidence.frame_path,
            camera_state=evidence.camera_state.value,
            person_present=evidence.person_present,
            scene_clear=evidence.scene_clear,
            motion_score=evidence.motion_score,
            objects=objects,
            facts=facts,
        )

        active = self.database.get_active_activity_episode("tea_preparation")
        inferences = self.recognizer.observe(observation_id, evidence, active)
        started = next(
            (item for item in inferences if item.kind == "tea_preparation_started"),
            None,
        )
        if active is None and started is not None:
            active = self.database.create_activity_episode(
                episode_id=uuid.uuid4().hex,
                activity_type="tea_preparation",
                display_name="Making tea",
                person_name=self.person_name,
                confidence=started.confidence,
                started_at=observed_at,
                observation_id=observation_id,
                reason=started.reason,
                evidence={
                    "inference": started.kind,
                    "confidence": started.confidence,
                    "observation_ids": list(started.evidence_observation_ids),
                },
            )
            for supporting_id in started.evidence_observation_ids:
                self.database.attach_episode_evidence(
                    active["id"],
                    supporting_id,
                    role="activity_inference",
                    reason=started.reason,
                )

        checkin = None
        if active is not None:
            self.database.attach_episode_evidence(
                active["id"],
                observation_id,
                role="observed_fact",
                reason="Live camera observation associated with the active episode.",
            )
            completed = next(
                (
                    item
                    for item in inferences
                    if item.kind == "tea_preparation_completed"
                ),
                None,
            )
            if completed is not None:
                active = self.database.transition_activity_episode(
                    active["id"],
                    to_state=EpisodeState.COMPLETED.value,
                    reason=completed.reason,
                    created_at=observed_at,
                    evidence={
                        "inference": completed.kind,
                        "confidence": completed.confidence,
                        "observation_ids": list(
                            completed.evidence_observation_ids
                        ),
                    },
                    observation_id=observation_id,
                    completion_source="visually_inferred",
                )
                self._cancel_open_checkin(active["id"], evidence.observed_at)
                self.recognizer.reset()
            else:
                active, checkin = self._apply_presence_policy(
                    session_id, active, observation_id, evidence
                )

        return {
            "observation_id": observation_id,
            "episode": active,
            "checkin": checkin,
            "inferences": [
                {
                    "kind": item.kind,
                    "confidence": item.confidence,
                    "reason": item.reason,
                    "evidence_observation_ids": list(
                        item.evidence_observation_ids
                    ),
                }
                for item in inferences
            ],
        }

    def respond(
        self, text: str, *, now: datetime | None = None
    ) -> dict[str, Any]:
        now = now or utc_now()
        normalized = " ".join(text.casefold().strip().split())
        active = self.database.get_active_activity_episode("tea_preparation")
        if active is None:
            return {"handled": False, "intent": "none", "episode": None}
        if active["state"] != EpisodeState.AWAITING_RESPONSE.value:
            return {"handled": False, "intent": "not_awaiting", "episode": active}

        checkin = self.database.get_latest_episode_checkin(active["id"])
        if checkin is None or checkin["status"] != "awaiting_response":
            return {"handled": False, "intent": "not_awaiting", "episode": active}

        if self._matches(normalized, ("another minute", "give me", "snooze", "wait")):
            intent = "snooze"
            self.database.resolve_checkin(
                checkin["id"],
                status="responded",
                response_text=text,
                response_intent=intent,
                responded_at=to_iso(now),
            )
            next_checkin = now + timedelta(seconds=self.config.snooze_seconds)
            active = self.database.transition_activity_episode(
                active["id"],
                to_state=EpisodeState.SNOOZED.value,
                reason="The user asked to postpone the check-in.",
                created_at=to_iso(now),
                evidence={"response_text": text, "intent": intent},
                absence_started_at=active["absence_started_at"],
                next_checkin_at=to_iso(next_checkin),
                cooldown_until=to_iso(next_checkin),
            )
        elif self._matches(normalized, ("i'm done", "i am done", "done", "finished")):
            intent = "completed"
            self.database.resolve_checkin(
                checkin["id"],
                status="responded",
                response_text=text,
                response_intent=intent,
                responded_at=to_iso(now),
            )
            active = self.database.transition_activity_episode(
                active["id"],
                to_state=EpisodeState.COMPLETED.value,
                reason="The user confirmed the activity was complete.",
                created_at=to_iso(now),
                evidence={"response_text": text, "intent": intent},
                completion_source="user_confirmed",
            )
            self.recognizer.reset()
        elif self._matches(normalized, ("cancel", "dismiss", "never mind")):
            intent = "dismissed"
            self.database.resolve_checkin(
                checkin["id"],
                status="responded",
                response_text=text,
                response_intent=intent,
                responded_at=to_iso(now),
            )
            active = self.database.transition_activity_episode(
                active["id"],
                to_state=EpisodeState.DISMISSED.value,
                reason="The user dismissed the activity check-in.",
                created_at=to_iso(now),
                evidence={"response_text": text, "intent": intent},
            )
            self.recognizer.reset()
        else:
            return {"handled": False, "intent": "unknown", "episode": active}

        return {"handled": True, "intent": intent, "episode": active}

    def _apply_presence_policy(
        self,
        session_id: str,
        episode: dict[str, Any],
        observation_id: int,
        evidence: FrameEvidence,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        state = EpisodeState(episode["state"])
        now = evidence.observed_at
        now_iso = to_iso(now)

        if not evidence.usable_for_departure:
            if state is EpisodeState.POSSIBLY_INTERRUPTED:
                episode = self.database.transition_activity_episode(
                    episode["id"],
                    to_state=EpisodeState.ONGOING.value,
                    reason=(
                        "Departure evidence was discarded because the camera "
                        "became unavailable or the scene was occluded."
                    ),
                    created_at=now_iso,
                    evidence={
                        "camera_state": evidence.camera_state.value,
                        "scene_clear": evidence.scene_clear,
                    },
                    observation_id=observation_id,
                )
            elif state not in {EpisodeState.AWAITING_RESPONSE, EpisodeState.SNOOZED}:
                episode = self.database.update_episode_absence(
                    episode["id"],
                    updated_at=now_iso,
                    absence_started_at=None,
                    observation_id=observation_id,
                )
            return episode, None

        if evidence.person_present:
            if state in {
                EpisodeState.POSSIBLY_INTERRUPTED,
                EpisodeState.AWAITING_RESPONSE,
                EpisodeState.SNOOZED,
            }:
                self._cancel_open_checkin(episode["id"], now)
                episode = self.database.transition_activity_episode(
                    episode["id"],
                    to_state=EpisodeState.RESUMED.value,
                    reason="The person returned and activity evidence resumed.",
                    created_at=now_iso,
                    evidence={"person_present": True},
                    observation_id=observation_id,
                )
            elif state is EpisodeState.RESUMED:
                episode = self.database.transition_activity_episode(
                    episode["id"],
                    to_state=EpisodeState.ONGOING.value,
                    reason="Continued observations confirmed the resumed activity.",
                    created_at=now_iso,
                    evidence={"person_present": True},
                    observation_id=observation_id,
                )
            elif episode["absence_started_at"] is not None:
                episode = self.database.update_episode_absence(
                    episode["id"],
                    updated_at=now_iso,
                    absence_started_at=None,
                    observation_id=observation_id,
                )
            return episode, None

        absence_started = from_iso(episode["absence_started_at"])
        if absence_started is None:
            episode = self.database.update_episode_absence(
                episode["id"],
                updated_at=now_iso,
                absence_started_at=now_iso,
                observation_id=observation_id,
            )
            absence_started = now

        absent_seconds = max(0.0, (now - absence_started).total_seconds())
        if (
            state in {EpisodeState.ONGOING, EpisodeState.RESUMED}
            and absent_seconds >= self.config.departure_confirm_seconds
        ):
            episode = self.database.transition_activity_episode(
                episode["id"],
                to_state=EpisodeState.POSSIBLY_INTERRUPTED.value,
                reason="The person was absent across sustained, usable observations.",
                created_at=now_iso,
                evidence={
                    "absent_seconds": absent_seconds,
                    "camera_state": evidence.camera_state.value,
                },
                observation_id=observation_id,
                absence_started_at=to_iso(absence_started),
            )
            state = EpisodeState.POSSIBLY_INTERRUPTED

        due_from_snooze = (
            state is EpisodeState.SNOOZED
            and from_iso(episode["next_checkin_at"]) is not None
            and now >= from_iso(episode["next_checkin_at"])  # type: ignore[arg-type]
        )
        first_checkin_due = (
            state is EpisodeState.POSSIBLY_INTERRUPTED
            and absent_seconds >= self.config.checkin_after_seconds
        )
        cooldown_until = from_iso(episode["cooldown_until"])
        cooldown_elapsed = cooldown_until is None or now >= cooldown_until
        if (first_checkin_due or due_from_snooze) and cooldown_elapsed:
            prompt = (
                f"{episode['person_name']}, I noticed you were making tea earlier. "
                "Are you coming back?"
            )
            episode = self.database.transition_activity_episode(
                episode["id"],
                to_state=EpisodeState.AWAITING_RESPONSE.value,
                reason="The check-in policy reached its sustained-absence threshold.",
                created_at=now_iso,
                evidence={
                    "absent_seconds": absent_seconds,
                    "threshold_seconds": self.config.checkin_after_seconds,
                },
                observation_id=observation_id,
                absence_started_at=to_iso(absence_started),
                cooldown_until=to_iso(
                    now + timedelta(seconds=self.config.cooldown_seconds)
                ),
                increment_reminder=True,
            )
            checkin = self.database.create_checkin(
                checkin_id=uuid.uuid4().hex,
                episode_id=episode["id"],
                session_id=session_id,
                prompt=prompt,
                triggered_at=now_iso,
            )
            return episode, checkin

        return episode, None

    def _cancel_open_checkin(self, episode_id: str, now: datetime) -> None:
        checkin = self.database.get_latest_episode_checkin(episode_id)
        if checkin is None or checkin["status"] != "awaiting_response":
            return
        self.database.resolve_checkin(
            checkin["id"],
            status="cancelled",
            response_text=None,
            response_intent="return_observed",
            responded_at=to_iso(now),
        )

    @staticmethod
    def _matches(text: str, phrases: Iterable[str]) -> bool:
        return any(phrase in text for phrase in phrases)
