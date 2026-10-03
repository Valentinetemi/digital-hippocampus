"""Gemini-backed extraction of validated symbolic events from video evidence."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
EVENT_SCHEMA_VERSION = "symbolic-events-v1"

EventType = Literal[
    "picked_up",
    "placed",
    "moved",
    "opened",
    "closed",
    "entered",
    "exited",
]


class GeminiEventCandidate(BaseModel):
    """Strict schema for one model-proposed, observable temporal event."""

    model_config = ConfigDict(extra="forbid")

    event_type: EventType = Field(
        description="Observable action or change; never a static frame description."
    )
    start_time: float = Field(ge=0, description="Event start in seconds.")
    end_time: float = Field(ge=0, description="Event end in seconds.")
    actor: str | None = Field(
        default=None,
        description="Generic visible actor, or null when unsupported.",
    )
    object: str | None = Field(
        default=None,
        description="Visible affected object, or null when unsupported.",
    )
    source_location: str | None = Field(
        default=None,
        description="Visible starting location, or null when unsupported.",
    )
    destination_location: str | None = Field(
        default=None,
        description="Visible ending location, or null when unsupported.",
    )
    confidence: float = Field(
        ge=0,
        le=1,
        description="Model confidence that the visual event is supported.",
    )
    evidence_timestamps: list[float] = Field(
        min_length=1,
        description=(
            "One or more timestamps in seconds that support the event. Prefer "
            "multiple timestamps, but one is valid for a short event."
        ),
    )

    @field_validator(
        "actor", "object", "source_location", "destination_location", mode="after"
    )
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = " ".join(value.split()).strip()
        return normalized or None


class GeminiEventBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    events: list[GeminiEventCandidate] = Field(
        default_factory=list,
        description="Observable events in chronological order; empty is valid.",
    )


class EventEvidence(BaseModel):
    """A Gemini timestamp deterministically linked to local frame provenance."""

    timestamp_seconds: float
    frame_id: int | None
    frame_number: int | None
    frame_timestamp_seconds: float | None


class SymbolicEvent(BaseModel):
    """Application-validated event ready for persistence."""

    event_id: str
    video_id: str
    event_type: EventType
    start_time: float
    end_time: float
    actor: str | None
    object: str | None
    source_location: str | None
    destination_location: str | None
    confidence: float
    evidence: list[EventEvidence]


@dataclass(frozen=True)
class EventExtractionResult:
    events: list[SymbolicEvent]
    rejected_event_count: int


SYSTEM_INSTRUCTION = """You extract symbolic events from visual evidence.
Do not create persistent identities or relationships; application code handles those.
Use only visually observable changes and actions across time. Ignore spoken audio and narration.
Do not infer intent, hidden causes, off-camera actions, names, or cross-video identity.
Treat the supplied perception manifest as noisy supporting evidence; the video is primary.
Return an empty event list when no allowed event is clearly supported.
"""


def build_event_prompt(video: dict[str, Any]) -> str:
    """Build a compact grounding manifest from existing perception records."""

    manifest_frames: list[dict[str, Any]] = []
    for frame in video.get("frames", []):
        objects: list[dict[str, Any]] = []
        motion: dict[str, Any] | None = None
        for observation in frame.get("observations", []):
            if observation["kind"] == "object":
                objects.append(
                    {
                        "class": observation["label"],
                        "confidence": observation["confidence"],
                        "bbox_xywh": observation.get("details", {}).get("bbox_xywh"),
                    }
                )
            elif observation["kind"] == "motion":
                motion = {
                    "label": observation["label"],
                    "details": observation.get("details", {}),
                }
        manifest_frames.append(
            {
                "frame_id": frame["id"],
                "frame_number": frame["frame_number"],
                "timestamp_seconds": frame["timestamp_seconds"],
                "objects": objects,
                "motion": motion,
            }
        )

    manifest = {
        "video_id": video["id"],
        "duration_seconds": video["duration_seconds"],
        "sampled_frames": manifest_frames,
    }
    return """Analyze the uploaded video's visual stream across time and extract only
observable symbolic events.

Allowed event types: picked_up, placed, moved, opened, closed, entered, exited.

Rules:
- An event must be a visible action or state change across time, not a description of one frame.
- The uploaded video is the primary temporal evidence.
- The perception manifest below is supporting provenance and may miss short actions.
- Supply at least one valid evidence timestamp for every event; prefer multiple when visible.
- Do not add timestamps merely to increase the evidence count.
- Use null for actor, object, or locations when visual evidence does not support them.
- Use generic visible descriptions such as "person"; do not assign identity.
- Omit uncertain events rather than guessing.
- Keep all timestamps within the video duration.
- Return events in chronological order.

Existing sampled perception manifest:
""" + json.dumps(manifest, separators=(",", ":"), ensure_ascii=True)


class GeminiSymbolicEventExtractor:
    """Uploads one video to Gemini and returns schema-constrained candidates."""

    provider = "gemini"
    schema_version = EVENT_SCHEMA_VERSION

    MIME_TYPES = {
        ".mp4": "video/mp4",
        ".mov": "video/mov",
        ".avi": "video/avi",
        ".webm": "video/webm",
        ".m4v": "video/mp4",
    }

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str = DEFAULT_GEMINI_MODEL,
        client: Any | None = None,
        poll_interval_seconds: float = 2.0,
        processing_timeout_seconds: float = 600.0,
    ):
        if client is None and not api_key:
            raise ValueError("GEMINI_API_KEY is required for symbolic event extraction")
        self.api_key = api_key
        self.model = model
        self._client = client
        self.poll_interval_seconds = poll_interval_seconds
        self.processing_timeout_seconds = processing_timeout_seconds

    def extract_candidates(self, video: dict[str, Any]) -> GeminiEventBatch:
        video_path = Path(video["stored_path"])
        mime_type = self.MIME_TYPES.get(video_path.suffix.lower())
        if mime_type is None:
            raise ValueError(
                f"Gemini event extraction does not support {video_path.suffix or 'this file type'}"
            )

        client, owns_client = self._get_client()
        uploaded_file: Any | None = None
        try:
            uploaded_file = client.files.upload(
                file=str(video_path),
                config={"mime_type": mime_type, "display_name": video_path.name},
            )
            uploaded_file = self._wait_until_active(client, uploaded_file)
            if not uploaded_file.uri:
                raise RuntimeError("Gemini did not return a URI for the uploaded video")

            interaction = client.interactions.create(
                model=self.model,
                input=[
                    {
                        "type": "video",
                        "uri": uploaded_file.uri,
                        "mime_type": uploaded_file.mime_type or mime_type,
                        "processing": {"type": "static", "fps": 1.0},
                    },
                    {"type": "text", "text": build_event_prompt(video)},
                ],
                system_instruction=SYSTEM_INSTRUCTION,
                generation_config={"temperature": 0.1},
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": GeminiEventBatch.model_json_schema(),
                },
                store=False,
            )
            output_text = getattr(interaction, "output_text", None)
            if not output_text:
                raise RuntimeError("Gemini returned no structured event output")
            return GeminiEventBatch.model_validate_json(output_text)
        finally:
            try:
                if uploaded_file is not None and uploaded_file.name:
                    client.files.delete(name=uploaded_file.name)
            finally:
                if owns_client:
                    client.close()

    def extract(self, video: dict[str, Any]) -> EventExtractionResult:
        candidates = self.extract_candidates(video)
        return validate_and_resolve_events(
            candidates,
            video_id=video["id"],
            duration_seconds=float(video["duration_seconds"]),
            frames=video.get("frames", []),
        )

    def _get_client(self) -> tuple[Any, bool]:
        if self._client is not None:
            return self._client, False
        from google import genai

        return genai.Client(api_key=self.api_key), True

    def _wait_until_active(self, client: Any, uploaded_file: Any) -> Any:
        deadline = time.monotonic() + self.processing_timeout_seconds
        while True:
            state = getattr(getattr(uploaded_file, "state", None), "name", None)
            if state == "ACTIVE":
                return uploaded_file
            if state == "FAILED":
                error = getattr(uploaded_file, "error", None)
                raise RuntimeError(f"Gemini could not process the video: {error}")
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for Gemini to process the video")
            time.sleep(self.poll_interval_seconds)
            uploaded_file = client.files.get(name=uploaded_file.name)


def validate_and_resolve_events(
    batch: GeminiEventBatch,
    *,
    video_id: str,
    duration_seconds: float,
    frames: list[dict[str, Any]],
) -> EventExtractionResult:
    """Apply deterministic domain checks and attach nearest local frame refs."""

    accepted: list[SymbolicEvent] = []
    rejected = 0
    seen: set[tuple[Any, ...]] = set()

    for candidate in batch.events:
        timestamps = sorted(
            {round(float(value), 3) for value in candidate.evidence_timestamps}
        )
        structurally_valid = (
            duration_seconds > 0
            and 0 <= candidate.start_time < candidate.end_time <= duration_seconds
            and bool(timestamps)
            and all(0 <= timestamp <= duration_seconds for timestamp in timestamps)
        )
        identity = (
            candidate.event_type,
            round(candidate.start_time, 3),
            round(candidate.end_time, 3),
            candidate.actor,
            candidate.object,
            candidate.source_location,
            candidate.destination_location,
        )
        if not structurally_valid or identity in seen:
            rejected += 1
            continue

        seen.add(identity)
        evidence = [
            _resolve_evidence_timestamp(timestamp, frames) for timestamp in timestamps
        ]
        accepted.append(
            SymbolicEvent(
                event_id=uuid.uuid4().hex,
                video_id=video_id,
                event_type=candidate.event_type,
                start_time=round(candidate.start_time, 3),
                end_time=round(candidate.end_time, 3),
                actor=candidate.actor,
                object=candidate.object,
                source_location=candidate.source_location,
                destination_location=candidate.destination_location,
                confidence=round(candidate.confidence, 4),
                evidence=evidence,
            )
        )

    accepted.sort(key=lambda event: (event.start_time, event.end_time, event.event_id))
    return EventExtractionResult(accepted, rejected)


def _resolve_evidence_timestamp(
    timestamp: float, frames: list[dict[str, Any]]
) -> EventEvidence:
    if not frames:
        return EventEvidence(
            timestamp_seconds=round(timestamp, 3),
            frame_id=None,
            frame_number=None,
            frame_timestamp_seconds=None,
        )

    nearest = min(
        frames,
        key=lambda frame: (
            abs(float(frame["timestamp_seconds"]) - timestamp),
            float(frame["timestamp_seconds"]),
            int(frame["id"]),
        ),
    )
    return EventEvidence(
        timestamp_seconds=round(timestamp, 3),
        frame_id=int(nearest["id"]),
        frame_number=int(nearest["frame_number"]),
        frame_timestamp_seconds=round(float(nearest["timestamp_seconds"]), 3),
    )
