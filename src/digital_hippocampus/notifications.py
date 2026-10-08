"""Opt-in caregiver escalation policy with a development-safe local outbox."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from .database import MemoryDatabase
from .live_memory import CameraState, from_iso, to_iso


@dataclass(frozen=True)
class CaregiverPolicyConfig:
    enabled: bool = False
    recipient_label: str | None = None
    escalation_delay_seconds: float = 120.0
    notification_limit: int = 1
    external_delivery_enabled: bool = False

    def __post_init__(self) -> None:
        if self.escalation_delay_seconds <= 0:
            raise ValueError("Caregiver escalation delay must be greater than zero")
        if self.notification_limit <= 0:
            raise ValueError("Caregiver notification limit must be greater than zero")
        if self.enabled and not (self.recipient_label or "").strip():
            raise ValueError("Select a caregiver preview recipient")
        if self.external_delivery_enabled:
            raise ValueError(
                "External caregiver delivery is not configured in this development build"
            )


class LocalCaregiverPreview:
    """Create a local preview only; this adapter has no network send capability."""

    adapter_name = "local_preview"

    def __init__(
        self,
        database: MemoryDatabase,
        config: CaregiverPolicyConfig | None = None,
    ) -> None:
        self.database = database
        self.config = config or CaregiverPolicyConfig()

    def evaluate(
        self,
        *,
        episode: dict[str, Any] | None,
        checkin: dict[str, Any] | None,
        observed_at: datetime,
        person_present: bool | None,
        camera_state: CameraState,
        scene_clear: bool,
    ) -> dict[str, Any] | None:
        if not self.config.enabled or episode is None or checkin is None:
            return None
        if episode["state"] != "awaiting_response":
            return None
        if checkin["status"] != "awaiting_response":
            return None
        if (
            camera_state is not CameraState.AVAILABLE
            or not scene_clear
            or person_present is not False
        ):
            return None

        triggered_at = from_iso(checkin["triggered_at"])
        if triggered_at is None:
            return None
        due_at = triggered_at + timedelta(
            seconds=self.config.escalation_delay_seconds
        )
        if observed_at < due_at:
            return None
        if (
            self.database.count_caregiver_previews(episode["id"])
            >= self.config.notification_limit
        ):
            return None

        recipient = (self.config.recipient_label or "Caregiver").strip()
        confidence_percent = round(float(episode["confidence"]) * 100)
        message = (
            f"Local preview for {recipient}: the system inferred that "
            f"{episode['person_name']} was {episode['display_name'].casefold()} "
            f"earlier ({confidence_percent}% activity confidence). A personal "
            "check-in is still awaiting a response. In the latest clear camera "
            "observation, a person was not detected."
        )
        uncertainty = (
            "This is an uncertain camera-based observation, not proof of danger, "
            "departure, or the state of any appliance. No message was sent."
        )
        return self.database.create_caregiver_preview(
            notification_id=uuid.uuid4().hex,
            episode_id=episode["id"],
            checkin_id=checkin["id"],
            recipient_label=recipient,
            message=message,
            uncertainty_note=uncertainty,
            evidence={
                "activity_is_inferred": True,
                "activity_confidence": episode["confidence"],
                "checkin_triggered_at": checkin["triggered_at"],
                "latest_observed_fact": {"person_present": False},
                "camera_state": camera_state.value,
                "scene_clear": scene_clear,
                "delay_seconds": self.config.escalation_delay_seconds,
            },
            created_at=to_iso(observed_at),
        )
