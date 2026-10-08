from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from digital_hippocampus.database import MemoryDatabase
from digital_hippocampus.live_memory import CameraState, to_iso
from digital_hippocampus.notifications import (
    CaregiverPolicyConfig,
    LocalCaregiverPreview,
)


class CaregiverPreviewTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database = MemoryDatabase(
            Path(self.temporary_directory.name) / "memory.db"
        )
        self.now = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)
        self.database.create_live_session("session", "fixture", to_iso(self.now))
        observation_id = self.database.add_live_observation(
            session_id="session",
            observed_at=to_iso(self.now),
            frame_path=None,
            camera_state="available",
            person_present=True,
            scene_clear=True,
            motion_score=0.1,
            objects=[],
            facts={"observed": {"person_present": True}},
        )
        self.episode = self.database.create_activity_episode(
            episode_id="episode",
            activity_type="tea_preparation",
            display_name="Making tea",
            person_name="Temi",
            confidence=0.81,
            started_at=to_iso(self.now),
            observation_id=observation_id,
            reason="Temporal evidence",
            evidence={"activity_is_inferred": True},
        )
        self.episode = self.database.transition_activity_episode(
            "episode",
            to_state="awaiting_response",
            reason="Sustained absence threshold reached",
            created_at=to_iso(self.now + timedelta(seconds=10)),
            evidence={"absent_seconds": 10},
            observation_id=observation_id,
            absence_started_at=to_iso(self.now),
            increment_reminder=True,
        )
        self.checkin = self.database.create_checkin(
            checkin_id="checkin",
            episode_id="episode",
            session_id="session",
            prompt="Temi, I noticed you were making tea earlier. Are you coming back?",
            triggered_at=to_iso(self.now + timedelta(seconds=10)),
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def evaluate(
        self,
        preview: LocalCaregiverPreview,
        seconds: float,
        *,
        camera_state: CameraState = CameraState.AVAILABLE,
        scene_clear: bool = True,
        person_present: bool | None = False,
    ) -> dict | None:
        return preview.evaluate(
            episode=self.episode,
            checkin=self.checkin,
            observed_at=self.now + timedelta(seconds=seconds),
            person_present=person_present,
            camera_state=camera_state,
            scene_clear=scene_clear,
        )

    def test_preview_requires_explicit_opt_in_and_recipient(self) -> None:
        disabled = LocalCaregiverPreview(self.database)
        self.assertIsNone(self.evaluate(disabled, 200))
        with self.assertRaisesRegex(ValueError, "recipient"):
            CaregiverPolicyConfig(enabled=True, recipient_label="")
        with self.assertRaisesRegex(ValueError, "External"):
            CaregiverPolicyConfig(
                enabled=True,
                recipient_label="Caregiver",
                external_delivery_enabled=True,
            )

    def test_preview_waits_for_delay_and_usable_absence_evidence(self) -> None:
        preview = LocalCaregiverPreview(
            self.database,
            CaregiverPolicyConfig(
                enabled=True,
                recipient_label="Ada (demo)",
                escalation_delay_seconds=30,
                notification_limit=1,
            ),
        )

        self.assertIsNone(self.evaluate(preview, 39))
        self.assertIsNone(
            self.evaluate(
                preview,
                50,
                camera_state=CameraState.OCCLUDED,
                scene_clear=False,
                person_present=None,
            )
        )
        notification = self.evaluate(preview, 50)

        assert notification is not None
        self.assertEqual(notification["status"], "preview")
        self.assertEqual(notification["adapter"], "local_preview")
        self.assertIn("No message was sent", notification["uncertainty_note"])
        self.assertTrue(notification["evidence"]["activity_is_inferred"])
        self.assertIsNone(self.evaluate(preview, 60))
        self.assertEqual(self.database.count_caregiver_previews("episode"), 1)


if __name__ == "__main__":
    unittest.main()
