from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from digital_hippocampus.database import MemoryDatabase
from digital_hippocampus.live_memory import (
    CameraState,
    EpisodeCoordinator,
    EpisodePolicyConfig,
    EpisodeState,
    FrameEvidence,
    ObjectFact,
    to_iso,
)


class LiveEpisodePolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database = MemoryDatabase(
            Path(self.temporary_directory.name) / "memory.db"
        )
        self.base = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)
        self.session_id = "live-session"
        self.database.create_live_session(
            self.session_id, "test-fixture", to_iso(self.base)
        )
        self.config = EpisodePolicyConfig(
            departure_confirm_seconds=2,
            checkin_after_seconds=5,
            snooze_seconds=60,
            cooldown_seconds=60,
        )
        self.coordinator = EpisodeCoordinator(
            self.database,
            person_name="Temi",
            config=self.config,
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def evidence(
        self,
        seconds: float,
        *,
        person: bool | None,
        camera_state: CameraState = CameraState.AVAILABLE,
        scene_clear: bool = True,
        cup_x: float = 0.1,
        motion: float = 0.03,
        include_tea_objects: bool = True,
    ) -> FrameEvidence:
        objects: list[ObjectFact] = []
        if person:
            objects.append(ObjectFact("person", 0.94, (0.0, 0.0, 0.9, 1.0)))
        if include_tea_objects:
            objects.extend(
                [
                    ObjectFact("cup", 0.88, (cup_x, 0.5, 0.08, 0.12)),
                    ObjectFact("bottle", 0.84, (0.28, 0.3, 0.1, 0.35)),
                ]
            )
        return FrameEvidence(
            observed_at=self.base + timedelta(seconds=seconds),
            camera_state=camera_state,
            scene_clear=scene_clear,
            person_present=person,
            motion_score=motion,
            objects=tuple(objects),
        )

    def begin_tea(self) -> dict:
        self.coordinator.observe(self.session_id, self.evidence(0, person=True))
        self.coordinator.observe(self.session_id, self.evidence(1, person=True))
        result = self.coordinator.observe(
            self.session_id, self.evidence(2, person=True)
        )
        assert result["episode"] is not None
        return result["episode"]

    def interrupt_and_trigger(self) -> dict:
        self.begin_tea()
        self.coordinator.observe(self.session_id, self.evidence(3, person=False))
        self.coordinator.observe(self.session_id, self.evidence(5, person=False))
        return self.coordinator.observe(
            self.session_id, self.evidence(8, person=False)
        )

    def test_interrupted_tea_preparation_triggers_one_grounded_checkin(self) -> None:
        result = self.interrupt_and_trigger()

        self.assertEqual(
            result["episode"]["state"], EpisodeState.AWAITING_RESPONSE.value
        )
        self.assertEqual(
            result["checkin"]["prompt"],
            "Temi, I noticed you were making tea earlier. Are you coming back?",
        )

        repeated = self.coordinator.observe(
            self.session_id, self.evidence(9, person=False)
        )
        self.assertIsNone(repeated["checkin"])
        with self.database.connect() as connection:
            count = connection.execute("SELECT COUNT(*) FROM checkins").fetchone()[0]
        self.assertEqual(count, 1)

    def test_visually_inferred_completion_prevents_checkin(self) -> None:
        self.begin_tea()
        self.coordinator.observe(
            self.session_id, self.evidence(3, person=True, cup_x=0.55)
        )
        completion = self.coordinator.observe(
            self.session_id, self.evidence(4, person=True, cup_x=0.57)
        )

        self.assertEqual(completion["episode"]["state"], EpisodeState.COMPLETED)
        self.assertEqual(
            completion["episode"]["completion_source"], "visually_inferred"
        )
        for seconds in (5, 8, 12):
            result = self.coordinator.observe(
                self.session_id, self.evidence(seconds, person=False)
            )
            self.assertIsNone(result["checkin"])

    def test_return_before_threshold_resumes_without_checkin(self) -> None:
        self.begin_tea()
        self.coordinator.observe(self.session_id, self.evidence(3, person=False))
        interrupted = self.coordinator.observe(
            self.session_id, self.evidence(5, person=False)
        )
        returned = self.coordinator.observe(
            self.session_id, self.evidence(6, person=True)
        )

        self.assertEqual(
            interrupted["episode"]["state"],
            EpisodeState.POSSIBLY_INTERRUPTED.value,
        )
        self.assertEqual(returned["episode"]["state"], EpisodeState.RESUMED.value)
        self.assertIsNone(returned["checkin"])

    def test_snooze_postpones_the_next_checkin(self) -> None:
        first = self.interrupt_and_trigger()
        response = self.coordinator.respond(
            "Give me another minute", now=self.base + timedelta(seconds=9)
        )

        self.assertTrue(response["handled"])
        self.assertEqual(response["episode"]["state"], EpisodeState.SNOOZED.value)
        early = self.coordinator.observe(
            self.session_id, self.evidence(68, person=False)
        )
        due = self.coordinator.observe(
            self.session_id, self.evidence(69, person=False)
        )
        self.assertIsNone(early["checkin"])
        self.assertIsNotNone(due["checkin"])
        self.assertNotEqual(first["checkin"]["id"], due["checkin"]["id"])

    def test_user_completion_and_dismissal_are_terminal(self) -> None:
        self.interrupt_and_trigger()
        completed = self.coordinator.respond(
            "I'm done", now=self.base + timedelta(seconds=9)
        )
        self.assertEqual(completed["episode"]["state"], EpisodeState.COMPLETED)
        self.assertEqual(completed["episode"]["completion_source"], "user_confirmed")
        result = self.coordinator.observe(
            self.session_id, self.evidence(20, person=False)
        )
        self.assertIsNone(result["checkin"])

        # A separate coordinator/database path checks the other terminal intent.
        second_session = "dismiss-session"
        self.database.create_live_session(
            second_session, "test-fixture", to_iso(self.base + timedelta(minutes=2))
        )
        dismissed_coordinator = EpisodeCoordinator(
            self.database, person_name="Temi", config=self.config
        )
        for seconds in (120, 121, 122):
            dismissed_coordinator.observe(
                second_session, self.evidence(seconds, person=True)
            )
        for seconds in (123, 125, 128):
            dismissed_coordinator.observe(
                second_session, self.evidence(seconds, person=False)
            )
        dismissed = dismissed_coordinator.respond(
            "Cancel", now=self.base + timedelta(seconds=129)
        )
        self.assertEqual(dismissed["episode"]["state"], EpisodeState.DISMISSED)

    def test_occlusion_and_camera_failure_do_not_confirm_departure(self) -> None:
        episode = self.begin_tea()
        for seconds, camera_state in (
            (3, CameraState.OCCLUDED),
            (10, CameraState.UNAVAILABLE),
            (20, CameraState.OCCLUDED),
        ):
            result = self.coordinator.observe(
                self.session_id,
                self.evidence(
                    seconds,
                    person=None,
                    camera_state=camera_state,
                    scene_clear=False,
                    include_tea_objects=False,
                ),
            )
            self.assertIsNone(result["checkin"])

        stored = self.database.get_activity_episode(episode["id"])
        assert stored is not None
        self.assertEqual(stored["state"], EpisodeState.ONGOING)
        self.assertIsNone(stored["absence_started_at"])

    def test_restart_preserves_episode_without_replaying_old_checkin(self) -> None:
        first = self.interrupt_and_trigger()
        restarted = EpisodeCoordinator(
            self.database, person_name="Temi", config=self.config
        )

        result = restarted.observe(
            self.session_id, self.evidence(10, person=False)
        )

        self.assertEqual(result["episode"]["id"], first["episode"]["id"])
        self.assertEqual(
            result["episode"]["state"], EpisodeState.AWAITING_RESPONSE.value
        )
        self.assertIsNone(result["checkin"])
        with self.database.connect() as connection:
            count = connection.execute("SELECT COUNT(*) FROM checkins").fetchone()[0]
        self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
