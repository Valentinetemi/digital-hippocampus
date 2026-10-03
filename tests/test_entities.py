from __future__ import annotations

import unittest

from digital_hippocampus.entities import EntityResolver, calculate_iou


class BoundingBoxIoUTest(unittest.TestCase):
    def test_identical_boxes_have_complete_overlap(self) -> None:
        bbox = (10.0, 20.0, 30.0, 40.0)

        self.assertEqual(calculate_iou(bbox, bbox), 1.0)

    def test_separate_boxes_have_no_overlap(self) -> None:
        first_bbox = (0.0, 0.0, 10.0, 10.0)
        second_bbox = (20.0, 20.0, 10.0, 10.0)

        self.assertEqual(calculate_iou(first_bbox, second_bbox), 0.0)

    def test_partially_overlapping_boxes_have_partial_iou(self) -> None:
        first_bbox = (0.0, 0.0, 10.0, 10.0)
        second_bbox = (5.0, 5.0, 10.0, 10.0)

        self.assertAlmostEqual(calculate_iou(first_bbox, second_bbox), 25 / 175)


class EntityResolverTest(unittest.TestCase):
    def test_tracked_entity_requires_three_consecutive_frames(self) -> None:
        resolver = EntityResolver("video-1", min_consecutive_frames=3)

        first = resolver.resolve_track(
            track_id=7,
            label="laptop",
            frame_number=0,
            timestamp=0.0,
            bbox=(10.0, 10.0, 20.0, 20.0),
            frame_width=100,
            detection_confidence=0.9,
            embedding=(1.0, 0.0),
        )
        second = resolver.resolve_track(
            track_id=7,
            label="laptop",
            frame_number=1,
            timestamp=0.1,
            bbox=(11.0, 10.0, 20.0, 20.0),
            frame_width=100,
            detection_confidence=0.88,
        )
        third = resolver.resolve_track(
            track_id=7,
            label="laptop",
            frame_number=2,
            timestamp=0.2,
            bbox=(12.0, 10.0, 20.0, 20.0),
            frame_width=100,
            detection_confidence=0.87,
        )

        self.assertIsNone(first)
        self.assertIsNone(second)
        self.assertIsNotNone(third)
        self.assertEqual(len(resolver.entities), 1)
        self.assertEqual(resolver.state_events[0].event_type, "APPEARED")

    def test_track_id_break_reuses_entity_with_similar_embedding(self) -> None:
        resolver = EntityResolver(
            "video-1",
            min_consecutive_frames=2,
            max_missed_frames=1,
            reid_similarity_threshold=0.8,
        )
        resolver.resolve_track(
            track_id=1,
            label="laptop",
            frame_number=0,
            timestamp=0.0,
            bbox=(5.0, 5.0, 20.0, 20.0),
            frame_width=100,
            detection_confidence=0.9,
            embedding=(1.0, 0.0),
        )
        first_entity = resolver.resolve_track(
            track_id=1,
            label="laptop",
            frame_number=1,
            timestamp=0.1,
            bbox=(6.0, 5.0, 20.0, 20.0),
            frame_width=100,
            detection_confidence=0.9,
        )
        resolver.advance_frame(
            frame_number=2,
            timestamp=0.2,
            visible_track_ids=set(),
        )
        resolver.resolve_track(
            track_id=9,
            label="laptop",
            frame_number=3,
            timestamp=0.3,
            bbox=(7.0, 5.0, 20.0, 20.0),
            frame_width=100,
            detection_confidence=0.88,
            embedding=(0.99, 0.01),
        )
        reidentified = resolver.resolve_track(
            track_id=9,
            label="laptop",
            frame_number=4,
            timestamp=0.4,
            bbox=(8.0, 5.0, 20.0, 20.0),
            frame_width=100,
            detection_confidence=0.87,
        )

        self.assertIsNotNone(first_entity)
        self.assertIs(reidentified, first_entity)
        self.assertEqual(len(resolver.entities), 1)

    def test_zone_change_emits_moved_event(self) -> None:
        resolver = EntityResolver("video-1", min_consecutive_frames=2)
        resolver.resolve_track(
            track_id=1,
            label="laptop",
            frame_number=0,
            timestamp=0.0,
            bbox=(5.0, 5.0, 10.0, 10.0),
            frame_width=90,
            detection_confidence=0.9,
            embedding=(1.0, 0.0),
        )
        resolver.resolve_track(
            track_id=1,
            label="laptop",
            frame_number=1,
            timestamp=0.1,
            bbox=(6.0, 5.0, 10.0, 10.0),
            frame_width=90,
            detection_confidence=0.9,
        )
        resolver.resolve_track(
            track_id=1,
            label="laptop",
            frame_number=2,
            timestamp=0.2,
            bbox=(40.0, 5.0, 10.0, 10.0),
            frame_width=90,
            detection_confidence=0.88,
        )

        moved = next(
            event for event in resolver.state_events if event.event_type == "MOVED"
        )
        self.assertEqual(moved.source_zone, "left")
        self.assertEqual(moved.destination_zone, "center")

    def test_two_detections_in_one_frame_cannot_use_the_same_entity(self) -> None:
        resolver = EntityResolver("video-1", min_iou=0.3)

        first = resolver.resolve_detection(
            label="person",
            timestamp=1.0,
            bbox=(10.0, 10.0, 20.0, 20.0),
            detection_confidence=0.9,
        )
        second = resolver.resolve_detection(
            label="person",
            timestamp=1.0,
            bbox=(12.0, 12.0, 20.0, 20.0),
            detection_confidence=0.85,
        )

        self.assertEqual(first.entity_id, "video-1:person-1")
        self.assertEqual(second.entity_id, "video-1:person-2")
        self.assertEqual(len(resolver.entities), 2)

    def test_resolve_detection_updates_a_matching_entity(self) -> None:
        resolver = EntityResolver("video-1", min_iou=0.3)
        original = resolver.resolve_detection(
            label="person",
            timestamp=1.0,
            bbox=(10.0, 10.0, 20.0, 20.0),
            detection_confidence=0.9,
        )

        resolved = resolver.resolve_detection(
            label="person",
            timestamp=2.0,
            bbox=(12.0, 12.0, 20.0, 20.0),
            detection_confidence=0.85,
        )

        self.assertIs(resolved, original)
        self.assertEqual(resolved.entity_id, "video-1:person-1")
        self.assertEqual(resolved.first_seen, 1.0)
        self.assertEqual(resolved.last_seen, 2.0)
        self.assertEqual(resolved.last_bbox, (12.0, 12.0, 20.0, 20.0))
        self.assertEqual(resolved.detection_confidence, 0.85)
        self.assertGreaterEqual(resolved.identity_confidence, 0.3)
        self.assertEqual(len(resolver.entities), 1)
        self.assertEqual(resolver.counters["person"], 1)

    def test_resolve_detection_creates_entity_when_no_match_exists(self) -> None:
        resolver = EntityResolver("video-1", min_iou=0.3)
        first = resolver.resolve_detection(
            label="person",
            timestamp=1.0,
            bbox=(0.0, 0.0, 10.0, 10.0),
            detection_confidence=0.9,
        )

        second = resolver.resolve_detection(
            label="person",
            timestamp=2.0,
            bbox=(100.0, 100.0, 10.0, 10.0),
            detection_confidence=0.8,
        )

        self.assertEqual(first.entity_id, "video-1:person-1")
        self.assertEqual(second.entity_id, "video-1:person-2")
        self.assertEqual(len(resolver.entities), 2)

    def test_find_best_match_returns_recent_same_label_entity(self) -> None:
        resolver = EntityResolver("video-1", min_iou=0.3, max_time_gap=2.5)
        person = resolver.create_entity(
            label="person",
            timestamp=1.0,
            bbox=(10.0, 10.0, 20.0, 20.0),
            detection_confidence=0.9,
        )

        match, score = resolver.find_best_match(
            label="person",
            timestamp=2.0,
            bbox=(12.0, 12.0, 20.0, 20.0),
        )

        self.assertIs(match, person)
        self.assertGreater(score, 0.3)

    def test_find_best_match_ignores_a_different_label(self) -> None:
        resolver = EntityResolver("video-1")
        resolver.create_entity(
            label="chair",
            timestamp=1.0,
            bbox=(10.0, 10.0, 20.0, 20.0),
            detection_confidence=0.9,
        )

        match, score = resolver.find_best_match(
            label="person",
            timestamp=2.0,
            bbox=(10.0, 10.0, 20.0, 20.0),
        )

        self.assertIsNone(match)
        self.assertEqual(score, 0.0)

    def test_find_best_match_ignores_a_stale_entity(self) -> None:
        resolver = EntityResolver("video-1", max_time_gap=2.5)
        resolver.create_entity(
            label="person",
            timestamp=1.0,
            bbox=(10.0, 10.0, 20.0, 20.0),
            detection_confidence=0.9,
        )

        match, score = resolver.find_best_match(
            label="person",
            timestamp=4.0,
            bbox=(10.0, 10.0, 20.0, 20.0),
        )

        self.assertIsNone(match)
        self.assertEqual(score, 0.0)

    def test_create_entity_stores_a_new_entity(self) -> None:
        resolver = EntityResolver("video-1")

        entity = resolver.create_entity(
            label="Person",
            timestamp=2.5,
            bbox=(10.0, 20.0, 30.0, 40.0),
            detection_confidence=0.91,
        )

        self.assertEqual(entity.entity_id, "video-1:person-1")
        self.assertEqual(entity.video_id, "video-1")
        self.assertEqual(entity.label, "person")
        self.assertEqual(entity.first_seen, 2.5)
        self.assertEqual(entity.last_seen, 2.5)
        self.assertEqual(entity.last_bbox, (10.0, 20.0, 30.0, 40.0))
        self.assertEqual(entity.detection_confidence, 0.91)
        self.assertEqual(entity.identity_confidence, 1.0)
        self.assertIs(resolver.entities[entity.entity_id], entity)

    def test_create_entity_increments_each_label_separately(self) -> None:
        resolver = EntityResolver("video-1")

        first_person = resolver.create_entity(
            label="person",
            timestamp=0.0,
            bbox=(0.0, 0.0, 10.0, 10.0),
            detection_confidence=0.9,
        )
        chair = resolver.create_entity(
            label="chair",
            timestamp=0.0,
            bbox=(20.0, 20.0, 10.0, 10.0),
            detection_confidence=0.8,
        )
        second_person = resolver.create_entity(
            label="person",
            timestamp=1.0,
            bbox=(40.0, 40.0, 10.0, 10.0),
            detection_confidence=0.85,
        )

        self.assertEqual(first_person.entity_id, "video-1:person-1")
        self.assertEqual(chair.entity_id, "video-1:chair-1")
        self.assertEqual(second_person.entity_id, "video-1:person-2")


if __name__ == "__main__":
    unittest.main()
