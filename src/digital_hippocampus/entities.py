"""Entity records, track debouncing, and deterministic state events."""

from dataclasses import dataclass
from math import sqrt
from typing import Literal


# Bounding boxes use the same x, y, width, height order as YOLO observations.
BBoxXYWH = tuple[float, float, float, float]


def calculate_iou(first_bbox: BBoxXYWH, second_bbox: BBoxXYWH) -> float:
    """Return the overlap between two bounding boxes as a value from 0 to 1."""

    first_x, first_y, first_width, first_height = first_bbox
    second_x, second_y, second_width, second_height = second_bbox

    intersection_left = max(first_x, second_x)
    intersection_top = max(first_y, second_y)
    intersection_right = min(
        first_x + first_width,
        second_x + second_width,
    )
    intersection_bottom = min(
        first_y + first_height,
        second_y + second_height,
    )

    intersection_width = max(0.0, intersection_right - intersection_left)
    intersection_height = max(0.0, intersection_bottom - intersection_top)
    intersection_area = intersection_width * intersection_height

    first_area = first_width * first_height
    second_area = second_width * second_height
    union_area = first_area + second_area - intersection_area #union area is the total area coverred by both boxes, and it count there overlapping section only once 
    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


@dataclass
class Entity:
    """One physical object tracked within a single video."""

    entity_id: str
    video_id: str
    label: str
    first_seen: float
    last_seen: float
    last_bbox: BBoxXYWH
    # Confidence that the detector assigned the correct object label.
    detection_confidence: float
    # Confidence that the latest observation belongs to this entity.
    identity_confidence: float


EntityEventType = Literal["APPEARED", "DISAPPEARED", "MOVED"]


@dataclass(frozen=True)
class EntityStateEvent:
    event_id: str
    video_id: str
    entity_id: str
    event_type: EntityEventType
    timestamp: float
    confidence: float
    evidence_frame_number: int
    evidence_timestamp: float
    source_zone: str | None = None
    destination_zone: str | None = None


@dataclass
class PendingTrack:
    label: str
    consecutive_frames: int
    first_timestamp: float
    last_timestamp: float
    last_frame_number: int
    bbox: BBoxXYWH
    confidence: float
    embedding: tuple[float, ...] | None


class EntityResolver:
    """Owns the entity identities created for one video."""

    def __init__(
        self,
        video_id: str,
        *,
        min_iou: float = 0.3,
        max_time_gap: float = 2.5,
        min_consecutive_frames: int = 3,
        reid_similarity_threshold: float = 0.82,
        max_missed_frames: int = 15,
    ) -> None:
        if not video_id.strip():
            raise ValueError("video_id must not be empty")
        if not 0.0 <= min_iou <= 1.0:
            raise ValueError("min_iou must be between 0 and 1")
        if max_time_gap <= 0:
            raise ValueError("max_time_gap must be greater than zero")
        if min_consecutive_frames <= 0:
            raise ValueError("min_consecutive_frames must be greater than zero")
        if not 0.0 <= reid_similarity_threshold <= 1.0:
            raise ValueError("reid_similarity_threshold must be between 0 and 1")
        if max_missed_frames <= 0:
            raise ValueError("max_missed_frames must be greater than zero")

        self.video_id = video_id
        self.min_iou = min_iou
        self.max_time_gap = max_time_gap
        self.min_consecutive_frames = min_consecutive_frames
        self.reid_similarity_threshold = reid_similarity_threshold
        self.max_missed_frames = max_missed_frames
        self.entities: dict[str, Entity] = {}
        self.counters: dict[str, int] = {}
        self.track_to_entity: dict[int, str] = {}
        self.pending_tracks: dict[int, PendingTrack] = {}
        self.entity_embeddings: dict[str, tuple[float, ...]] = {}
        self.entity_last_frame: dict[str, int] = {}
        self.entity_zones: dict[str, str] = {}
        self.disappeared_entities: set[str] = set()
        self.state_events: list[EntityStateEvent] = []

    def needs_embedding(self, track_id: int) -> bool:
        """Return whether this track needs an appearance embedding for ReID."""

        return track_id not in self.track_to_entity and track_id not in self.pending_tracks

    def resolve_track(
        self,
        *,
        track_id: int,
        label: str,
        frame_number: int,
        timestamp: float,
        bbox: BBoxXYWH,
        frame_width: int,
        detection_confidence: float,
        embedding: tuple[float, ...] | None = None,
    ) -> Entity | None:
        """Debounce a ByteTrack track and resolve it to one persistent entity."""

        normalized_label = " ".join(label.split()).casefold()
        zone = self._zone_for_bbox(bbox, frame_width)
        entity_id = self.track_to_entity.get(track_id)
        if entity_id is not None:
            entity = self.entities[entity_id]
            self._update_tracked_entity(
                entity,
                frame_number=frame_number,
                timestamp=timestamp,
                bbox=bbox,
                confidence=detection_confidence,
                zone=zone,
                embedding=embedding,
            )
            return entity

        pending = self.pending_tracks.get(track_id)
        if (
            pending is None
            or pending.label != normalized_label
            or pending.last_frame_number + 1 != frame_number
        ):
            pending = PendingTrack(
                label=normalized_label,
                consecutive_frames=1,
                first_timestamp=timestamp,
                last_timestamp=timestamp,
                last_frame_number=frame_number,
                bbox=bbox,
                confidence=detection_confidence,
                embedding=embedding,
            )
            self.pending_tracks[track_id] = pending
        else:
            pending.consecutive_frames += 1
            pending.last_timestamp = timestamp
            pending.last_frame_number = frame_number
            pending.bbox = bbox
            pending.confidence = detection_confidence
            if embedding is not None:
                pending.embedding = embedding

        if pending.consecutive_frames < self.min_consecutive_frames:
            return None

        entity, similarity = self._find_reid_match(
            label=normalized_label,
            embedding=pending.embedding,
            frame_number=frame_number,
        )
        if entity is None:
            entity = self.create_entity(
                label=normalized_label,
                timestamp=pending.first_timestamp,
                bbox=pending.bbox,
                detection_confidence=pending.confidence,
            )
            entity.identity_confidence = 1.0
            self._emit_state_event(
                entity,
                "APPEARED",
                pending.first_timestamp,
                pending.confidence,
                evidence_frame_number=frame_number,
                evidence_timestamp=timestamp,
                destination_zone=zone,
            )
        else:
            entity.identity_confidence = similarity
            if entity.entity_id in self.disappeared_entities:
                self.disappeared_entities.remove(entity.entity_id)
                self._emit_state_event(
                    entity,
                    "APPEARED",
                    pending.first_timestamp,
                    similarity,
                    evidence_frame_number=frame_number,
                    evidence_timestamp=timestamp,
                    destination_zone=zone,
                )

        self.track_to_entity[track_id] = entity.entity_id
        self.entity_last_frame[entity.entity_id] = frame_number
        self.entity_zones.setdefault(entity.entity_id, zone)
        if pending.embedding is not None:
            self.entity_embeddings[entity.entity_id] = pending.embedding
        del self.pending_tracks[track_id]
        self._update_tracked_entity(
            entity,
            frame_number=frame_number,
            timestamp=timestamp,
            bbox=pending.bbox,
            confidence=pending.confidence,
            zone=zone,
            embedding=pending.embedding,
        )
        return entity

    def advance_frame(
        self, *, frame_number: int, timestamp: float, visible_track_ids: set[int]
    ) -> None:
        """Expire missing confirmed entities and stale unconfirmed tracks."""

        stale_pending = [
            track_id
            for track_id, pending in self.pending_tracks.items()
            if pending.last_frame_number < frame_number - 1
        ]
        for track_id in stale_pending:
            del self.pending_tracks[track_id]

        visible_entities = {
            self.track_to_entity[track_id]
            for track_id in visible_track_ids
            if track_id in self.track_to_entity
        }
        for entity_id, last_frame in list(self.entity_last_frame.items()):
            if entity_id in visible_entities or entity_id in self.disappeared_entities:
                continue
            if frame_number - last_frame < self.max_missed_frames:
                continue
            entity = self.entities[entity_id]
            self.disappeared_entities.add(entity_id)
            self._emit_state_event(
                entity,
                "DISAPPEARED",
                timestamp,
                entity.identity_confidence,
                evidence_frame_number=frame_number,
                evidence_timestamp=timestamp,
                source_zone=self.entity_zones.get(entity_id),
            )

    def finalize(self, timestamp: float) -> None:
        """Close every entity still visible when the video ends."""

        for entity_id, entity in self.entities.items():
            if entity_id in self.disappeared_entities:
                continue
            self.disappeared_entities.add(entity_id)
            self._emit_state_event(
                entity,
                "DISAPPEARED",
                timestamp,
                entity.identity_confidence,
                evidence_frame_number=self.entity_last_frame[entity_id],
                evidence_timestamp=entity.last_seen,
                source_zone=self.entity_zones.get(entity_id),
            )

    def summary(self) -> dict[str, object]:
        entities_by_class: dict[str, int] = {}
        for entity in self.entities.values():
            entities_by_class[entity.label] = entities_by_class.get(entity.label, 0) + 1
        ordered_events = sorted(
            self.state_events,
            key=lambda event: (event.timestamp, event.event_id),
        )
        return {
            "entities_by_class": entities_by_class,
            "entity_count": len(self.entities),
            "event_count": len(ordered_events),
            "events": [
                {
                    "event_id": event.event_id,
                    "event_type": event.event_type,
                    "entity_id": event.entity_id,
                    "timestamp": event.timestamp,
                    "confidence": event.confidence,
                    "evidence_frame_number": event.evidence_frame_number,
                    "evidence_timestamp": event.evidence_timestamp,
                    "source_zone": event.source_zone,
                    "destination_zone": event.destination_zone,
                }
                for event in ordered_events
            ],
        }

    def _find_reid_match(
        self,
        *,
        label: str,
        embedding: tuple[float, ...] | None,
        frame_number: int,
    ) -> tuple[Entity | None, float]:
        if embedding is None:
            return None, 0.0
        best_entity: Entity | None = None
        best_similarity = self.reid_similarity_threshold
        for entity_id, entity in self.entities.items():
            if entity.label != label:
                continue
            if self.entity_last_frame.get(entity_id) == frame_number:
                continue
            stored_embedding = self.entity_embeddings.get(entity_id)
            if stored_embedding is None:
                continue
            similarity = self._cosine_similarity(stored_embedding, embedding)
            if similarity >= best_similarity:
                best_entity = entity
                best_similarity = similarity
        return best_entity, best_similarity

    def _update_tracked_entity(
        self,
        entity: Entity,
        *,
        frame_number: int,
        timestamp: float,
        bbox: BBoxXYWH,
        confidence: float,
        zone: str,
        embedding: tuple[float, ...] | None,
    ) -> None:
        previous_zone = self.entity_zones.get(entity.entity_id)
        if previous_zone is not None and previous_zone != zone:
            self._emit_state_event(
                entity,
                "MOVED",
                timestamp,
                min(confidence, entity.identity_confidence),
                evidence_frame_number=frame_number,
                evidence_timestamp=timestamp,
                source_zone=previous_zone,
                destination_zone=zone,
            )
        entity.last_seen = timestamp
        entity.last_bbox = bbox
        entity.detection_confidence = confidence
        self.entity_last_frame[entity.entity_id] = frame_number
        self.entity_zones[entity.entity_id] = zone
        if embedding is not None:
            previous = self.entity_embeddings.get(entity.entity_id)
            self.entity_embeddings[entity.entity_id] = (
                embedding
                if previous is None
                else self._normalize_vector(
                    tuple(0.8 * old + 0.2 * new for old, new in zip(previous, embedding))
                )
            )

    def _emit_state_event(
        self,
        entity: Entity,
        event_type: EntityEventType,
        timestamp: float,
        confidence: float,
        *,
        evidence_frame_number: int,
        evidence_timestamp: float,
        source_zone: str | None = None,
        destination_zone: str | None = None,
    ) -> None:
        self.state_events.append(
            EntityStateEvent(
                event_id=f"{self.video_id}:state-{len(self.state_events) + 1}",
                video_id=self.video_id,
                entity_id=entity.entity_id,
                event_type=event_type,
                timestamp=timestamp,
                confidence=max(0.0, min(1.0, confidence)),
                evidence_frame_number=evidence_frame_number,
                evidence_timestamp=evidence_timestamp,
                source_zone=source_zone,
                destination_zone=destination_zone,
            )
        )

    @staticmethod
    def _zone_for_bbox(bbox: BBoxXYWH, frame_width: int) -> str:
        center_x = bbox[0] + bbox[2] / 2
        ratio = center_x / max(frame_width, 1)
        if ratio < 1 / 3:
            return "left"
        if ratio < 2 / 3:
            return "center"
        return "right"

    @staticmethod
    def _normalize_vector(values: tuple[float, ...]) -> tuple[float, ...]:
        magnitude = sqrt(sum(value * value for value in values))
        if magnitude == 0:
            return values
        return tuple(value / magnitude for value in values)

    @classmethod
    def _cosine_similarity(
        cls, first: tuple[float, ...], second: tuple[float, ...]
    ) -> float:
        if len(first) != len(second) or not first:
            return 0.0
        first = cls._normalize_vector(first)
        second = cls._normalize_vector(second)
        return sum(left * right for left, right in zip(first, second))

    def find_best_match(
        self,
        *,
        label: str,
        timestamp: float,
        bbox: BBoxXYWH,
    ) -> tuple[Entity | None, float]:
        """Find the strongest recent same-label entity without changing it."""

        normalized_label = " ".join(label.split()).casefold()
        best_entity: Entity | None = None
        best_iou = 0.0

        for entity in self.entities.values():
            if entity.label != normalized_label:
                continue

            time_gap = timestamp - entity.last_seen
            # One entity cannot represent two detections from the same frame.
            if time_gap <= 0 or time_gap > self.max_time_gap:
                continue

            iou = calculate_iou(entity.last_bbox, bbox)
            if iou >= self.min_iou and iou > best_iou:
                best_entity = entity
                best_iou = iou

        return best_entity, best_iou

    def resolve_detection(
        self,
        *,
        label: str,
        timestamp: float,
        bbox: BBoxXYWH,
        detection_confidence: float,
    ) -> Entity:
        """Update the best matching entity, or create one when no match exists."""

        normalized_label = " ".join(label.split()).casefold()
        if not normalized_label:
            raise ValueError("label must not be empty")
        if timestamp < 0:
            raise ValueError("timestamp must not be negative")
        if not 0.0 <= detection_confidence <= 1.0:
            raise ValueError("detection_confidence must be between 0 and 1")

        x, y, width, height = bbox
        if width <= 0 or height <= 0:
            raise ValueError("bbox width and height must be greater than zero")
        normalized_bbox = (x, y, width, height)

        entity, match_iou = self.find_best_match(
            label=normalized_label,
            timestamp=timestamp,
            bbox=normalized_bbox,
        )
        if entity is None:
            return self.create_entity(
                label=normalized_label,
                timestamp=timestamp,
                bbox=normalized_bbox,
                detection_confidence=detection_confidence,
            )

        entity.last_seen = timestamp
        entity.last_bbox = normalized_bbox
        entity.detection_confidence = detection_confidence
        entity.identity_confidence = match_iou
        return entity

    def create_entity(
        self,
        *,
        label: str,
        timestamp: float,
        bbox: BBoxXYWH,
        detection_confidence: float,
    ) -> Entity:
        """Create and remember an entity from one object detection."""

        normalized_label = " ".join(label.split()).casefold()
        if not normalized_label:
            raise ValueError("label must not be empty")
        if timestamp < 0:
            raise ValueError("timestamp must not be negative")
        if not 0.0 <= detection_confidence <= 1.0:
            raise ValueError("detection_confidence must be between 0 and 1")

        x, y, width, height = bbox
        if width <= 0 or height <= 0:
            raise ValueError("bbox width and height must be greater than zero")

        entity_number = self.counters.get(normalized_label, 0) + 1
        self.counters[normalized_label] = entity_number
        entity_label = normalized_label.replace(" ", "-")
        entity_id = f"{self.video_id}:{entity_label}-{entity_number}"

        entity = Entity(
            entity_id=entity_id,
            video_id=self.video_id,
            label=normalized_label,
            first_seen=timestamp,
            last_seen=timestamp,
            last_bbox=(x, y, width, height),
            detection_confidence=detection_confidence,
            identity_confidence=1.0,
        )
        self.entities[entity_id] = entity
        return entity
