"""SQLite persistence for videos, frames, and observations."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS videos (
    id TEXT PRIMARY KEY,
    original_name TEXT NOT NULL,
    stored_path TEXT NOT NULL,
    uploaded_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    status TEXT NOT NULL CHECK(status IN ('processing', 'complete', 'failed')),
    duration_seconds REAL,
    fps REAL,
    total_frames INTEGER,
    error TEXT
);

CREATE TABLE IF NOT EXISTS entities (
    entity_id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    label TEXT NOT NULL,
    first_seen REAL NOT NULL CHECK(first_seen >= 0),
    last_seen REAL NOT NULL CHECK(last_seen >= first_seen),
    last_bbox_x REAL NOT NULL,
    last_bbox_y REAL NOT NULL,
    last_bbox_width REAL NOT NULL CHECK(last_bbox_width > 0),
    last_bbox_height REAL NOT NULL CHECK(last_bbox_height > 0),
    detection_confidence REAL NOT NULL
        CHECK(detection_confidence >= 0 AND detection_confidence <= 1),
    identity_confidence REAL NOT NULL
        CHECK(identity_confidence >= 0 AND identity_confidence <= 1)
);

CREATE TABLE IF NOT EXISTS frames (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    frame_number INTEGER NOT NULL,
    timestamp_seconds REAL NOT NULL,
    image_path TEXT NOT NULL,
    UNIQUE(video_id, frame_number)
);

CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    label TEXT NOT NULL,
    confidence REAL,
    details_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS frames_video_time
    ON frames(video_id, timestamp_seconds);
CREATE INDEX IF NOT EXISTS observations_frame
    ON observations(frame_id);
CREATE INDEX IF NOT EXISTS entities_video_time
    ON entities(video_id, last_seen);

CREATE TABLE IF NOT EXISTS entity_state_events (
    event_id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    entity_id TEXT NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    event_type TEXT NOT NULL CHECK(event_type IN (
        'APPEARED', 'DISAPPEARED', 'MOVED'
    )),
    timestamp_seconds REAL NOT NULL CHECK(timestamp_seconds >= 0),
    confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
    source_zone TEXT,
    destination_zone TEXT
);

CREATE INDEX IF NOT EXISTS entity_state_events_video_time
    ON entity_state_events(video_id, timestamp_seconds);

CREATE TABLE IF NOT EXISTS event_extractions (
    id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('processing', 'complete', 'failed')),
    event_count INTEGER NOT NULL DEFAULT 0,
    rejected_event_count INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS symbolic_events (
    event_id TEXT PRIMARY KEY,
    extraction_id TEXT NOT NULL REFERENCES event_extractions(id) ON DELETE CASCADE,
    video_id TEXT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    event_type TEXT NOT NULL CHECK(event_type IN (
        'picked_up', 'placed', 'moved', 'opened', 'closed', 'entered', 'exited'
    )),
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    actor TEXT,
    object_label TEXT,
    source_location TEXT,
    destination_location TEXT,
    confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK(start_time >= 0 AND end_time > start_time)
);

CREATE TABLE IF NOT EXISTS event_evidence (
    event_id TEXT NOT NULL REFERENCES symbolic_events(event_id) ON DELETE CASCADE,
    timestamp_seconds REAL NOT NULL CHECK(timestamp_seconds >= 0),
    frame_id INTEGER REFERENCES frames(id) ON DELETE SET NULL,
    frame_number INTEGER,
    frame_timestamp_seconds REAL,
    PRIMARY KEY(event_id, timestamp_seconds)
);

CREATE INDEX IF NOT EXISTS event_extractions_video
    ON event_extractions(video_id, started_at);
CREATE INDEX IF NOT EXISTS symbolic_events_video_time
    ON symbolic_events(video_id, start_time);
CREATE INDEX IF NOT EXISTS event_evidence_frame
    ON event_evidence(frame_id);

CREATE TABLE IF NOT EXISTS live_sessions (
    id TEXT PRIMARY KEY,
    camera_source TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('running', 'stopped', 'failed')),
    started_at TEXT NOT NULL,
    stopped_at TEXT,
    last_frame_at TEXT,
    dropped_frames INTEGER NOT NULL DEFAULT 0,
    error TEXT
);

CREATE TABLE IF NOT EXISTS live_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES live_sessions(id) ON DELETE CASCADE,
    observed_at TEXT NOT NULL,
    frame_path TEXT,
    camera_state TEXT NOT NULL CHECK(camera_state IN (
        'available', 'occluded', 'unavailable'
    )),
    person_present INTEGER,
    scene_clear INTEGER NOT NULL,
    motion_score REAL NOT NULL DEFAULT 0,
    objects_json TEXT NOT NULL DEFAULT '[]',
    facts_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS live_observations_session_time
    ON live_observations(session_id, observed_at);

CREATE TABLE IF NOT EXISTS activity_episodes (
    id TEXT PRIMARY KEY,
    activity_type TEXT NOT NULL,
    display_name TEXT NOT NULL,
    person_name TEXT NOT NULL,
    state TEXT NOT NULL CHECK(state IN (
        'ongoing', 'possibly_interrupted', 'awaiting_response',
        'snoozed', 'resumed', 'completed', 'dismissed'
    )),
    confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
    started_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT,
    completion_source TEXT CHECK(completion_source IN (
        'visually_inferred', 'user_confirmed'
    )),
    absence_started_at TEXT,
    next_checkin_at TEXT,
    cooldown_until TEXT,
    reminder_count INTEGER NOT NULL DEFAULT 0,
    last_observation_id INTEGER REFERENCES live_observations(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS activity_episodes_state_time
    ON activity_episodes(state, updated_at);

CREATE TABLE IF NOT EXISTS episode_transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id TEXT NOT NULL REFERENCES activity_episodes(id) ON DELETE CASCADE,
    from_state TEXT,
    to_state TEXT NOT NULL,
    reason TEXT NOT NULL,
    evidence_json TEXT NOT NULL DEFAULT '{}',
    observation_id INTEGER REFERENCES live_observations(id) ON DELETE SET NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS episode_transitions_episode_time
    ON episode_transitions(episode_id, created_at);

CREATE TABLE IF NOT EXISTS episode_evidence (
    episode_id TEXT NOT NULL REFERENCES activity_episodes(id) ON DELETE CASCADE,
    observation_id INTEGER NOT NULL REFERENCES live_observations(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK(role IN ('observed_fact', 'activity_inference', 'transition')),
    reason TEXT NOT NULL,
    PRIMARY KEY(episode_id, observation_id, role)
);

CREATE TABLE IF NOT EXISTS checkins (
    id TEXT PRIMARY KEY,
    episode_id TEXT NOT NULL REFERENCES activity_episodes(id) ON DELETE CASCADE,
    session_id TEXT REFERENCES live_sessions(id) ON DELETE SET NULL,
    prompt TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('awaiting_response', 'responded', 'cancelled')),
    triggered_at TEXT NOT NULL,
    response_text TEXT,
    response_intent TEXT,
    responded_at TEXT
);

CREATE INDEX IF NOT EXISTS checkins_episode_time
    ON checkins(episode_id, triggered_at);
"""


class MemoryDatabase:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def create_video(self, video_id: str, original_name: str, stored_path: Path) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO videos (id, original_name, stored_path, status)
                VALUES (?, ?, ?, 'processing')
                """,
                (video_id, original_name, str(stored_path)),
            )

    def finish_video(
        self,
        video_id: str,
        *,
        duration_seconds: float,
        fps: float,
        total_frames: int,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE videos
                SET status = 'complete', duration_seconds = ?, fps = ?,
                    total_frames = ?, error = NULL
                WHERE id = ?
                """,
                (duration_seconds, fps, total_frames, video_id),
            )

    def fail_video(self, video_id: str, error: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "UPDATE videos SET status = 'failed', error = ? WHERE id = ?",
                (error[:1000], video_id),
            )

    def upsert_entities(self, entities: Iterable[Any]) -> None:
        """Insert new entities and update the latest state of known entities."""

        entity_list = list(entities)
        if not entity_list:
            return

        with self.connect() as connection:
            connection.executemany(
                """
                INSERT INTO entities (
                    entity_id, video_id, label, first_seen, last_seen,
                    last_bbox_x, last_bbox_y, last_bbox_width, last_bbox_height,
                    detection_confidence, identity_confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(entity_id) DO UPDATE SET
                    label = excluded.label,
                    last_seen = excluded.last_seen,
                    last_bbox_x = excluded.last_bbox_x,
                    last_bbox_y = excluded.last_bbox_y,
                    last_bbox_width = excluded.last_bbox_width,
                    last_bbox_height = excluded.last_bbox_height,
                    detection_confidence = excluded.detection_confidence,
                    identity_confidence = excluded.identity_confidence
                """,
                [
                    (
                        entity.entity_id,
                        entity.video_id,
                        entity.label,
                        entity.first_seen,
                        entity.last_seen,
                        *entity.last_bbox,
                        entity.detection_confidence,
                        entity.identity_confidence,
                    )
                    for entity in entity_list
                ],
            )

    def get_video_entities(self, video_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            return self._entities(connection, video_id)

    def replace_entity_state_events(
        self, video_id: str, events: Iterable[Any]
    ) -> None:
        event_list = list(events)
        with self.connect() as connection:
            connection.execute(
                "DELETE FROM entity_state_events WHERE video_id = ?", (video_id,)
            )
            connection.executemany(
                """
                INSERT INTO entity_state_events (
                    event_id, video_id, entity_id, event_type,
                    timestamp_seconds, confidence, source_zone, destination_zone
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        event.event_id,
                        event.video_id,
                        event.entity_id,
                        event.event_type,
                        event.timestamp,
                        event.confidence,
                        event.source_zone,
                        event.destination_zone,
                    )
                    for event in event_list
                ],
            )

    def get_entity_state_events(self, video_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            return self._entity_state_events(connection, video_id)

    def add_frame_with_observations(
        self,
        video_id: str,
        frame_number: int,
        timestamp_seconds: float,
        image_path: Path,
        observations: Iterable[Any],
    ) -> None:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO frames
                    (video_id, frame_number, timestamp_seconds, image_path)
                VALUES (?, ?, ?, ?)
                """,
                (video_id, frame_number, timestamp_seconds, str(image_path)),
            )
            frame_id = cursor.lastrowid
            connection.executemany(
                """
                INSERT INTO observations
                    (frame_id, kind, label, confidence, details_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        frame_id,
                        item.kind,
                        item.label,
                        item.confidence,
                        json.dumps(item.details),
                    )
                    for item in observations
                ],
            )

    def get_video(self, video_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            video = connection.execute(
                "SELECT * FROM videos WHERE id = ?", (video_id,)
            ).fetchone()
            if video is None:
                return None

            frames = connection.execute(
                """
                SELECT id, frame_number, timestamp_seconds, image_path
                FROM frames WHERE video_id = ? ORDER BY timestamp_seconds
                """,
                (video_id,),
            ).fetchall()
            frame_results: list[dict[str, Any]] = []
            for frame in frames:
                observations = connection.execute(
                    """
                    SELECT kind, label, confidence, details_json
                    FROM observations WHERE frame_id = ? ORDER BY id
                    """,
                    (frame["id"],),
                ).fetchall()
                frame_result = dict(frame)
                frame_result["observations"] = [
                    {
                        "kind": row["kind"],
                        "label": row["label"],
                        "confidence": row["confidence"],
                        "details": json.loads(row["details_json"]),
                    }
                    for row in observations
                ]
                frame_results.append(frame_result)

            result = dict(video)
            result["frames"] = frame_results
            result["entities"] = self._entities(connection, video_id)
            result["entity_events"] = self._entity_state_events(
                connection, video_id
            )
            result["event_extraction"] = self._latest_event_extraction(
                connection, video_id
            )
            result["symbolic_events"] = self._symbolic_events(
                connection, video_id
            )
            return result

    def list_videos(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    v.*,
                    COUNT(f.id) AS sampled_frames,
                    (
                        SELECT status FROM event_extractions ee
                        WHERE ee.video_id = v.id
                        ORDER BY ee.rowid DESC LIMIT 1
                    ) AS event_status
                FROM videos v
                LEFT JOIN frames f ON f.video_id = v.id
                GROUP BY v.id
                ORDER BY v.uploaded_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    def _entities(
        self, connection: sqlite3.Connection, video_id: str
    ) -> list[dict[str, Any]]:
        rows = connection.execute(
            """
            SELECT entity_id, video_id, label, first_seen, last_seen,
                   last_bbox_x, last_bbox_y, last_bbox_width, last_bbox_height,
                   detection_confidence, identity_confidence
            FROM entities
            WHERE video_id = ?
            ORDER BY first_seen, entity_id
            """,
            (video_id,),
        ).fetchall()
        return [
            {
                "entity_id": row["entity_id"],
                "video_id": row["video_id"],
                "label": row["label"],
                "first_seen": row["first_seen"],
                "last_seen": row["last_seen"],
                "last_bbox": (
                    row["last_bbox_x"],
                    row["last_bbox_y"],
                    row["last_bbox_width"],
                    row["last_bbox_height"],
                ),
                "detection_confidence": row["detection_confidence"],
                "identity_confidence": row["identity_confidence"],
            }
            for row in rows
        ]

    def _entity_state_events(
        self, connection: sqlite3.Connection, video_id: str
    ) -> list[dict[str, Any]]:
        rows = connection.execute(
            """
            SELECT event_id, video_id, entity_id, event_type,
                   timestamp_seconds, confidence, source_zone, destination_zone
            FROM entity_state_events
            WHERE video_id = ?
            ORDER BY timestamp_seconds, event_id
            """,
            (video_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def create_event_extraction(
        self,
        extraction_id: str,
        video_id: str,
        *,
        provider: str,
        model: str,
        schema_version: str,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO event_extractions
                    (id, video_id, provider, model, schema_version, status)
                VALUES (?, ?, ?, ?, ?, 'processing')
                """,
                (extraction_id, video_id, provider, model, schema_version),
            )

    def complete_event_extraction(
        self,
        extraction_id: str,
        events: Iterable[Any],
        *,
        rejected_event_count: int,
    ) -> None:
        event_list = list(events)
        with self.connect() as connection:
            for event in event_list:
                connection.execute(
                    """
                    INSERT INTO symbolic_events (
                        event_id, extraction_id, video_id, event_type,
                        start_time, end_time, actor, object_label,
                        source_location, destination_location, confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.event_id,
                        extraction_id,
                        event.video_id,
                        event.event_type,
                        event.start_time,
                        event.end_time,
                        event.actor,
                        event.object,
                        event.source_location,
                        event.destination_location,
                        event.confidence,
                    ),
                )
                connection.executemany(
                    """
                    INSERT INTO event_evidence (
                        event_id, timestamp_seconds, frame_id,
                        frame_number, frame_timestamp_seconds
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            event.event_id,
                            evidence.timestamp_seconds,
                            evidence.frame_id,
                            evidence.frame_number,
                            evidence.frame_timestamp_seconds,
                        )
                        for evidence in event.evidence
                    ],
                )
            connection.execute(
                """
                UPDATE event_extractions
                SET status = 'complete', event_count = ?,
                    rejected_event_count = ?, error = NULL,
                    completed_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (len(event_list), rejected_event_count, extraction_id),
            )

    def fail_event_extraction(self, extraction_id: str, error: str) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE event_extractions
                SET status = 'failed', error = ?, completed_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (error[:2000], extraction_id),
            )

    def get_latest_event_extraction(self, video_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            return self._latest_event_extraction(connection, video_id)

    def get_symbolic_events(self, video_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            return self._symbolic_events(connection, video_id)

    def _latest_event_extraction(
        self, connection: sqlite3.Connection, video_id: str
    ) -> dict[str, Any] | None:
        row = connection.execute(
            """
            SELECT id, provider, model, schema_version, status, event_count,
                   rejected_event_count, error, started_at, completed_at
            FROM event_extractions
            WHERE video_id = ?
            ORDER BY rowid DESC
            LIMIT 1
            """,
            (video_id,),
        ).fetchone()
        return dict(row) if row else None

    def _symbolic_events(
        self, connection: sqlite3.Connection, video_id: str
    ) -> list[dict[str, Any]]:
        extraction = connection.execute(
            """
            SELECT id FROM event_extractions
            WHERE video_id = ? AND status = 'complete'
            ORDER BY rowid DESC
            LIMIT 1
            """,
            (video_id,),
        ).fetchone()
        if extraction is None:
            return []

        rows = connection.execute(
            """
            SELECT event_id, event_type, start_time, end_time, actor,
                   object_label, source_location, destination_location,
                   confidence
            FROM symbolic_events
            WHERE extraction_id = ?
            ORDER BY start_time, end_time, event_id
            """,
            (extraction["id"],),
        ).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            evidence_rows = connection.execute(
                """
                SELECT timestamp_seconds, frame_id, frame_number,
                       frame_timestamp_seconds
                FROM event_evidence
                WHERE event_id = ?
                ORDER BY timestamp_seconds
                """,
                (row["event_id"],),
            ).fetchall()
            event = dict(row)
            event["object"] = event.pop("object_label")
            event["evidence"] = [dict(item) for item in evidence_rows]
            results.append(event)
        return results

    def list_object_labels(self) -> list[str]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT DISTINCT label FROM observations
                WHERE kind = 'object' ORDER BY label
                """
            ).fetchall()
            return [str(row["label"]) for row in rows]

    def find_latest_object_frame(self, label: str | None = None) -> dict[str, Any] | None:
        parameters: tuple[Any, ...] = ()
        label_filter = ""
        if label is not None:
            label_filter = "AND lower(target.label) = lower(?)"
            parameters = (label,)

        with self.connect() as connection:
            row = connection.execute(
                f"""
                SELECT
                    f.id AS frame_id,
                    f.video_id,
                    f.frame_number,
                    f.timestamp_seconds,
                    f.image_path,
                    v.original_name,
                    v.uploaded_at
                FROM frames f
                JOIN videos v ON v.id = f.video_id
                WHERE EXISTS (
                    SELECT 1 FROM observations target
                    WHERE target.frame_id = f.id
                      AND target.kind = 'object'
                      {label_filter}
                )
                ORDER BY v.rowid DESC, f.timestamp_seconds DESC
                LIMIT 1
                """,
                parameters,
            ).fetchone()
            return dict(row) if row else None

    def get_frame_objects(
        self, frame_id: int, label: str | None = None
    ) -> list[dict[str, Any]]:
        parameters: tuple[Any, ...] = (frame_id,)
        label_filter = ""
        if label is not None:
            label_filter = "AND lower(label) = lower(?)"
            parameters = (frame_id, label)

        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT id, label, confidence, details_json
                FROM observations
                WHERE frame_id = ? AND kind = 'object' {label_filter}
                ORDER BY confidence DESC, id
                """,
                parameters,
            ).fetchall()
            return [
                {
                    "id": row["id"],
                    "class": row["label"],
                    "confidence": row["confidence"],
                    "details": json.loads(row["details_json"]),
                }
                for row in rows
            ]

    def replace_frame_objects(self, frame_id: int, observations: Iterable[Any]) -> None:
        objects = [item for item in observations if item.kind == "object"]
        with self.connect() as connection:
            connection.execute(
                "DELETE FROM observations WHERE frame_id = ? AND kind = 'object'",
                (frame_id,),
            )
            connection.executemany(
                """
                INSERT INTO observations
                    (frame_id, kind, label, confidence, details_json)
                VALUES (?, 'object', ?, ?, ?)
                """,
                [
                    (
                        frame_id,
                        item.label,
                        item.confidence,
                        json.dumps(item.details),
                    )
                    for item in objects
                ],
            )

    def create_live_session(
        self,
        session_id: str,
        camera_source: str,
        started_at: str,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO live_sessions (id, camera_source, status, started_at)
                VALUES (?, ?, 'running', ?)
                """,
                (session_id, camera_source, started_at),
            )

    def update_live_session_frame(
        self,
        session_id: str,
        observed_at: str,
        *,
        dropped_frames: int,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE live_sessions
                SET last_frame_at = ?, dropped_frames = ?
                WHERE id = ?
                """,
                (observed_at, dropped_frames, session_id),
            )

    def finish_live_session(
        self,
        session_id: str,
        stopped_at: str,
        *,
        error: str | None = None,
    ) -> None:
        status = "failed" if error else "stopped"
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE live_sessions
                SET status = ?, stopped_at = ?, error = ?
                WHERE id = ?
                """,
                (status, stopped_at, error[:2000] if error else None, session_id),
            )

    def add_live_observation(
        self,
        *,
        session_id: str,
        observed_at: str,
        frame_path: Path | None,
        camera_state: str,
        person_present: bool | None,
        scene_clear: bool,
        motion_score: float,
        objects: Iterable[dict[str, Any]],
        facts: dict[str, Any],
    ) -> int:
        with self.connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO live_observations (
                    session_id, observed_at, frame_path, camera_state,
                    person_present, scene_clear, motion_score,
                    objects_json, facts_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    observed_at,
                    str(frame_path) if frame_path else None,
                    camera_state,
                    None if person_present is None else int(person_present),
                    int(scene_clear),
                    motion_score,
                    json.dumps(list(objects)),
                    json.dumps(facts),
                ),
            )
            assert cursor.lastrowid is not None
            return int(cursor.lastrowid)

    def create_activity_episode(
        self,
        *,
        episode_id: str,
        activity_type: str,
        display_name: str,
        person_name: str,
        confidence: float,
        started_at: str,
        observation_id: int,
        reason: str,
        evidence: dict[str, Any],
    ) -> dict[str, Any]:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO activity_episodes (
                    id, activity_type, display_name, person_name, state,
                    confidence, started_at, updated_at, last_observation_id
                ) VALUES (?, ?, ?, ?, 'ongoing', ?, ?, ?, ?)
                """,
                (
                    episode_id,
                    activity_type,
                    display_name,
                    person_name,
                    confidence,
                    started_at,
                    started_at,
                    observation_id,
                ),
            )
            connection.execute(
                """
                INSERT INTO episode_transitions (
                    episode_id, from_state, to_state, reason,
                    evidence_json, observation_id, created_at
                ) VALUES (?, NULL, 'ongoing', ?, ?, ?, ?)
                """,
                (episode_id, reason, json.dumps(evidence), observation_id, started_at),
            )
            connection.execute(
                """
                INSERT INTO episode_evidence (
                    episode_id, observation_id, role, reason
                ) VALUES (?, ?, 'activity_inference', ?)
                """,
                (episode_id, observation_id, reason),
            )
        episode = self.get_activity_episode(episode_id)
        assert episode is not None
        return episode

    def get_activity_episode(self, episode_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM activity_episodes WHERE id = ?", (episode_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_active_activity_episode(
        self, activity_type: str | None = None
    ) -> dict[str, Any] | None:
        parameters: tuple[Any, ...] = ()
        activity_filter = ""
        if activity_type is not None:
            activity_filter = "AND activity_type = ?"
            parameters = (activity_type,)
        with self.connect() as connection:
            row = connection.execute(
                f"""
                SELECT * FROM activity_episodes
                WHERE state NOT IN ('completed', 'dismissed') {activity_filter}
                ORDER BY rowid DESC LIMIT 1
                """,
                parameters,
            ).fetchone()
            return dict(row) if row else None

    def list_activity_episodes(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM activity_episodes
                ORDER BY updated_at DESC, rowid DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    def attach_episode_evidence(
        self,
        episode_id: str,
        observation_id: int,
        *,
        role: str,
        reason: str,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO episode_evidence (
                    episode_id, observation_id, role, reason
                ) VALUES (?, ?, ?, ?)
                """,
                (episode_id, observation_id, role, reason),
            )
            connection.execute(
                """
                UPDATE activity_episodes
                SET last_observation_id = ?
                WHERE id = ?
                """,
                (observation_id, episode_id),
            )

    def transition_activity_episode(
        self,
        episode_id: str,
        *,
        to_state: str,
        reason: str,
        created_at: str,
        evidence: dict[str, Any],
        observation_id: int | None = None,
        completion_source: str | None = None,
        absence_started_at: str | None = None,
        next_checkin_at: str | None = None,
        cooldown_until: str | None = None,
        increment_reminder: bool = False,
    ) -> dict[str, Any]:
        with self.connect() as connection:
            current = connection.execute(
                "SELECT state FROM activity_episodes WHERE id = ?", (episode_id,)
            ).fetchone()
            if current is None:
                raise KeyError(f"Unknown episode: {episode_id}")
            from_state = str(current["state"])
            terminal = to_state in {"completed", "dismissed"}
            connection.execute(
                """
                UPDATE activity_episodes
                SET state = ?, updated_at = ?,
                    completed_at = CASE WHEN ? THEN ? ELSE completed_at END,
                    completion_source = COALESCE(?, completion_source),
                    absence_started_at = ?, next_checkin_at = ?,
                    cooldown_until = ?,
                    reminder_count = reminder_count + ?,
                    last_observation_id = COALESCE(?, last_observation_id)
                WHERE id = ?
                """,
                (
                    to_state,
                    created_at,
                    int(terminal),
                    created_at,
                    completion_source,
                    absence_started_at,
                    next_checkin_at,
                    cooldown_until,
                    int(increment_reminder),
                    observation_id,
                    episode_id,
                ),
            )
            connection.execute(
                """
                INSERT INTO episode_transitions (
                    episode_id, from_state, to_state, reason,
                    evidence_json, observation_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode_id,
                    from_state,
                    to_state,
                    reason,
                    json.dumps(evidence),
                    observation_id,
                    created_at,
                ),
            )
            if observation_id is not None:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO episode_evidence (
                        episode_id, observation_id, role, reason
                    ) VALUES (?, ?, 'transition', ?)
                    """,
                    (episode_id, observation_id, reason),
                )
        episode = self.get_activity_episode(episode_id)
        assert episode is not None
        return episode

    def update_episode_absence(
        self,
        episode_id: str,
        *,
        updated_at: str,
        absence_started_at: str | None,
        observation_id: int | None = None,
    ) -> dict[str, Any]:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE activity_episodes
                SET updated_at = ?, absence_started_at = ?,
                    last_observation_id = COALESCE(?, last_observation_id)
                WHERE id = ?
                """,
                (updated_at, absence_started_at, observation_id, episode_id),
            )
        episode = self.get_activity_episode(episode_id)
        if episode is None:
            raise KeyError(f"Unknown episode: {episode_id}")
        return episode

    def create_checkin(
        self,
        *,
        checkin_id: str,
        episode_id: str,
        session_id: str | None,
        prompt: str,
        triggered_at: str,
    ) -> dict[str, Any]:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO checkins (
                    id, episode_id, session_id, prompt, status, triggered_at
                ) VALUES (?, ?, ?, ?, 'awaiting_response', ?)
                """,
                (checkin_id, episode_id, session_id, prompt, triggered_at),
            )
        result = self.get_checkin(checkin_id)
        assert result is not None
        return result

    def get_checkin(self, checkin_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM checkins WHERE id = ?", (checkin_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_latest_episode_checkin(
        self, episode_id: str
    ) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM checkins WHERE episode_id = ?
                ORDER BY triggered_at DESC, rowid DESC LIMIT 1
                """,
                (episode_id,),
            ).fetchone()
            return dict(row) if row else None

    def resolve_checkin(
        self,
        checkin_id: str,
        *,
        status: str,
        response_text: str | None,
        response_intent: str | None,
        responded_at: str,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE checkins
                SET status = ?, response_text = ?, response_intent = ?,
                    responded_at = ?
                WHERE id = ?
                """,
                (
                    status,
                    response_text,
                    response_intent,
                    responded_at,
                    checkin_id,
                ),
            )

    def get_episode_timeline(self, episode_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            transitions = connection.execute(
                """
                SELECT id, from_state, to_state, reason, evidence_json,
                       observation_id, created_at
                FROM episode_transitions
                WHERE episode_id = ? ORDER BY created_at, id
                """,
                (episode_id,),
            ).fetchall()
            return [
                {
                    **dict(row),
                    "evidence": json.loads(row["evidence_json"]),
                }
                for row in transitions
            ]

    def get_episode_evidence(self, episode_id: str) -> list[dict[str, Any]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT eo.role, eo.reason, lo.*
                FROM episode_evidence eo
                JOIN live_observations lo ON lo.id = eo.observation_id
                WHERE eo.episode_id = ?
                ORDER BY lo.observed_at, lo.id
                """,
                (episode_id,),
            ).fetchall()
            results: list[dict[str, Any]] = []
            for row in rows:
                item = dict(row)
                item["objects"] = json.loads(item.pop("objects_json"))
                item["facts"] = json.loads(item.pop("facts_json"))
                results.append(item)
            return results
