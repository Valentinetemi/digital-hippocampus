# Live demo notes

Status on October 8, 2026: the first live milestone is implemented for the October 18
readiness target. The Amazon-facing voice experience is a browser-based, clearly
labelled Alexa-style simulation.

## What the milestone does

1. A capture thread reads the configured OpenCV source and updates the MJPEG preview.
2. A bounded queue holds at most two frames. When perception falls behind, the oldest
   queued frame is discarded and the dropped-frame counter increases.
3. A separate worker samples the newest frame, runs the existing local perception
   layer, saves an evidence JPEG, and writes factual observations to SQLite.
4. `TemporalTeaRecognizer` evaluates a sequence rather than one detection. A start
   currently requires at least three time-separated samples containing a person, cup,
   preparation object, and motion.
5. `EpisodeCoordinator` updates the durable episode and policy state from each new
   observation. It counts absence only when the camera is available and the scene is
   clear.
6. A due check-in is saved before the UI sees it. The browser speaks that exact saved
   prompt and accepts speech, typed input, or quick-button responses.

Observed facts and inferred claims are stored separately. Object labels, person
presence/non-detection, camera health, motion score, timestamps, and evidence frames are
facts produced by the perception path. “Making tea,” “possibly interrupted,” and visual
completion are explicitly recorded inferences with confidence and reasons. The system
does not infer water temperature, appliance safety, or danger.

## Demo script

1. Start the server and open the dashboard.
2. Select **10 seconds — short demo timer**. This timer is deliberately labelled as a
   demo setting.
3. Select **Start observation**.
4. In a clear view, keep a person, cup, and flask/bottle visible while moving through
   the preparation for at least three perception samples.
5. Leave the clear camera view. After sustained usable non-detections reach the timer,
   the saved prompt is spoken: “Temi, I noticed you were making tea earlier. Are you
   coming back?”
6. Say or select one response:
   - “Give me another minute” moves the episode to `snoozed` for 60 seconds.
   - “I’m done” closes it with `completion_source = user_confirmed`.
   - “Cancel” closes it as `dismissed`.
   - Returning to the clear view cancels an unanswered check-in and records `resumed`.

The completion heuristic is intentionally conservative. It marks visual completion only
when the detected cup moves at least 16% of normalized frame size from its preparation
position and remains close to the visible person across two observations. The UI still
shows this as `visually_inferred`, not confirmed fact.

## Configuration

Global command-line options must appear before `serve`:

```bash
PYTHONPATH=src .venv/bin/python -m digital_hippocampus.main \
  --data-dir data \
  --camera-source 0 \
  --person-name Temi \
  --perception-interval 0.75 \
  --yolo-model yolo11n.pt \
  serve --host 127.0.0.1 --port 8000
```

Equivalent environment variables are:

| Variable | Purpose | Default |
| --- | --- | --- |
| `DIGITAL_HIPPOCAMPUS_CAMERA_SOURCE` | OpenCV index, URL, or path | `0` |
| `DIGITAL_HIPPOCAMPUS_PERSON_NAME` | Check-in name | `Temi` |
| `DIGITAL_HIPPOCAMPUS_PERCEPTION_INTERVAL` | Worker sample interval in seconds | `0.75` |
| `GEMINI_API_KEY` | Optional uploaded-video event extraction | unset |
| `GEMINI_MODEL` | Optional uploaded-video Gemini model | `gemini-3.8-flash` |

The live check-in, departure, snooze, cooldown, and caregiver-preview timings are sent
from the dashboard when observation starts. Credentials are not required by the live
pipeline and must not be committed.

## Persistence

SQLite remains the source of truth under `data/memory.db`:

- `live_sessions`: source, health lifecycle, last frame, and dropped-frame count
- `live_observations`: timestamped camera facts, objects, motion, and evidence path
- `activity_episodes`: current durable state, confidence, timers, completion source,
  and reminder count
- `episode_transitions`: immutable state changes with reasons and evidence snapshots
- `episode_evidence`: observed facts and inferences linked to the episode
- `checkins`: the exact saved prompt, originating session, response, and intent
- `caregiver_notifications`: local preview-only outbox entries and uncertainty notes

The current outstanding check-in stays visible after restart, but only check-ins whose
originating session matches the current live session are eligible for browser speech.
The browser also records spoken check-in IDs in session storage to avoid repeat speech
on refresh.

## Verification

Run:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
node --check src/digital_hippocampus/static/app.js
```

Result on October 8, 2026: **44 Python tests passed in 0.707 seconds** and the JavaScript
syntax check passed.

| Required behavior | Automated coverage |
| --- | --- |
| Interrupted tea triggers one check-in | `test_interrupted_tea_preparation_triggers_one_grounded_checkin` |
| Completed preparation avoids check-in | `test_visually_inferred_completion_prevents_checkin` |
| Return before threshold cancels pending path | `test_return_before_threshold_resumes_without_checkin` |
| Snooze postpones check-in | `test_snooze_postpones_the_next_checkin` |
| Completion/dismissal stop reminders | `test_user_completion_and_dismissal_are_terminal` |
| Occlusion/camera failure do not confirm departure | `test_occlusion_and_camera_failure_do_not_confirm_departure` |
| Restart does not replay an old notification | `test_restart_preserves_episode_without_replaying_old_checkin` |
| Capture discards stale frames | `test_bounded_queue_discards_the_oldest_stale_frame` |
| Caregiver path is delayed, limited, and preview-only | `test_preview_waits_for_delay_and_usable_absence_evidence` |

The repeatable tests feed timestamped factual observations or synthetic camera frames;
they are fixtures, not a scripted tea timer. Production episode creation still depends
on live multi-frame object and motion evidence.

## Measured latency

Measurements were taken in the repository's macOS ARM64 environment on October 8, 2026:

- Basic OpenCV signals without YOLO, 50 runs on the existing 1672×941 PNG: median
  **11.73 ms**, p95 **13.54 ms**, maximum **24.81 ms**.
- HTTP dashboard/status smoke test: successful against a real local server process.
- Real YOLO measurement: **no valid inference number was obtained**. The installed
  Python 3.14.7 environment remained in Torch dynamic-library mapping for more than four
  minutes before the benchmark was terminated. The stack sample showed the process in
  `dyld`/`mmap` while importing Torch dependencies, before model construction completed.
- End-to-end laptop-camera latency: not measured because the managed build sandbox does
  not expose camera hardware. At runtime the dashboard reports each actual worker pass as
  `Last inference`, measured across camera-state classification, perception, evidence
  write, episode update, and caregiver-policy evaluation.

Use Python 3.12 or 3.13 for the next hardware rehearsal, record cold model load and at
least 100 warm perception passes, and capture p50/p95 end-to-end values from the live
dashboard before the submission recording.

## Current limitations

- The stock COCO checkpoint usually represents a flask as `bottle` and has no reliable
  dedicated tea-action class. Lighting, distance, occlusion, and unusual vessels can
  cause missed object detections.
- Temporal inference is rule-based and currently supports one tea-preparation episode at
  a time. It is evidence-driven, but it is not a general action-recognition model.
- A sustained person non-detection is weaker than identity-aware departure. The policy
  reduces false alarms with timing and camera-health gates, but cannot eliminate them.
- Visual completion is an uncertain cup-carrying heuristic. User confirmation remains
  stronger and is stored separately.
- Camera reconnect behavior is basic. A failed source reports `unavailable`; restart
  observation after correcting the source.
- Speech synthesis and recognition depend on browser support. Text and quick responses
  are the reliable fallback.
- The local HTTP server is intended for a trusted demo machine, not an internet-facing
  deployment. It has no authentication, TLS, multi-user isolation, or retention controls.
- Evidence frames may contain sensitive household imagery. The milestone stores them
  locally and does not yet implement consent/retention deletion UX.
- Caregiver notification is preview-only. No email, SMS, Alexa, or other real message is
  sent.

## Genuine build friction

- The managed sandbox initially denied binding a localhost port. After explicit approval,
  the real server and status endpoint smoke test succeeded.
- The first full test attempt appeared stalled while importing optional ML dependencies.
  Lazy package imports were added so the database/live core no longer loads OpenCV,
  Gemini, or Pydantic unless needed; the full suite subsequently completed normally.
- The local YOLO/Torch stack on Python 3.14.7 did not finish dynamic-library mapping in
  more than four minutes. This prevented an honest warm-model latency number and is the
  main environment risk for the hardware rehearsal.
- The installed DevRelay research gateway required MLH authentication even for public
  community/challenge searches. The sign-in was interrupted, so no community findings
  were used to shape this milestone.
- Camera access is outside the managed sandbox, so physical webcam and browser microphone
  permission flows still require a local manual rehearsal.
