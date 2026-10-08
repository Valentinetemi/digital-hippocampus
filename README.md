# Digital Hippocampus

Digital Hippocampus is an experimental external memory system for the physical world.

I'm building it around a simple question:

> **What if AI could help preserve everyday context when human memory fails?**

For people living with memory impairment, forgetting is not always about facts. It can be about continuity.

Where did I leave this?

Did I already do that?

What happened a few minutes ago?

Was this object here before?

What changed while I wasn't paying attention?

The goal of Digital Hippocampus is to explore whether AI can act as a kind of **external episodic memory layer** - continuously observing the physical world, remembering important events and objects over time, and helping reconstruct context later.

The long-term vision is especially focused on supporting people with dementia and other forms of memory impairment.

This is not meant to replace human memory, diagnosis, treatment, or caregiving.

It is an attempt to build a system that can help answer:

> **"What happened?"**

when remembering becomes difficult.

And because a system like this could one day be relied on by real people, uncertainty matters.

If the system does not know, it should be able to say:

> **"I don't know."**

rather than confidently inventing a memory that never happened.

## Live milestone

The first live milestone now runs this pipeline:

```text
laptop camera / configurable OpenCV source
    -> responsive capture worker
    -> bounded two-frame queue (stale frames are discarded)
    -> asynchronous YOLO + visual perception worker
    -> timestamped observed facts and evidence frames in SQLite
    -> multi-frame tea-preparation inference
    -> persistent episode state and check-in policy
    -> browser speech + spoken/text responses
```

The browser shows the live feed, camera health, activity confidence, episode state,
transition timeline, evidence frames, and conversation. The voice surface is explicitly
labelled **Alexa-style simulation**; it does not connect to an Alexa device or service.

An episode can move through `ongoing`, `possibly_interrupted`, `awaiting_response`,
`snoozed`, `resumed`, `completed`, and `dismissed`. Completion records whether it was
visually inferred or user-confirmed. Camera failure and occlusion do not advance the
absence timer. Restarting preserves episode/check-in state without announcing an old
notification again.

### Setup and run

Python 3.11+ is required. Python 3.12 or 3.13 is recommended for the current ML stack;
see the measured Python 3.14 loader issue in [the live demo notes](docs/live-demo.md).

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
PYTHONPATH=src .venv/bin/python -m digital_hippocampus.main \
  --camera-source 0 \
  --person-name Temi \
  serve
```

Open <http://127.0.0.1:8000>, choose the explicitly labelled short demo timer, and
select **Start observation**. On macOS, grant camera access to the terminal or host app
running Python. The browser may separately request microphone access for speech input;
typed and quick-button responses remain available.

The camera source may be an OpenCV device index, stream URL, or video path:

```bash
PYTHONPATH=src .venv/bin/python -m digital_hippocampus.main \
  --camera-source rtsp://camera.local/live \
  serve
```

Live capture and the upload archive use the local `yolo11n.pt` checkpoint by default.
`--no-yolo` is useful for basic visual/archive checks, but live tea inference cannot
start without object facts. No API key is required for the live milestone.

### Environment variables

- `DIGITAL_HIPPOCAMPUS_CAMERA_SOURCE`: camera index, URL, or path; default `0`
- `DIGITAL_HIPPOCAMPUS_PERSON_NAME`: name used in the check-in; default `Temi`
- `DIGITAL_HIPPOCAMPUS_PERCEPTION_INTERVAL`: seconds between perception samples;
  default `0.75`
- `GEMINI_API_KEY`: optional; enables the older uploaded-video symbolic-event path
- `GEMINI_MODEL`: optional uploaded-video event model override

Keep secrets in the environment. `.env` is ignored, but the application does not load
it automatically. The live path stays local; enabling Gemini temporarily uploads only
an explicitly submitted archive video and deletes the remote file after extraction.

### Caregiver preview

The optional caregiver extension is disabled by default. When explicitly enabled in the
UI it requires a selected recipient, escalation delay, and notification limit. It writes
uncertainty-labelled messages to a local preview outbox only. This development build has
no external delivery adapter and rejects attempts to enable one.

### Uploaded-video archive

The original upload pipeline remains available under the expandable archive section:
timestamped frames, YOLO/ByteTrack entities, optional Gemini temporal events, and factual
object search. A local video can also be processed from the command line:

```bash
PYTHONPATH=src .venv/bin/python -m digital_hippocampus.main process path/to/video.mp4
```

### Test

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
node --check src/digital_hippocampus/static/app.js
```

The current suite contains 44 tests, including every requested check-in behavior,
bounded capture, occlusion/camera failure handling, persistence across restart, local
caregiver preview safety, legacy perception, symbolic events, and HTTP endpoints.

See [docs/live-demo.md](docs/live-demo.md) for the architecture, schema, measured
latency, known limitations, demo script, and genuine build friction.
