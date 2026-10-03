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

## First working version

The current pipeline is:

```text
uploaded video
    -> timestamped frame samples
    -> perception (YOLO11 Nano objects, light, focus, color, motion)
    -> observations stored in SQLite
    -> Gemini temporal event extraction (when configured)
    -> validated symbolic events stored in SQLite
    -> browser timeline
```

### Run it

The existing virtual environment already contains OpenCV and NumPy. From the project
root, start the upload app with:

```bash
PYTHONPATH=src .venv/bin/python -m digital_hippocampus.main serve
```

Then open <http://127.0.0.1:8000>, choose a video, and select **Upload and observe**.
Uploaded videos, extracted frames, and the SQLite database are written under `data/`.

To enable Gemini symbolic-event extraction, set an API key before starting the app:

```bash
export GEMINI_API_KEY="your-key"
PYTHONPATH=src .venv/bin/python -m digital_hippocampus.main serve
```

The default event model is `gemini-3.8-flash`. Override it with `GEMINI_MODEL` or
`--gemini-model`. Without `GEMINI_API_KEY`, the existing local perception pipeline runs
normally and event extraction is skipped.

When Gemini extraction is enabled, the original video is temporarily uploaded to the
Gemini Files API. The application requests structured JSON events and deletes the remote
file after the request finishes. Locally stored video, frames, observations, and events
remain under `data/`.

You can also process a video without the browser:

```bash
PYTHONPATH=src .venv/bin/python -m digital_hippocampus.main process path/to/video.mp4
```

The perception layer uses the Ultralytics YOLO11 Nano detection model (`yolo11n.pt`) by
default. Ultralytics downloads the checkpoint on the first run; after that, inference is
local and does not need an API key. You can also provide another model name or local
checkpoint:

```bash
PYTHONPATH=src .venv/bin/python -m digital_hippocampus.main \
  --yolo-model path/to/model.pt serve
```

To run only the basic visual observations without YOLO:

```bash
PYTHONPATH=src .venv/bin/python -m digital_hippocampus.main --no-yolo serve
```

### Data model

- `videos`: upload metadata and processing state
- `frames`: sampled frame number, timestamp, and image path
- `observations`: typed perception result attached to a frame
- `event_extractions`: Gemini extraction status, model, schema version, and errors
- `symbolic_events`: validated actions and changes across time
- `event_evidence`: Gemini evidence timestamps linked to the nearest sampled frames

Symbolic event types are currently limited to `picked_up`, `placed`, `moved`, `opened`,
`closed`, `entered`, and `exited`. Every accepted event must have at least one evidence
timestamp inside the video duration. Multiple timestamps are preferred but not required;
the original video remains the primary temporal evidence. Event IDs are assigned by the
application, and nearest-frame links are resolved deterministically rather than by Gemini.

### Ask the archive

The browser includes an **Ask the archive** field. It currently supports factual,
object-based questions such as:

- `What was the last observation?`
- `Where was the chair last seen?`
- `When was a person last seen?`

The answer includes the source video, timestamp, YOLO class, confidence, and bounding
box as pixel coordinates (`x`, `y`, `width`, `height`) measured from the frame's
top-left corner. This first version uses local class matching rather than a language
model, so it only answers from stored object detections and does not invent missing
events.

Run the automated pipeline checks with:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```
