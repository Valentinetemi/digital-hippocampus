"""Command-line entry point for Digital Hippocampus."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .perception import DEFAULT_YOLO_MODEL
from .pipeline import VideoPipeline
from .symbolic_events import DEFAULT_GEMINI_MODEL, GeminiSymbolicEventExtractor
from .server import run_server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Turn videos into timestamped, searchable observations."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory used for uploaded videos, extracted frames, and SQLite data.",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=1.0,
        help="Seconds between sampled frames (default: 1.0).",
    )
    model_options = parser.add_mutually_exclusive_group()
    model_options.add_argument(
        "--yolo-model",
        default=DEFAULT_YOLO_MODEL,
        help="Ultralytics model name or local path (default: yolo11n.pt).",
    )
    model_options.add_argument(
        "--no-yolo",
        action="store_const",
        const=None,
        dest="yolo_model",
        help="Disable object detection and use only basic visual observations.",
    )
    parser.add_argument(
        "--gemini-model",
        default=os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        help=f"Gemini model for symbolic events (default: {DEFAULT_GEMINI_MODEL}).",
    )
    parser.add_argument(
        "--no-symbolic-events",
        action="store_true",
        help="Skip Gemini event extraction even when GEMINI_API_KEY is set.",
    )
    parser.add_argument(
        "--camera-source",
        default=os.environ.get("DIGITAL_HIPPOCAMPUS_CAMERA_SOURCE", "0"),
        help=(
            "OpenCV camera index, stream URL, or video path for live observation "
            "(default: 0)."
        ),
    )
    parser.add_argument(
        "--person-name",
        default=os.environ.get("DIGITAL_HIPPOCAMPUS_PERSON_NAME", "Temi"),
        help="Name used in the grounded check-in prompt (default: Temi).",
    )
    parser.add_argument(
        "--perception-interval",
        type=float,
        default=float(
            os.environ.get("DIGITAL_HIPPOCAMPUS_PERCEPTION_INTERVAL", "0.75")
        ),
        help="Seconds between live temporal-perception samples (default: 0.75).",
    )

    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="Start the video upload web app.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    process = commands.add_parser("process", help="Process one local video.")
    process.add_argument("video", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.sample_interval <= 0:
        raise SystemExit("--sample-interval must be greater than zero")
    if args.perception_interval <= 0:
        raise SystemExit("--perception-interval must be greater than zero")

    event_extractor = None
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if gemini_api_key and not args.no_symbolic_events:
        event_extractor = GeminiSymbolicEventExtractor(
            gemini_api_key,
            model=args.gemini_model,
        )

    if args.command == "serve":
        run_server(
            host=args.host,
            port=args.port,
            data_dir=args.data_dir,
            sample_interval=args.sample_interval,
            yolo_model=args.yolo_model,
            event_extractor=event_extractor,
            camera_source=args.camera_source,
            person_name=args.person_name,
            perception_interval_seconds=args.perception_interval,
        )
        return

    pipeline = VideoPipeline(
        data_dir=args.data_dir,
        sample_interval=args.sample_interval,
        yolo_model=args.yolo_model,
        event_extractor=event_extractor,
    )
    result = pipeline.process(args.video)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
