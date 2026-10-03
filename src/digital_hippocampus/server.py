"""HTTP API and static-file server for Digital Hippocampus."""

from __future__ import annotations

import json
import mimetypes
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .pipeline import VideoPipeline
from .symbolic_events import GeminiSymbolicEventExtractor


MAX_UPLOAD_BYTES = 500 * 1024 * 1024
MAX_QUESTION_BYTES = 16 * 1024
STATIC_DIR = Path(__file__).with_name("static")
STATIC_ROUTES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}


class VideoRequestHandler(BaseHTTPRequestHandler):
    pipeline: VideoPipeline

    def log_message(self, format: str, *args: object) -> None:
        print(f"{self.address_string()} - {format % args}")

    def send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        static_route = STATIC_ROUTES.get(path)
        if static_route is not None:
            filename, content_type = static_route
            self.serve_static_file(filename, content_type)
            return
        if path == "/api/videos":
            self.send_json(self.pipeline.database.list_videos())
            return
        if path.startswith("/api/videos/"):
            video_id = path.removeprefix("/api/videos/")
            video = self.pipeline.database.get_video(video_id)
            if video is None:
                self.send_json({"error": "Video not found"}, HTTPStatus.NOT_FOUND)
            else:
                self.send_json(video)
            return
        if path.startswith("/frames/"):
            self.serve_frame(path)
            return
        self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        request_path = urlparse(self.path).path
        if request_path == "/api/questions":
            self.answer_question()
            return
        if request_path != "/api/videos":
            self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return

        raw_length = self.headers.get("Content-Length")
        encoded_name = self.headers.get("X-Filename")
        if not raw_length or not encoded_name:
            self.send_json(
                {"error": "Content-Length and X-Filename headers are required"},
                HTTPStatus.BAD_REQUEST,
            )
            return
        try:
            content_length = int(raw_length)
        except ValueError:
            self.send_json({"error": "Invalid Content-Length"}, HTTPStatus.BAD_REQUEST)
            return
        if content_length <= 0 or content_length > MAX_UPLOAD_BYTES:
            self.send_json(
                {"error": "Video must be between 1 byte and 500 MB"},
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            )
            return

        original_name = Path(unquote(encoded_name)).name
        video_id = uuid.uuid4().hex
        try:
            target = self.pipeline.upload_path(video_id, original_name)
            remaining = content_length
            with target.open("wb") as output:
                while remaining:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ConnectionError(
                            "Upload ended before all bytes were received"
                        )
                    output.write(chunk)
                    remaining -= len(chunk)
            result = self.pipeline.process(
                target,
                original_name=original_name,
                video_id=video_id,
                source_is_staged=True,
            )
            self.send_json(result, HTTPStatus.CREATED)
        except ValueError as exc:
            if "target" in locals():
                target.unlink(missing_ok=True)
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self.send_json(
                {"error": f"Processing failed: {exc}"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def answer_question(self) -> None:
        raw_length = self.headers.get("Content-Length")
        try:
            content_length = int(raw_length or "0")
        except ValueError:
            content_length = 0
        if content_length <= 0 or content_length > MAX_QUESTION_BYTES:
            self.send_json(
                {"error": "Invalid question request"}, HTTPStatus.BAD_REQUEST
            )
            return
        try:
            payload = json.loads(self.rfile.read(content_length))
            question = payload.get("question", "") if isinstance(payload, dict) else ""
            result = self.pipeline.answer_question(str(question))
            self.send_json(result)
        except (json.JSONDecodeError, UnicodeDecodeError):
            self.send_json(
                {"error": "Request body must be valid JSON"}, HTTPStatus.BAD_REQUEST
            )
        except ValueError as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self.send_json(
                {"error": f"Could not search observations: {exc}"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def serve_static_file(self, filename: str, content_type: str) -> None:
        path = STATIC_DIR / filename
        if not path.is_file():
            self.send_json({"error": "Static file not found"}, HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve_frame(self, request_path: str) -> None:
        parts = request_path.strip("/").split("/")
        if len(parts) != 3:
            self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return
        _, video_id, filename = parts
        if not video_id.isalnum() or Path(filename).name != filename:
            self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return
        path = self.pipeline.frame_dir / video_id / filename
        if not path.is_file():
            self.send_json({"error": "Frame not found"}, HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header(
            "Content-Type", mimetypes.guess_type(path.name)[0] or "image/jpeg"
        )
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_server(
    *,
    host: str,
    port: int,
    data_dir: Path,
    sample_interval: float,
    yolo_model: Path | str | None,
    event_extractor: GeminiSymbolicEventExtractor | None,
) -> None:
    pipeline = VideoPipeline(
        data_dir,
        sample_interval,
        yolo_model,
        event_extractor,
    )
    handler = type(
        "ConfiguredVideoHandler", (VideoRequestHandler,), {"pipeline": pipeline}
    )
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Digital Hippocampus is running at http://{host}:{port}")
    print(f"Observations are stored in {pipeline.database.path}")
    if event_extractor is None:
        print("Gemini symbolic events are disabled (set GEMINI_API_KEY to enable).")
    else:
        print(f"Gemini symbolic events use {event_extractor.model}.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()
