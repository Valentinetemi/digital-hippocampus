"""Digital Hippocampus video-memory prototype."""

from typing import Any


__all__ = ["VideoPipeline"]


def __getattr__(name: str) -> Any:
    """Avoid loading OpenCV and the ML stack for database-only consumers."""

    if name == "VideoPipeline":
        from .pipeline import VideoPipeline

        return VideoPipeline
    raise AttributeError(name)
