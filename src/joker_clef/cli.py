"""Command line interface for the Joker Clef speech recogniser."""

from __future__ import annotations
import argparse

from .transcriber import transcribe_clip


def respond_to_clip(clip_path: str) -> str:
    """Transcribe a clip and format the Joker Clef response."""
    transcription = transcribe_clip(clip_path)
    return f"Joker Clef heard: {transcription}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe an audio clip and return a Joker Clef response",
    )
    parser.add_argument("clip", help="Path to the audio clip to transcribe")
    args = parser.parse_args(argv)
    print(respond_to_clip(args.clip))


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
