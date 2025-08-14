from __future__ import annotations

from joker_clef.cli import respond_to_clip


def test_respond_to_clip_uses_transcriber(monkeypatch):
    """Joker Clef round-trip should include the transcribed text."""
    def fake_transcribe(path: str) -> str:
        assert path == "dummy.wav"
        return "hello world"

    monkeypatch.setattr("joker_clef.cli.transcribe_clip", fake_transcribe)

    response = respond_to_clip("dummy.wav")
    assert response == "Joker Clef heard: hello world"
