"""Interface to transcribe audio using local Whisper models."""

from __future__ import annotations

from pathlib import Path


def transcribe_file(
    path: str | Path,
    model: str = "base",
    device: str = "cpu",
    **kwargs,
) -> str:
    """Transcribe an audio file using a local Whisper model.

    Parameters
    ----------
    path:
        Path to the audio file.
    model:
        Whisper model name, e.g. ``"base"``.
    device:
        Device to run the model on, e.g. ``"cpu"`` or ``"cuda"``.
    **kwargs:
        Additional keyword arguments passed to ``whisper_model.transcribe``.

    Returns
    -------
    str
        Transcribed text.
    """

    audio_path = Path(path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        import whisper
    except Exception as exc:
        raise RuntimeError(
            "Unable to transcribe audio: the 'whisper' package is not available"
        ) from exc

    whisper_model = whisper.load_model(model, device=device)
    result = whisper_model.transcribe(str(audio_path), **kwargs)
    return result.get("text", "")
