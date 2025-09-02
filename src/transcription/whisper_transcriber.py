"""Interface to transcribe audio using OpenAI's Whisper models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def transcribe_file(path: str | Path, model: str = "whisper-1", api_key: Optional[str] = None) -> str:
    """Transcribe an audio file using Whisper.

    The function first attempts to use the OpenAI Whisper API. If an API key is
    not available or the request fails, it falls back to the local
    ``whisper`` package (if installed).

    Parameters
    ----------
    path:
        Path to the audio file.
    model:
        Whisper model name. For the OpenAI API the default is ``"whisper-1"``.
        For the local fallback this translates to ``"base"`` unless another
        model name is provided.
    api_key:
        Optional explicit OpenAI API key. If ``None``, the ``OPENAI_API_KEY``
        environment variable is consulted.

    Returns
    -------
    str
        Transcribed text.
    """

    audio_path = Path(path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Try OpenAI API first
    try:
        import openai

        key = api_key or os.getenv("OPENAI_API_KEY")
        if key:
            openai.api_key = key
            with audio_path.open("rb") as handle:
                response = openai.Audio.transcribe(model=model, file=handle)
            text = response.get("text")
            if text is not None:
                return text
    except Exception:
        pass

    # Fallback to local whisper package
    try:
        import whisper
    except Exception as exc:
        raise RuntimeError(
            "Unable to transcribe audio: neither OpenAI API nor local whisper "
            "model is available"
        ) from exc

    local_model = model if model != "whisper-1" else "base"
    whisper_model = whisper.load_model(local_model)
    result = whisper_model.transcribe(str(audio_path))
    return result.get("text", "")
