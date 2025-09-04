"""Transcription pipeline for audio chunks."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable, List
from transcription.whisper_transcriber import transcribe_file


def transcribe_chunk(path: Path, device: str = "cpu", model: str = "base") -> str:
    """Transcribe a single audio chunk using the Whisper model.

    Args:
        path (Path): Path to the audio file to transcribe.

    Replace the body of this function with calls to your actual
    speech-to-text engine. For now, it simply returns an empty string.
    """

    return transcribe_file(path, model=model, device=device)


def transcribe_chunks(
    chunks_dir: str | Path, device: str = "cpu", model: str = "base"
) -> List[str]:
    """Sequentially transcribe audio chunks located in ``chunks_dir``.

    Each chunk is processed one at a time to limit memory usage. After a chunk
    is transcribed its file is deleted. Once all chunks are processed the
    temporary directory is removed as well.

    Parameters
    ----------
    chunks_dir:
        Directory containing chunked audio files.

    Returns
    -------
    list[str]
        Transcribed text for each chunk in order.
    """

    chunks_path = Path(chunks_dir)
    transcripts: List[str] = []

    try:
        for chunk in sorted(chunks_path.glob("chunk_*.wav")):
            transcripts.append(transcribe_chunk(chunk, device=device, model=model))
            try:
                chunk.unlink()
            except FileNotFoundError:
                pass
    finally:
        shutil.rmtree(chunks_path, ignore_errors=True)

    return transcripts
