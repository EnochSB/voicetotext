"""Audio chunking utilities."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List


def split_audio(input_path: str | os.PathLike[str], chunk_length: int) -> List[Path]:
    """Split an audio file into fixed-length chunks using ``ffmpeg``.

    Parameters
    ----------
    input_path:
        Path to the source audio file.
    chunk_length:
        Length of each chunk in **seconds**.

    Returns
    -------
    list[pathlib.Path]
        A list of paths to the generated chunks. The returned paths live in a
        temporary directory which the caller is responsible for cleaning up
        once the chunks are no longer required.
    """

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="chunks_"))
    output_pattern = tmp_dir / "chunk_%03d.wav"

    cmd = [
        "ffmpeg",
        "-i",
        str(input_path),
        "-f",
        "segment",
        "-segment_time",
        str(chunk_length),
        "-c",
        "copy",
        str(output_pattern),
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError(f"ffmpeg failed to split audio: {exc.stderr.decode().strip()}") from exc

    chunks = sorted(tmp_dir.glob("chunk_*.wav"))
    return chunks
