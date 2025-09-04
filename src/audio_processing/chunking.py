"""Audio chunking utilities."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List

import webrtcvad
from pydub import AudioSegment


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


def split_by_silence(
    input_path: str | os.PathLike[str],
    vad_mode: int = 3,
    frame_duration: int = 30,
    min_silence_ms: int = 500,
) -> List[Path]:
    """Split an audio file at silence using ``webrtcvad``.

    The audio is first converted to 16kHz mono PCM before running voice
    activity detection. Periods of sustained silence (``min_silence_ms``)
    trigger the start of a new chunk.

    Parameters
    ----------
    input_path:
        Path to the source audio file.
    vad_mode:
        Aggressiveness of the VAD, from 0 (least) to 3 (most) aggressive.
    frame_duration:
        Duration of analysis frames in milliseconds. Must be 10, 20 or 30.
    min_silence_ms:
        Minimum length of silence required to split, in milliseconds.

    Returns
    -------
    list[pathlib.Path]
        Paths to the generated chunks located in a temporary directory.
    """

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)

    vad = webrtcvad.Vad(vad_mode)
    bytes_per_frame = int(audio.frame_rate * (frame_duration / 1000) * audio.sample_width)
    raw = audio.raw_data

    tmp_dir = Path(tempfile.mkdtemp(prefix="vad_chunks_"))
    chunks: List[Path] = []
    current = AudioSegment.empty()
    silence_buffer: list[AudioSegment] = []
    chunk_idx = 0

    for i in range(0, len(raw), bytes_per_frame):
        frame = raw[i : i + bytes_per_frame]
        if len(frame) < bytes_per_frame:
            break

        segment = AudioSegment(
            frame,
            frame_rate=audio.frame_rate,
            sample_width=audio.sample_width,
            channels=1,
        )
        is_speech = vad.is_speech(frame, audio.frame_rate)

        if is_speech:
            if silence_buffer:
                for buf in silence_buffer:
                    current += buf
                silence_buffer.clear()
            current += segment
        else:
            if len(current) == 0:
                continue
            silence_buffer.append(segment)
            total_silence = sum(len(s) for s in silence_buffer)
            if total_silence >= min_silence_ms:
                output = tmp_dir / f"chunk_{chunk_idx:03d}.wav"
                current.export(output, format="wav")
                chunks.append(output)
                chunk_idx += 1
                current = AudioSegment.empty()
                silence_buffer.clear()

    if len(current) > 0:
        output = tmp_dir / f"chunk_{chunk_idx:03d}.wav"
        current.export(output, format="wav")
        chunks.append(output)

    return chunks
