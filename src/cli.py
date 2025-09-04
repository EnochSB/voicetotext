"""Command-line interface for the voice-to-text pipeline."""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from audio_processing.chunking import split_audio, split_by_silence
from transcription.pipeline import transcribe_chunk
from text_correction.korean_corrector import correct_chunks


LOGGER = logging.getLogger(__name__)


def run_pipeline(
    input_audio: str | Path,
    output_text: str | Path,
    chunk_length: int = 30,
    split_method: str = "fixed",
    device: str = "cpu",
    model: str = "base",
    temperature: float = 0.0,
    beam_size: int = 5,
) -> None:
    """Run the audio transcription pipeline and save results.

    Parameters
    ----------
    input_audio:
        Path to the source audio file.
    output_text:
        Destination path to write the transcribed text.
    chunk_length:
        Length of each audio chunk in seconds when using ``split_method='fixed'``.
        Defaults to 30.
    split_method:
        Strategy for splitting audio. ``"fixed"`` splits into equal-length
        chunks while ``"vad"`` uses voice activity detection to split on
        silence.
    device:
        Device to run the transcription model on (e.g. ``"cpu"`` or ``"cuda"``).
    model:
        Whisper model name to use for transcription.
    temperature:
        Sampling temperature for the Whisper decoder. Defaults to 0.0.
    beam_size:
        Beam size for beam search decoding. Defaults to 5.
    """

    input_audio = Path(input_audio)
    output_path = Path(output_text)

    if split_method == "vad":
        LOGGER.info("Splitting audio '%s' using VAD", input_audio)
        chunks = split_by_silence(input_audio)
    else:
        LOGGER.info(
            "Splitting audio '%s' into %s-second chunks", input_audio, chunk_length
        )
        chunks = split_audio(input_audio, chunk_length)
    LOGGER.info("Generated %d chunks", len(chunks))

    transcripts: list[str] = []
    total = len(chunks)
    LOGGER.info("Starting transcription of %d chunks", total)
    for idx, chunk in enumerate(sorted(chunks), start=1):
        LOGGER.info("Transcribing chunk %d/%d", idx, total)
        transcripts.append(
            transcribe_chunk(
                chunk,
                device=device,
                model=model,
                temperature=temperature,
                beam_size=beam_size,
            )
        )
        try:
            chunk.unlink()
        except FileNotFoundError:
            pass

    # Clean up temporary directory
    if chunks:
        shutil.rmtree(chunks[0].parent, ignore_errors=True)

    LOGGER.info("Correcting transcribed text")
    final_text = correct_chunks(transcripts)
    output_path.write_text(final_text, encoding="utf-8")
    LOGGER.info("Transcription written to '%s'", output_path)


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the CLI."""

    parser = argparse.ArgumentParser(description="Voice to text converter")
    parser.add_argument("input", help="Path to input audio file")
    parser.add_argument("output", help="Path to save transcribed text")
    parser.add_argument(
        "--chunk-length",
        type=int,
        default=30,
        help="Length of audio chunks in seconds when using fixed splitting (default: 30)",
    )
    parser.add_argument(
        "--split-method",
        choices=["fixed", "vad"],
        default="fixed",
        help="Audio splitting method: 'fixed' for equal lengths, 'vad' to split on silence",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device for local transcription model (default: cpu)",
    )
    parser.add_argument(
        "--model",
        default="base",
        help="Whisper model size to use (default: base)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for transcription (default: 0.0)",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size for beam search (default: 5)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the CLI."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)

    run_pipeline(
        args.input,
        args.output,
        args.chunk_length,
        args.split_method,
        args.device,
        args.model,
        args.temperature,
        args.beam_size,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
