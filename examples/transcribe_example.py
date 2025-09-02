"""Simple demonstration of the transcription pipeline.

This script accepts a path to an audio file on the command line. If no
path is provided a small sample file will be downloaded at runtime.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import urllib.request
from pathlib import Path

# Ensure project ``src`` directory is on the module search path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cli import run_pipeline  # pylint: disable=import-error


def download_sample() -> Path:
    """Download a short sample audio file and return its path."""
    url = "https://raw.githubusercontent.com/ggerganov/whisper.cpp/master/samples/jfk.wav"

    tmp_dir = Path(tempfile.mkdtemp())
    sample_path = tmp_dir / "sample.wav"
    urllib.request.urlretrieve(url, sample_path)  # noqa: S310
    return sample_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Transcribe an audio file")
    parser.add_argument(
        "--audio",
        help="Path to a WAV audio file. If omitted, a sample will be downloaded.",
    )
    parser.add_argument(
        "--output",
        default="transcript.txt",
        help="Path for the generated transcript (default: transcript.txt)",
    )
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

    args = parser.parse_args()

    if args.audio:
        audio_path = Path(args.audio)
        if not audio_path.exists():
            parser.error(f"Audio file not found: {audio_path}")
    else:
        audio_path = download_sample()
        logger.info("Downloaded sample audio to %s", audio_path)

    logger.info("Starting transcription pipeline")
    run_pipeline(audio_path, args.output)
    logger.info("Transcript written to %s", args.output)

    if not args.audio:
        # Clean up downloaded sample
        try:
            audio_path.unlink()
            audio_path.parent.rmdir()
        except OSError:
            pass

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
