#!/usr/bin/env python
"""Transcribe an audio file using the GPU-enabled pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project ``src`` directory is on the module search path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cli import run_pipeline  # pylint: disable=import-error


def main() -> int:
    parser = argparse.ArgumentParser(description="Transcribe audio on GPU")
    parser.add_argument("input", help="Path to a WAV audio file")
    parser.add_argument(
        "output",
        default="transcript.txt",
        nargs="?",
        help="Path for the generated transcript (default: transcript.txt)",
    )
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args = parser.parse_args()

    run_pipeline(args.input, args.output, device="cuda")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
