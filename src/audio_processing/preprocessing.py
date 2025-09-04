from __future__ import annotations

"""Audio preprocessing utilities."""

from pathlib import Path
import os
import tempfile

import numpy as np
from pydub import AudioSegment
import noisereduce as nr


def preprocess_audio(input_path: str | Path) -> Path:
    """Apply noise reduction and volume normalization to an audio file.

    Parameters
    ----------
    input_path:
        Path to the source audio file.

    Returns
    -------
    pathlib.Path
        Path to a temporary preprocessed WAV file. Caller is responsible for
        deleting the file when finished.
    """
    input_path = Path(input_path)
    audio = AudioSegment.from_file(input_path).set_channels(1)

    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    reduced = nr.reduce_noise(y=samples, sr=audio.frame_rate)

    processed = AudioSegment(
        reduced.tobytes(),
        frame_rate=audio.frame_rate,
        sample_width=audio.sample_width,
        channels=1,
    )
    processed = processed.apply_gain(-processed.max_dBFS)

    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    tmp_path = Path(tmp)
    processed.export(tmp_path, format="wav")
    return tmp_path
