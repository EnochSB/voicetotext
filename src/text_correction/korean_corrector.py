"""Korean text post-processing utilities."""

from __future__ import annotations

from typing import Iterable, List


def correct_text(raw_text: str) -> str:
    """Return a grammatically corrected version of ``raw_text``.

    The function attempts to use ``language_tool_python`` to fix common
    spacing and spelling mistakes in Korean text. If the library is not
    available or an error occurs during correction, the original text is
    returned unchanged.
    """

    try:
        import language_tool_python
    except Exception:
        return raw_text

    try:
        tool = language_tool_python.LanguageTool("ko-KR")
        return tool.correct(raw_text)
    except Exception:
        return raw_text


def correct_chunks(chunks: Iterable[str]) -> str:
    """Correct multiple chunks and join them into a single string.

    Parameters
    ----------
    chunks:
        Iterable of raw transcription strings.

    Returns
    -------
    str
        The corrected transcription with chunks joined by a single space.
    """

    corrected: List[str] = []
    for chunk in chunks:
        if chunk:
            corrected.append(correct_text(chunk))
    return " ".join(corrected)
