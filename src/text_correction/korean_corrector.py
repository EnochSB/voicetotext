"""Korean text post-processing utilities."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple


def _mask_user_words(text: str, words: Iterable[str]) -> Tuple[str, Dict[str, str]]:
    """Replace ``words`` in ``text`` with placeholders.

    The placeholders allow the spell checker to run without altering
    domain-specific terms supplied by the user.
    """

    placeholders: Dict[str, str] = {}
    for i, word in enumerate(words):
        placeholder = f"__USERWORD_{i}__"
        placeholders[placeholder] = word
        text = text.replace(word, placeholder)
    return text, placeholders


def _unmask_user_words(text: str, placeholders: Dict[str, str]) -> str:
    """Restore placeholders inserted by :func:`_mask_user_words`."""

    for placeholder, word in placeholders.items():
        text = text.replace(placeholder, word)
    return text


def correct_text(
    raw_text: str,
    secondary: str | None = None,
    user_dict: Iterable[str] | None = None,
) -> str:
    """Return a grammatically corrected version of ``raw_text``.

    Parameters
    ----------
    raw_text:
        Text to correct.
    secondary:
        Name of an optional secondary corrector (``"hanspell"`` or
        ``"kospacing"``).
    user_dict:
        Iterable of words that should be preserved during correction.

    The function attempts to use ``language_tool_python`` to fix common spacing
    and spelling mistakes in Korean text. Optional secondary correctors are
    applied afterwards. If any library is unavailable or an error occurs during
    correction, the original text is returned unchanged.
    """

    text = raw_text
    placeholders: Dict[str, str] = {}

    if user_dict:
        text, placeholders = _mask_user_words(text, user_dict)

    try:
        import language_tool_python

        tool = language_tool_python.LanguageTool("ko-KR")
        corrected = tool.correct(text)
    except Exception:
        corrected = text

    if secondary == "hanspell":
        try:
            from hanspell import spell_checker

            corrected = spell_checker.check(corrected).checked
        except Exception:
            pass
    elif secondary == "kospacing":
        try:
            from kospacing import KoSpacing

            spacing = KoSpacing()
            corrected = spacing(corrected)
        except Exception:
            pass

    if placeholders:
        corrected = _unmask_user_words(corrected, placeholders)

    return corrected


def correct_chunks(
    chunks: Iterable[str],
    secondary: str | None = None,
    user_dict: Iterable[str] | None = None,
) -> str:
    """Correct multiple chunks and join them into a single string.

    Parameters
    ----------
    chunks:
        Iterable of raw transcription strings.
    secondary:
        Optional secondary corrector passed to :func:`correct_text`.
    user_dict:
        Words that should remain unchanged during correction.

    Returns
    -------
    str
        The corrected transcription with chunks joined by a single space.
    """

    corrected: List[str] = []
    for chunk in chunks:
        if chunk:
            corrected.append(
                correct_text(chunk, secondary=secondary, user_dict=user_dict)
            )
    return " ".join(corrected)
