"""Input normalisation: what an application can do to a text before a filter sees it.

The transforms in :mod:`jp_inject.transforms` rewrite how a payload is
written. This module goes the other way, with the means a defender has: no
knowledge of what the text is supposed to say, only rules about characters.
Two levels are defined, each including the one before:

``nfkc``
    Unicode NFKC, then removal of invisible format characters (category Cf).
    Folds full-width and half-width forms and drops zero-width characters.

``fold``
    ``nfkc``, then traditional kanji are replaced by their modern forms and
    cross-script look-alikes are repaired where the neighbouring characters
    show which script was meant (kanji ``口`` inside a katakana word becomes
    katakana ``ロ``).

Neither level can undo a payload written in kana or romaji, or change its
wording: that would take a kana-kanji converter, which guesses.

The ``fold`` tables are the inverse of the tables the transforms use. A
defender in the field does not know the attacker's table, so figures for
``fold`` are a best case for the defender.
"""

from __future__ import annotations

import unicodedata

from ._tables import KYUJITAI, _LOOKALIKE_PAIRS

__all__ = ["LEVELS", "fold", "nfkc", "normalize"]

LEVELS = ("nfkc", "fold")

_SHINJITAI_TABLE = str.maketrans({old: new for new, old in KYUJITAI.items()})

# Look-alike pairs by the scripts they join. In each pair the first member is katakana.
_KATAKANA_TO_KANJI = {a: b for a, b in _LOOKALIKE_PAIRS if "一" <= b <= "鿿"}
_KANJI_TO_KATAKANA = {b: a for a, b in _KATAKANA_TO_KANJI.items()}
_KATAKANA_TO_HIRAGANA = {a: b for a, b in _LOOKALIKE_PAIRS if "ぁ" <= b <= "ゖ"}
_HIRAGANA_TO_KATAKANA = {b: a for a, b in _KATAKANA_TO_HIRAGANA.items()}
_LONG_VOWEL, _KANJI_ONE = "ー", "一"


def _script(ch: str) -> str:
    """``"hiragana"``, ``"katakana"``, ``"kanji"`` or ``""`` for anything else."""
    if "ぁ" <= ch <= "ゖ":
        return "hiragana"
    if "ァ" <= ch <= "ヺ" or ch == _LONG_VOWEL:
        return "katakana"
    if "一" <= ch <= "鿿" or "㐀" <= ch <= "䶿" or ch == "々":
        return "kanji"
    return ""


def nfkc(text: str) -> str:
    """NFKC, then drop invisible format characters (Unicode category Cf)."""
    folded = unicodedata.normalize("NFKC", text)
    return "".join(ch for ch in folded if unicodedata.category(ch) != "Cf")


def _repair_lookalikes(text: str) -> str:
    """Rewrite look-alike characters to the script their neighbours are written in.

    A character is only changed when the neighbours leave little doubt:

    * a look-alike kanji next to katakana becomes katakana (``プ口ンプ卜``);
      kanji ``一`` becomes the long-vowel mark only directly after katakana;
    * a look-alike katakana that touches no other katakana but touches a
      kanji becomes that kanji (``協カ``);
    * a look-alike hiragana between two katakana, or opening a katakana word,
      becomes katakana (``へルプ``);
    * a look-alike katakana that touches no other katakana but touches
      hiragana becomes hiragana (``調ベて``).

    The rules are applied left to right, so a repaired character counts as
    its new script for the character after it. A word written entirely in
    look-alikes (``力夕`` for ``カタ``) gives the rules nothing to go on and
    is left as it is.
    """
    chars = list(text)
    for i, ch in enumerate(chars):
        before = _script(chars[i - 1]) if i else ""
        after = _script(chars[i + 1]) if i + 1 < len(chars) else ""
        around = (before, after)
        if ch == _KANJI_ONE:
            if before == "katakana":
                chars[i] = _LONG_VOWEL
        elif ch in _KANJI_TO_KATAKANA:
            if "katakana" in around:
                chars[i] = _KANJI_TO_KATAKANA[ch]
        elif ch in _HIRAGANA_TO_KATAKANA:
            if after == "katakana" and before in ("katakana", ""):
                chars[i] = _HIRAGANA_TO_KATAKANA[ch]
        elif "katakana" not in around:
            if ch in _KATAKANA_TO_KANJI and "kanji" in around:
                chars[i] = _KATAKANA_TO_KANJI[ch]
            elif ch in _KATAKANA_TO_HIRAGANA and "hiragana" in around:
                chars[i] = _KATAKANA_TO_HIRAGANA[ch]
    return "".join(chars)


def fold(text: str) -> str:
    """:func:`nfkc`, then modern kanji forms and look-alike repair."""
    return _repair_lookalikes(nfkc(text).translate(_SHINJITAI_TABLE))


_FUNCTIONS = {"nfkc": nfkc, "fold": fold}


def normalize(text: str, level: str | None) -> str:
    """Apply one normalisation level; ``None`` returns the text unchanged."""
    if level is None:
        return text
    try:
        return _FUNCTIONS[level](text)
    except KeyError:
        raise ValueError(f"unknown normalisation level {level!r}; choose from {', '.join(LEVELS)}") from None
