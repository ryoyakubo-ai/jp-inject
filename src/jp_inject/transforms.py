"""Orthographic transforms for Japanese text (plus two comparison transforms).

Every transform here rewrites *how* a sentence is written without changing what
it says. They model what a Japanese keyboard and writing system make cheap for
an attacker:

* typing without kanji conversion (all hiragana), or with F7 / F8 / IME off
  (all katakana, half-width katakana, romaji);
* swapping modern kanji for their traditional, pre-reform forms (kyujitai);
* swapping characters for look-alikes that only exist because Japanese mixes
  three scripts (katakana ``ロ`` vs kanji ``口``);
* inserting zero-width characters, and writing ASCII in full-width forms.

Kana and romaji conversion needs to know how each kanji is read. Guessing
readings automatically is error-prone, so those transforms take a *reading*:
the sentence written in hiragana with words separated by ASCII spaces, e.g.::

    これ まで の しじ は すべて むし して ください。

Particles must be separate tokens so that ``は``/``へ``/``を`` can be romanised
as ``wa``/``e``/``o``.

Text inside ``{placeholders}`` is never modified, so a canary token survives
every transform unchanged.
"""

from __future__ import annotations

import re
from typing import Callable

from ._tables import (
    HALFWIDTH_KATAKANA,
    HALFWIDTH_PUNCT,
    KYUJITAI,
    LOOKALIKE,
    ROMAJI,
    ROMAJI_PUNCT,
)

__all__ = [
    "ZERO_WIDTH_SPACE",
    "count_substitutions",
    "insert_zero_width",
    "protect_placeholders",
    "reading_to_hankaku",
    "reading_to_hiragana",
    "reading_to_katakana",
    "reading_to_romaji",
    "to_fullwidth",
    "to_kyujitai",
    "to_lookalike",
]

ZERO_WIDTH_SPACE = "​"

_PLACEHOLDER = re.compile(r"(\{[a-z_]+\})")


def protect_placeholders(fn: Callable[[str], str]) -> Callable[[str], str]:
    """Wrap ``fn`` so that ``{placeholder}`` spans pass through untouched."""

    def wrapper(text: str, *args, **kwargs) -> str:
        parts = _PLACEHOLDER.split(text)
        return "".join(
            part if _PLACEHOLDER.fullmatch(part) else fn(part, *args, **kwargs)
            for part in parts
        )

    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


# --------------------------------------------------------------------------
# Transforms that work on the surface text
# --------------------------------------------------------------------------


@protect_placeholders
def to_fullwidth(text: str) -> str:
    """Rewrite printable ASCII as full-width forms (``Ignore`` -> ``Ｉｇｎｏｒｅ``).

    ASCII spaces become ideographic spaces (U+3000), as a Japanese IME in
    full-width alphanumeric mode would produce.
    """
    out = []
    for ch in text:
        code = ord(ch)
        if ch == " ":
            out.append("　")
        elif 0x21 <= code <= 0x7E:
            out.append(chr(code + 0xFEE0))
        else:
            out.append(ch)
    return "".join(out)


@protect_placeholders
def insert_zero_width(text: str, char: str = ZERO_WIDTH_SPACE, every: int = 1) -> str:
    """Insert an invisible character between visible characters.

    A zero-width character is placed after every ``every``-th character of each
    run of non-whitespace characters. The rendered text looks unchanged.
    """
    if every < 1:
        raise ValueError("every must be >= 1")
    out = []
    run = 0
    for i, ch in enumerate(text):
        out.append(ch)
        if ch.isspace():
            run = 0
            continue
        run += 1
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if nxt and not nxt.isspace() and run % every == 0:
            out.append(char)
    return "".join(out)


_KYUJITAI_TABLE = str.maketrans(KYUJITAI)
_LOOKALIKE_TABLE = str.maketrans(LOOKALIKE)


@protect_placeholders
def to_kyujitai(text: str) -> str:
    """Replace modern kanji (shinjitai) with their traditional forms (kyujitai).

    ``変更に対応`` -> ``變更に對應``. Readers of Japanese recognise both forms;
    a keyword list written in modern forms does not.
    """
    return text.translate(_KYUJITAI_TABLE)


@protect_placeholders
def to_lookalike(text: str) -> str:
    """Swap characters for visually similar ones from another Japanese script.

    Katakana ``ロ`` becomes kanji ``口``, kanji ``力`` becomes katakana ``カ``,
    the long-vowel mark ``ー`` becomes kanji ``一``, and so on.
    """
    return text.translate(_LOOKALIKE_TABLE)


def count_substitutions(text: str, table: dict[str, str]) -> int:
    """Number of characters in ``text`` (outside placeholders) that ``table`` maps."""
    visible = _PLACEHOLDER.sub("", text)
    return sum(1 for ch in visible if ch in table)


# --------------------------------------------------------------------------
# Transforms that work on a reading (space-separated hiragana)
# --------------------------------------------------------------------------

_HIRAGANA_START, _HIRAGANA_END = 0x3041, 0x3096
_KATAKANA_OFFSET = 0x60


def _is_hiragana(ch: str) -> bool:
    return _HIRAGANA_START <= ord(ch) <= _HIRAGANA_END


def _drop_token_spaces(reading: str) -> str:
    return reading.replace(" ", "")


def reading_to_hiragana(reading: str) -> str:
    """All-hiragana text, as typed without pressing the conversion key."""
    return _drop_token_spaces(reading)


@protect_placeholders
def _hira_to_kata(text: str) -> str:
    return "".join(
        chr(ord(ch) + _KATAKANA_OFFSET) if _is_hiragana(ch) else ch for ch in text
    )


def reading_to_katakana(reading: str) -> str:
    """All-katakana text (the F7 key on a Japanese IME)."""
    return _hira_to_kata(_drop_token_spaces(reading))


@protect_placeholders
def _kata_to_hankaku(text: str) -> str:
    out = []
    for ch in text:
        if ch in HALFWIDTH_KATAKANA:
            out.append(HALFWIDTH_KATAKANA[ch])
        elif ch in HALFWIDTH_PUNCT:
            out.append(HALFWIDTH_PUNCT[ch])
        else:
            out.append(ch)
    return "".join(out)


def reading_to_hankaku(reading: str) -> str:
    """All half-width katakana (the F8 key; common in legacy Japanese systems)."""
    return _kata_to_hankaku(reading_to_katakana(reading))


_PARTICLES = {"は": "wa", "へ": "e", "を": "o"}
_VOWELS = "aiueo"


_PUNCT_CLASS = "".join(re.escape(ch) for ch in ROMAJI_PUNCT)
_TOKEN_PARTS = re.compile(rf"([{_PUNCT_CLASS}]*)(.*?)([{_PUNCT_CLASS}]*)", re.S)


def _romanise_kana(kana: str) -> str:
    out: list[str] = []
    geminate = False
    i = 0
    while i < len(kana):
        ch = kana[i]
        pair = kana[i : i + 2]
        if ch == "っ":
            geminate = True
            i += 1
            continue
        if pair in ROMAJI:
            piece, step = ROMAJI[pair], 2
        elif ch in ROMAJI:
            piece, step = ROMAJI[ch], 1
        elif ch == "ん":
            nxt = kana[i + 1 : i + 3]
            nxt_roma = ROMAJI.get(nxt) or ROMAJI.get(nxt[:1], "")
            piece = "n'" if nxt_roma[:1] and nxt_roma[0] in _VOWELS + "y" else "n"
            step = 1
        elif ch == "ー":
            last = next((c for c in reversed("".join(out)) if c in _VOWELS), "")
            piece, step = last, 1
        elif ch in ROMAJI_PUNCT:
            piece, step = ROMAJI_PUNCT[ch], 1
        else:
            piece, step = ch, 1

        if geminate and piece and piece[0].isalpha() and piece[0] not in _VOWELS:
            piece = ("t" if piece.startswith("ch") else piece[0]) + piece
        geminate = False
        out.append(piece)
        i += step
    return "".join(out)


def _romanise_token(token: str) -> str:
    """Romanise one token of a reading (modified Hepburn, long vowels spelled out)."""
    lead, core, trail = _TOKEN_PARTS.fullmatch(token).groups()
    roma = _PARTICLES[core] if core in _PARTICLES else _romanise_kana(core)
    return _romanise_kana(lead) + roma + _romanise_kana(trail)


@protect_placeholders
def _romanise_segment(segment: str) -> str:
    lines = []
    for line in segment.split("\n"):
        lines.append(" ".join(_romanise_token(tok) for tok in line.split(" ")))
    return "\n".join(lines)


def reading_to_romaji(reading: str) -> str:
    """Romaji (Latin letters), as typed with the IME switched off.

    Word spaces from the reading are kept, because romaji is conventionally
    written with spaces. Japanese punctuation becomes its ASCII counterpart.
    """
    text = _romanise_segment(reading)
    # Tidy spacing that tokenisation leaves around quotes and brackets.
    text = re.sub(r'"\s+(\{[a-z_]+\})\s+"', r'"\1"', text)
    text = re.sub(r"([(\[<])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]>.,:?!])", r"\1", text)
    return text
