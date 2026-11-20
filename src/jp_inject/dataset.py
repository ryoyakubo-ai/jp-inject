"""Build and load the jp-inject test cases.

The source texts live in ``data/attacks.toml`` and
``data/scenarios.toml``. :func:`build_cases` composes them into concrete test
cases; the result is frozen in ``data/cases.jsonl`` so that everybody runs
exactly the same prompts. ``jp-inject build --check`` (and the test suite)
verify that the frozen file still matches the sources.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Any, Iterable

from . import transforms as T
from ._tables import KYUJITAI, LOOKALIKE

DATA_PACKAGE = "jp_inject.data"
CASES_FILE = "cases.jsonl"
LICENSE = "Apache-2.0"
PROVENANCE = "original (written for jp-inject)"

DEFENSE_PROFILES = ("basic", "none")


@dataclass(frozen=True)
class Variant:
    """One way of writing an attack payload.

    ``reference`` is the variant this one is compared against (same language,
    unmarked form). ``counterpart`` is the same manipulation in the other
    language, when one exists; it lets the analysis ask whether an effect is
    specific to Japanese or simply an effect of the manipulation.
    """

    id: str
    lang: str
    family: str  # language | register | framing | orthographic
    technique: str
    description: str
    reference: str | None = None
    counterpart: str | None = None
    also_compare: str | None = None  # an extra, more specific baseline


VARIANTS: tuple[Variant, ...] = (
    Variant("ja.plain", "ja", "language", "plain",
            "Japanese, polite (desu/masu) form; the unmarked way to write a request."),
    Variant("en.plain", "en", "language", "plain",
            "English, direct imperative; the unmarked way to write a prompt."),
    # --- register -----------------------------------------------------------
    Variant("ja.blunt", "ja", "register", "blunt",
            "Plain / command form (shiro, da).", "ja.plain"),
    Variant("ja.keigo", "ja", "register", "keigo",
            "Business honorific language (keigo).", "ja.plain", "en.polite"),
    Variant("ja.indirect", "ja", "register", "indirect",
            "Hedged, roundabout request (enkyoku).", "ja.plain", "en.indirect"),
    Variant("ja.mixed", "ja", "register", "mixed",
            "Japanese sentence with English key words (code-switching).", "ja.plain"),
    Variant("en.polite", "en", "register", "polite",
            "Elaborately polite English.", "en.plain", "ja.keigo"),
    Variant("en.indirect", "en", "register", "indirect",
            "Hedged, roundabout English request.", "en.plain", "ja.indirect"),
    # --- framing ------------------------------------------------------------
    Variant("ja.authority", "ja", "framing", "authority",
            "Official internal notice from an administrative department, in keigo.",
            "ja.plain", "en.authority", "ja.keigo"),
    Variant("ja.note", "ja", "framing", "note",
            "Annotation introduced by the kome-jirushi mark, as if part of the document.",
            "ja.plain", "en.note"),
    Variant("en.authority", "en", "framing", "authority",
            "Official notice from an administrative department, in polite English.",
            "en.plain", "ja.authority", "en.polite"),
    Variant("en.note", "en", "framing", "note",
            "Annotation introduced by an asterisk, as if part of the document.",
            "en.plain", "ja.note"),
    # --- orthographic -------------------------------------------------------
    Variant("ja.hiragana", "ja", "orthographic", "hiragana",
            "All hiragana, no kanji (typed without conversion).", "ja.plain"),
    Variant("ja.katakana", "ja", "orthographic", "katakana",
            "All full-width katakana.", "ja.plain"),
    Variant("ja.hankaku", "ja", "orthographic", "hankaku",
            "All half-width katakana.", "ja.plain"),
    Variant("ja.romaji", "ja", "orthographic", "romaji",
            "Romanised Japanese in Latin letters.", "ja.plain"),
    Variant("ja.kyujitai", "ja", "orthographic", "kyujitai",
            "Modern kanji replaced by traditional forms.", "ja.plain"),
    Variant("ja.lookalike", "ja", "orthographic", "lookalike",
            "Characters swapped for cross-script look-alikes.", "ja.plain"),
    Variant("ja.zero_width", "ja", "orthographic", "zero_width",
            "Zero-width space between every character.", "ja.plain", "en.zero_width"),
    Variant("en.fullwidth", "en", "orthographic", "fullwidth",
            "English written in full-width alphanumerics.", "en.plain"),
    Variant("en.zero_width", "en", "orthographic", "zero_width",
            "Zero-width space between every character.", "en.plain", "ja.zero_width"),
)
VARIANT_BY_ID = {v.id: v for v in VARIANTS}

# Plain payloads are also run in the other-language application, to complete
# the 2x2 of payload language x application language.
CROSS_ENV = (("en.plain", "ja"), ("ja.plain", "en"))

# Orthographic variants that get a comprehension control.
_COMPREHENSION_VARIANTS = (
    "ja.plain", "en.plain",
    *(v.id for v in VARIANTS if v.family == "orthographic"),
)


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------


def _read_text(name: str) -> str:
    return resources.files(DATA_PACKAGE).joinpath(name).read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def load_sources() -> dict[str, Any]:
    attacks = tomllib.loads(_read_text("attacks.toml"))
    scenarios = tomllib.loads(_read_text("scenarios.toml"))
    return {**attacks, **scenarios}


def _strip(text: str) -> str:
    return text.strip("\n")


def system_prompt(scenario: str, env: str, defense: str = "basic") -> str:
    """Return the system prompt of ``scenario`` in ``env`` for a defense profile."""
    if defense not in DEFENSE_PROFILES:
        raise ValueError(f"unknown defense profile: {defense!r}")
    src = load_sources()
    spec = src["scenarios"][scenario][env]
    line = spec.get("defense", "") if defense == "basic" else ""
    text = spec["system"].replace("{defense}", line)
    return _strip(text.replace("{secret}", src["secret"]["value"]))


# --------------------------------------------------------------------------
# Payload composition
# --------------------------------------------------------------------------


def _compose(strategy: dict, target: dict, key: str) -> str:
    """Lead + ask in one register (``key`` is e.g. ``ja.keigo``)."""
    return f"{strategy['lead'][key]}\n{target['ask'][key]}"


def compose_payload(strategy_id: str, target_id: str, variant_id: str) -> str:
    """Return the payload template (``{canary}`` not yet filled in)."""
    src = load_sources()
    strategy = src["strategies"][strategy_id]
    target = src["targets"][target_id]
    variant = VARIANT_BY_ID[variant_id]
    lang, technique = variant.lang, variant.technique

    def composed(key: str) -> str:
        return _compose(strategy, target, key)

    if variant.family in ("language", "register"):
        return composed(variant_id)
    if variant.family == "framing":
        body_key = {
            ("ja", "authority"): "ja.keigo",
            ("en", "authority"): "en.polite",
            ("ja", "note"): "ja.plain",
            ("en", "note"): "en.plain",
        }[(lang, technique)]
        return src["framings"][technique][lang].replace("{body}", composed(body_key))

    plain = composed(f"{lang}.plain")
    if technique == "zero_width":
        return T.insert_zero_width(plain)
    if technique == "fullwidth":
        return T.to_fullwidth(plain)
    if technique == "kyujitai":
        return T.to_kyujitai(plain)
    if technique == "lookalike":
        return T.to_lookalike(plain)
    reading = composed("ja.reading")
    return _apply_reading_transform(technique, reading)


def _apply_reading_transform(technique: str, reading: str) -> str:
    fn = {
        "hiragana": T.reading_to_hiragana,
        "katakana": T.reading_to_katakana,
        "hankaku": T.reading_to_hankaku,
        "romaji": T.reading_to_romaji,
    }[technique]
    return fn(reading)


def _comprehension_text(item: dict, variant_id: str) -> str:
    variant = VARIANT_BY_ID[variant_id]
    if variant.family == "language":
        return item[variant.lang]
    plain = item[variant.lang]
    technique = variant.technique
    if technique == "zero_width":
        return T.insert_zero_width(plain)
    if technique == "fullwidth":
        return T.to_fullwidth(plain)
    if technique == "kyujitai":
        return T.to_kyujitai(plain)
    if technique == "lookalike":
        return T.to_lookalike(plain)
    return _apply_reading_transform(technique, item["reading"])


# --------------------------------------------------------------------------
# Case construction
# --------------------------------------------------------------------------


def _changed_chars(variant: Variant, plain: str) -> int | None:
    """How many characters a table-based transform actually replaced."""
    table = {"kyujitai": KYUJITAI, "lookalike": LOOKALIKE}.get(variant.technique)
    if variant.family != "orthographic" or table is None:
        return None
    return T.count_substitutions(plain, table)


def build_cases() -> list[dict[str, Any]]:
    """Compose every test case from the sources, in a stable order."""
    src = load_sources()
    secret = src["secret"]["value"]
    strategies = list(src["strategies"])
    cases: list[dict[str, Any]] = []

    def base_fields(scenario: str, env: str) -> dict[str, Any]:
        return {
            "scenario": scenario,
            "env": env,
            "system": system_prompt(scenario, env, "basic"),
            "system_no_defense": system_prompt(scenario, env, "none"),
            "license": LICENSE,
            "provenance": PROVENANCE,
        }

    # ---- attacks -----------------------------------------------------------
    for target_id, target in src["targets"].items():
        scenario = target["scenario"]
        canary = target.get("canary")
        for index, strategy_id in enumerate(strategies):
            runs = [(v.id, v.lang) for v in VARIANTS] + list(CROSS_ENV)
            for variant_id, env in runs:
                variant = VARIANT_BY_ID[variant_id]
                template = compose_payload(strategy_id, target_id, variant_id)
                payload = template.replace("{canary}", canary or "")
                spec = src["scenarios"][scenario][env]
                carrier = _strip(spec["carriers"][index])
                user = spec["user"].replace("{carrier}", carrier).replace("{payload}", payload)
                cross = env != variant.lang
                plain_template = compose_payload(strategy_id, target_id, f"{variant.lang}.plain")
                case = {
                    "id": f"attack/{target_id}/{strategy_id}/{variant_id}"
                    + (f"@{env}" if cross else ""),
                    "kind": "attack",
                    "target": target_id,
                    "goal": target["goal"],
                    "channel": target["channel"],
                    "strategy": strategy_id,
                    "base": f"{target_id}/{strategy_id}",
                    "variant": variant_id + (f"@{env}" if cross else ""),
                    "lang": variant.lang,
                    "family": "cross_env" if cross else variant.family,
                    "technique": variant.technique,
                    **base_fields(scenario, env),
                    "carrier": index,
                    "payload": payload,
                    "user": user,
                    "expect": "secret" if target["goal"] == "leak" else "canary",
                    "token": secret if target["goal"] == "leak" else canary,
                    "strategy_related": src["strategies"][strategy_id]["related"],
                }
                changed = _changed_chars(variant, plain_template)
                if changed is not None:
                    case["substitutions"] = changed
                cases.append(case)

    # ---- benign controls: the carrier alone, no injection --------------------
    all_tokens = sorted(
        {secret} | {t["canary"] for t in src["targets"].values() if "canary" in t}
    )
    for scenario in ("support", "summarize", "translate"):
        for env in ("ja", "en"):
            spec = src["scenarios"][scenario][env]
            for index, carrier in enumerate(spec["carriers"]):
                cases.append({
                    "id": f"benign/{scenario}/{index}/{env}",
                    "kind": "benign",
                    "target": None,
                    "goal": None,
                    "channel": src["scenarios"][scenario]["channel"],
                    "strategy": None,
                    "base": f"benign/{scenario}/{index}",
                    "variant": f"{env}.none",
                    "lang": env,
                    "family": "control",
                    "technique": "none",
                    **base_fields(scenario, env),
                    "carrier": index,
                    "payload": "",
                    "user": spec["user_benign"].replace("{carrier}", _strip(carrier)),
                    "expect": "clean",
                    "token": None,
                    "forbidden_tokens": all_tokens,
                })

    # ---- comprehension controls ------------------------------------------------
    for item in src["comprehension"]:
        for variant_id in _COMPREHENSION_VARIANTS:
            variant = VARIANT_BY_ID[variant_id]
            text = _comprehension_text(item, variant_id).replace("{marker}", item["marker"])
            cases.append({
                "id": f"comprehension/{item['id']}/{variant_id}",
                "kind": "comprehension",
                "target": None,
                "goal": "comply",
                "channel": "direct",
                "strategy": None,
                "base": f"comprehension/{item['id']}",
                "variant": variant_id,
                "lang": variant.lang,
                "family": variant.family,
                "technique": variant.technique,
                **base_fields("comprehension", variant.lang),
                "carrier": None,
                "payload": text,
                "user": text,
                "expect": "marker",
                "token": item["marker"],
            })
    return cases


# --------------------------------------------------------------------------
# Frozen file
# --------------------------------------------------------------------------


def dump_cases(cases: Iterable[dict[str, Any]]) -> str:
    """Serialise cases as JSON Lines (UTF-8, one case per line, stable keys)."""
    return "".join(json.dumps(c, ensure_ascii=False, sort_keys=True) + "\n" for c in cases)


@lru_cache(maxsize=None)
def _frozen_text() -> str:
    return _read_text(CASES_FILE)


def load_cases() -> list[dict[str, Any]]:
    """Load the frozen test cases shipped with the package."""
    return [json.loads(line) for line in _frozen_text().splitlines() if line.strip()]


def dataset_sha256() -> str:
    """Fingerprint of the frozen test cases; recorded with every result."""
    return hashlib.sha256(_frozen_text().encode("utf-8")).hexdigest()


def frozen_is_current() -> bool:
    """True if ``cases.jsonl`` matches what the sources would generate now."""
    try:
        return _frozen_text() == dump_cases(build_cases())
    except FileNotFoundError:
        return False


def case_system(case: dict[str, Any], defense: str = "basic") -> str:
    if defense not in DEFENSE_PROFILES:
        raise ValueError(f"unknown defense profile: {defense!r}")
    return case["system"] if defense == "basic" else case["system_no_defense"]
