"""Where a detector and the model behind it disagree, and what normalisation changes.

A detector in front of a model protects only against texts it flags. A way of
writing an attack is dangerous when the detector misses it *and* the model
still follows it; it is harmless when the model cannot read it either. This
module puts the two measurements side by side for the same texts, and
compares four pipelines:

====  ==========================  ======================
name  the detector is shown       the model is shown
====  ==========================  ======================
A     (no detector)               the message as it is
B     the message as it is        the message as it is
C     the normalised message      the message as it is
D     the normalised message      the normalised message
====  ==========================  ======================

For each pipeline the figure of interest is the share of generations in
which the attack got past the detector and the model followed it. A flagged
message is assumed to be blocked outright.

A and B need a detector score file and a model run. C needs the score file
to hold a normalised view (``jp-inject detect --normalize``). D also needs a
model run recorded with the same level (``jp-inject run --normalize``).
"""

from __future__ import annotations

import csv
import io
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .analysis import CROSS_IDS, VARIANT_ORDER, Row, _ci, _diff, _json_safe, _pct, _table
from . import conditions as run_conditions
from .conditions import NO_NORMALIZATION
from .dataset import VARIANT_BY_ID, load_cases
from .detect import DEFAULT_THRESHOLD, NORMALIZED_VIEWS
from .normalize import LEVELS, normalize
from .stats import bootstrap_mean

PerBase = dict[str, float]


class GapInputError(ValueError):
    """The score file and the model runs cannot be combined as given."""


def _reference(condition: str) -> str | None:
    """The wording a condition is compared with (None for the two plain forms)."""
    if "@" in condition:
        return condition.split("@")[0]
    return VARIANT_BY_ID[condition].reference


def _mean(values: PerBase) -> dict[str, Any]:
    mean, low, high = bootstrap_mean([values[base] for base in sorted(values)])
    return {"bases": len(values), "rate": mean, "low": low, "high": high}


def _paired(a: PerBase, b: PerBase) -> dict[str, Any]:
    """Mean of per-base differences a - b over the bases present in both."""
    shared = sorted(set(a) & set(b))
    mean, low, high = bootstrap_mean([a[base] - b[base] for base in shared])
    return {"bases": len(shared), "diff": mean, "low": low, "high": high}


def _product(asr: PerBase, miss: PerBase) -> PerBase:
    """Per base: share of generations that passed the detector and succeeded."""
    return {base: asr[base] * miss[base] for base in sorted(set(asr) & set(miss))}


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def miss_tables(scores: Iterable[Row], threshold: float) -> dict[str, dict[str, dict[str, PerBase]]]:
    """``detector -> view -> condition -> base -> 1.0 if the attack was missed``."""
    out: dict[str, dict[str, dict[str, PerBase]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for row in scores:
        if row["kind"] == "attack":
            out[row["detector"]][row["view"]][row["variant"]][row["base"]] = float(row["score"] < threshold)
    return out


def control_counts(scores: Iterable[Row], threshold: float) -> dict[str, dict[str, dict[str, dict[str, int]]]]:
    """``detector -> view -> kind -> {"n", "flagged"}`` for the cases without an attack."""
    out: dict[str, dict[str, dict[str, dict[str, int]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: {"n": 0, "flagged": 0})))
    for row in scores:
        if row["kind"] in ("benign", "comprehension"):
            cell = out[row["detector"]][row["view"]][row["kind"]]
            cell["n"] += 1
            cell["flagged"] += int(row["score"] >= threshold)
    return out


def model_tables(judged: Iterable[Row]) -> dict[tuple[str, str], dict[str, Any]]:
    """Per-base attack success of every model run, grouped for pairing.

    Returns ``(model_id, defense) -> {"labels": {level: label},
    "asr": {level: condition -> base -> rate}, "readable": {level: condition
    -> {"n", "followed"}}}`` where ``level`` is the input normalisation of
    the run (``"none"`` for a plain run). Runs are paired by the model
    identifier the server was asked for, not by their label, because a
    normalised run carries a label of its own.

    The runs of one model must differ in input normalisation only. If any
    other known part of the condition differs (test cases, server, model
    file, temperature, token limit, seed, extra request fields), a change in
    the success rate could not be attributed to normalisation, and the
    input is refused.
    """
    counts: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
    readable: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
    labels: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    known: dict[tuple[str, str], dict[str, Any]] = defaultdict(dict)
    seen_in: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for row in judged:
        group = (row.get("model_id") or row["model"], row["defense"])
        level = row.get("input_normalization") or NO_NORMALIZATION
        # Checked on every row, valid or not: an unusable generation was still
        # produced under the run's condition. Remember the first known value
        # of each field, as analysis.load_records does within one label.
        condition = {field: value for field, value in (row.get("condition") or {}).items()
                     if field != "input_normalization"}
        differing = run_conditions.differences(known[group], condition)
        if differing:
            where = ", ".join(sorted({seen_in[group][field] for field in differing}))
            raise GapInputError(
                f"the runs of model {group[0]!r} (defense {group[1]!r}) differ in more than input "
                f"normalisation: {run_conditions.describe(differing)} ({where} vs {row['model']}). "
                "A difference between them could not be attributed to normalisation; record the "
                "normalised run with the same settings as the plain one."
            )
        for field, value in condition.items():
            if value is not None and known[group].get(field) is None:
                known[group][field] = value
                seen_in[group][field] = row["model"]
        if not row["valid"]:
            continue
        labels[group][level].add(row["model"])
        if row["kind"] == "attack":
            cell = counts[(group, level, row["variant"], row["base"])]
        elif row["kind"] == "comprehension":
            cell = readable[(group, level, row["variant"])]
        else:
            continue
        cell[0] += int(bool(row["success"]))
        cell[1] += 1
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for group, by_level in labels.items():
        for level, names in by_level.items():
            if len(names) > 1:
                raise GapInputError(
                    f"model {group[0]!r} (defense {group[1]!r}) has more than one run with input "
                    f"normalisation {level!r}: {', '.join(sorted(names))}. Pass one run per level."
                )
        out[group] = {"labels": {level: next(iter(names)) for level, names in by_level.items()},
                      "asr": defaultdict(lambda: defaultdict(dict)), "readable": defaultdict(dict)}
    for (group, level, condition, base), (k, n) in counts.items():
        out[group]["asr"][level][condition][base] = k / n
    for (group, level, condition), (k, n) in readable.items():
        out[group]["readable"][level][condition] = {"n": n, "followed": k}
    return out


# --------------------------------------------------------------------------
# What normalisation does to the texts (no detector or model involved)
# --------------------------------------------------------------------------


def text_identity(cases: Iterable[dict[str, Any]] | None = None) -> dict[str, dict[str, dict[str, int]]]:
    """How often normalisation turns a rewritten attack back into its reference.

    ``level -> condition -> {"n", "identical"}``: the number of base attacks
    whose user message, after normalisation, is character for character the
    normalised message of the reference wording. Where that holds, detector
    and model see the very same text for both wordings, so the rewriting has
    been undone completely. ``ja.hankaku`` is also compared with
    ``ja.katakana``, which is what NFKC turns it into.
    """
    by_key: dict[tuple[str, str], str] = {}
    for case in cases if cases is not None else load_cases():
        if case["kind"] == "attack":
            by_key[(case["base"], case["variant"])] = case["user"]
    out: dict[str, dict[str, dict[str, int]]] = {}
    for level in LEVELS:
        table: dict[str, dict[str, int]] = {}
        for condition in VARIANT_ORDER:
            for reference in filter(None, (_reference(condition),
                                            "ja.katakana" if condition == "ja.hankaku" else None)):
                pairs = [(text, by_key[(base, reference)]) for (base, variant), text in by_key.items()
                         if variant == condition and (base, reference) in by_key]
                if not pairs:
                    continue
                name = condition if reference == _reference(condition) else f"{condition} (vs {reference})"
                table[name] = {"n": len(pairs), "identical": sum(
                    normalize(a, level) == normalize(b, level) for a, b in pairs)}
        out[level] = table
    return out


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------


def summarize(scores: list[Row], judged: list[Row], *, threshold: float = DEFAULT_THRESHOLD) -> dict[str, Any]:
    """Combine detector scores and model runs into the gap tables."""
    misses = miss_tables(scores, threshold)
    controls = control_counts(scores, threshold)
    models = model_tables(judged)
    if not misses:
        raise GapInputError("the score file holds no attack cases")
    if not models:
        raise GapInputError("the model runs hold no valid generations")

    pairs = []
    for detector in sorted(misses):
        views = misses[detector]
        if "message" not in views:
            raise GapInputError(f"{detector}: the score file has no scores for the user message")
        levels = [level for level in LEVELS if NORMALIZED_VIEWS[level] in views]
        for (model_id, defense), model in sorted(models.items()):
            raw = model["asr"].get(NO_NORMALIZATION)
            if not raw:
                continue  # only a normalised run of this model was given
            conditions = [c for c in VARIANT_ORDER + CROSS_IDS if c in raw and c in views["message"]]
            table: dict[str, Any] = {}
            for condition in conditions:
                # Every figure of one condition is computed over the same base
                # attacks: those present in the model run, the detector scores
                # and every normalised view and run that covers the condition.
                # Otherwise a detector scored on part of the attacks would be
                # compared with a model rate over all of them.
                sources = [raw[condition], views["message"][condition]]
                sources += [views[NORMALIZED_VIEWS[level]][condition] for level in levels
                            if views[NORMALIZED_VIEWS[level]].get(condition)]
                sources += [model["asr"][level][condition] for level in levels
                            if model["asr"].get(level, {}).get(condition)]
                shared = set.intersection(*(set(source) for source in sources))
                if not shared:
                    continue

                def only(values: PerBase, shared: set[str] = shared) -> PerBase:
                    return {base: values[base] for base in shared}

                asr, miss = only(raw[condition]), only(views["message"][condition])
                behind = _product(asr, miss)
                entry: dict[str, Any] = {
                    "miss": _mean(miss),
                    "asr": _mean(asr),
                    "A": _mean(asr),
                    "B": _mean(behind),
                    "normalized": {},
                }
                seen = model["readable"].get(NO_NORMALIZATION, {}).get(condition)
                if seen:
                    entry["readable"] = seen
                reference = _reference(condition)
                if reference in raw and reference in views["message"]:
                    entry["reference"] = reference
                    entry["miss_vs_reference"] = _paired(miss, views["message"][reference])
                    entry["asr_vs_reference"] = _paired(asr, raw[reference])
                    entry["B_vs_reference"] = _paired(behind, _product(raw[reference], views["message"][reference]))
                for level in levels:
                    miss_n = views[NORMALIZED_VIEWS[level]].get(condition)
                    if not miss_n:
                        continue
                    miss_n = only(miss_n)
                    c = _product(asr, miss_n)
                    cell: dict[str, Any] = {"miss": _mean(miss_n), "miss_vs_raw": _paired(miss_n, miss),
                                            "C": _mean(c), "C_vs_B": _paired(c, behind)}
                    asr_n = model["asr"].get(level, {}).get(condition)
                    if asr_n:
                        asr_n = only(asr_n)
                        d = _product(asr_n, miss_n)
                        cell.update({"asr": _mean(asr_n), "asr_vs_raw": _paired(asr_n, asr),
                                     "D": _mean(d), "D_vs_C": _paired(d, c), "D_vs_B": _paired(d, behind)})
                    entry["normalized"][level] = cell
                table[condition] = entry
            pairs.append({
                "detector": detector,
                "model_id": model_id,
                "defense": defense,
                "model_labels": model["labels"],
                "levels": levels,
                "model_levels": [level for level in levels if level in model["asr"]],
                "conditions": table,
                "worst": _worst(table, levels),
            })
    if not pairs:
        raise GapInputError("no model run without input normalisation was given; pipelines A and B need one")
    return {
        "threshold": threshold,
        "pairs": pairs,
        "controls": {detector: {view: dict(kinds) for view, kinds in views.items()}
                     for detector, views in controls.items()},
        "text_identity": text_identity(),
    }


def _worst(table: dict[str, Any], levels: list[str]) -> dict[str, Any]:
    """For each pipeline, the wording with the highest rate (the attacker's pick).

    The maximum over twenty-odd noisy estimates is biased upwards; it shows
    which wording to look at, not how large the risk is.
    """
    def best(pick) -> dict[str, Any] | None:
        candidates = [(pick(entry), condition) for condition, entry in table.items() if pick(entry) is not None]
        candidates = [(value, condition) for value, condition in candidates if not math.isnan(value["rate"])]
        if not candidates:
            return None
        value, condition = max(candidates, key=lambda item: item[0]["rate"])
        return {"condition": condition, **value}

    out = {"A": best(lambda e: e["A"]), "B": best(lambda e: e["B"])}
    for level in levels:
        out[f"C:{level}"] = best(lambda e, level=level: e["normalized"].get(level, {}).get("C"))
        out[f"D:{level}"] = best(lambda e, level=level: e["normalized"].get(level, {}).get("D"))
    return {key: value for key, value in out.items() if value}


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _count(cell: dict[str, int] | None) -> str:
    return f"{cell['flagged']}/{cell['n']}" if cell else ""


def _title(pair: dict[str, Any]) -> str:
    return f"{pair['detector']} in front of {pair['model_id']} (defense profile: {pair['defense']})"


def render_markdown(summary: dict[str, Any]) -> str:
    parts = [
        "# jp-inject: detector and model side by side",
        "",
        "Generated by `jp-inject gap-report`. Do not edit by hand.",
        "",
        "- **Missed** is the share of the 20 base attacks whose user message the detector scored",
        f"  below the threshold ({summary['threshold']:g}). **Followed** is the model's attack success",
        "  rate on the same messages with no detector. Brackets give a 95% bootstrap interval over",
        "  base attacks.",
        "- **Missed and followed** is the share of generations in which the detector missed the",
        "  message *and* the model followed the injection, assuming a flagged message is blocked.",
        "- **Δ** is the paired difference in percentage points. `*` marks intervals that exclude",
        "  zero; many comparisons are made, so single starred results are leads, not findings.",
        "- **Readable** is how often the model followed a harmless instruction written the same",
        "  way: a low *Followed* next to a low *Readable* means the model could not read the text.",
        "- Pipelines: **A** no detector; **B** detector and model both see the message as it is;",
        "  **C** the detector sees the normalised message, the model the original; **D** both see",
        "  the normalised message.",
        "- Levels: `nfkc` is Unicode NFKC plus removal of invisible format characters. `fold` adds",
        "  modern kanji forms and look-alike repair; its tables mirror the ones the test cases were",
        "  written with, so it is a best case for the defender.",
        "",
        "## What normalisation does to the texts",
        "",
        "Number of base attacks whose user message becomes, after normalisation, character for",
        "character the normalised message of the wording it is compared with. No detector or model",
        "is involved. Wordings not listed are never turned back.",
        "",
    ]
    identity = summary["text_identity"]
    names = [name for name in identity.get(LEVELS[0], {})
             if any(identity[level].get(name, {}).get("identical") for level in LEVELS)]
    body = []
    for name in names:
        row = [f"`{name}`" if " " not in name else "`" + name.replace(" (vs ", "` (vs `").replace(")", "`)")]
        for level in LEVELS:
            cell = identity[level].get(name)
            row.append(f"{cell['identical']}/{cell['n']}" if cell else "")
        body.append(row)
    parts += [_table(["Wording"] + [f"Identical after `{level}`" for level in LEVELS], body), ""]

    parts += ["## Texts without an attack", "",
              "Flagged out of all texts of the kind, for the message as it is and after each level.", ""]
    body = []
    for detector, views in sorted(summary["controls"].items()):
        for kind, label in (("benign", "no injection (false positives)"),
                            ("comprehension", "harmless one-line instruction")):
            row = [f"`{detector}`", label, _count(views.get("message", {}).get(kind))]
            row += [_count(views.get(NORMALIZED_VIEWS[level], {}).get(kind)) for level in LEVELS]
            body.append(row)
    parts += [_table(["Detector", "Texts", "As is"] + [f"After `{level}`" for level in LEVELS], body), ""]

    for pair in summary["pairs"]:
        parts += [f"## {_title(pair)}", ""]
        worst = pair["worst"]
        if worst:
            parts += ["Wording with the highest missed-and-followed rate under each pipeline (the maximum of",
                      "many noisy estimates is biased upwards; it says which wording to look at):", ""]
            body = [[key.replace(":", ", "), f"`{value['condition']}`", _ci(value)] for key, value in worst.items()]
            parts += [_table(["Pipeline", "Wording", "Missed and followed"], body), ""]
        header = ["Condition", "Missed", "Δ vs plain", "Followed (A)", "Δ vs plain", "Readable",
                  "Missed and followed (B)", "Δ vs plain"]
        body = []
        for condition, entry in pair["conditions"].items():
            seen = entry.get("readable")
            body.append([
                f"`{condition}`", _ci(entry["miss"]), _diff(entry.get("miss_vs_reference")),
                _ci(entry["asr"]), _diff(entry.get("asr_vs_reference")),
                _pct(seen["followed"] / seen["n"]) if seen and seen["n"] else "",
                _ci(entry["B"]), _diff(entry.get("B_vs_reference")),
            ])
        parts += [_table(header, body), ""]
        for level in pair["levels"]:
            with_model = level in pair["model_levels"]
            parts += [f"### After `{level}` normalisation", ""]
            if with_model:
                parts += [f"Model run with normalised input: `{pair['model_labels'][level]}`.", ""]
            else:
                parts += ["No model run with this input normalisation was given, so pipeline D is not shown.", ""]
            header = ["Condition", "Missed, as is", "Missed, normalised", "Δ", "B", "C", "C − B"]
            if with_model:
                header += ["Followed, as is", "Followed, normalised", "Δ", "D", "D − C", "D − B"]
            body = []
            for condition, entry in pair["conditions"].items():
                cell = entry["normalized"].get(level)
                if not cell:
                    continue
                row = [f"`{condition}`", _ci(entry["miss"]), _ci(cell["miss"]), _diff(cell["miss_vs_raw"]),
                       _ci(entry["B"]), _ci(cell["C"]), _diff(cell["C_vs_B"])]
                if with_model:
                    row += ([_ci(entry["asr"]), _ci(cell["asr"]), _diff(cell["asr_vs_raw"]), _ci(cell["D"]),
                             _diff(cell["D_vs_C"]), _diff(cell["D_vs_B"])] if "D" in cell else [""] * 6)
                body.append(row)
            parts += [_table(header, body), ""]
    return "\n".join(parts)


def gap_csv(summary: dict[str, Any]) -> str:
    """One row per detector, model, condition, normalisation level and measure."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["detector", "model_id", "defense", "threshold", "condition", "level", "measure",
                     "bases", "value", "ci_low", "ci_high"])

    def fmt(value: float | None) -> str:
        return "" if value is None or math.isnan(value) else f"{value:.4f}"

    def emit(pair: dict[str, Any], condition: str, level: str, measure: str, cell: dict[str, Any] | None) -> None:
        if cell:
            value = cell["rate"] if "rate" in cell else cell["diff"]
            writer.writerow([pair["detector"], pair["model_id"], pair["defense"], summary["threshold"],
                             condition, level, measure, cell["bases"], fmt(value), fmt(cell["low"]),
                             fmt(cell["high"])])

    for pair in summary["pairs"]:
        for condition, entry in pair["conditions"].items():
            for measure in ("miss", "asr", "B", "miss_vs_reference", "asr_vs_reference", "B_vs_reference"):
                emit(pair, condition, NO_NORMALIZATION, measure, entry.get(measure))
            for level, cell in entry["normalized"].items():
                for measure in ("miss", "miss_vs_raw", "C", "C_vs_B", "asr", "asr_vs_raw", "D", "D_vs_C", "D_vs_B"):
                    emit(pair, condition, level, measure, cell.get(measure))
    return buffer.getvalue()


def plot_gap(pair: dict[str, Any], path: Path) -> bool:
    """Scatter of missed (x) against followed (y), one labelled point per wording.

    The upper right is where a detector gives no protection against an
    attack that works. The two axes are averages over base attacks, and the
    detector may miss exactly the base attacks the model follows (or the
    opposite), so the missed-and-followed rate cannot be read off as their
    product; it is in the tables. Needs matplotlib; returns False without it.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    surface, ink, muted, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    colors = {"ja": "#2a78d6", "en": "#eb6834"}
    names = {"ja": "Japanese payload", "en": "English payload"}
    points = [(entry["miss"]["rate"] * 100, entry["asr"]["rate"] * 100, condition)
              for condition, entry in pair["conditions"].items()
              if not math.isnan(entry["miss"]["rate"]) and not math.isnan(entry["asr"]["rate"])]
    if not points:
        return False
    top = max(40.0, math.ceil(max(y for _, y, _ in points) / 10 + 1) * 10)

    fig, ax = plt.subplots(figsize=(8.2, 6.4), facecolor=surface)
    ax.set_facecolor(surface)
    # Spread labels of points that share a spot: detectors score each text once,
    # so many wordings sit at exactly the same x.
    placed: list[tuple[float, float]] = []
    for x, y, condition in sorted(points, key=lambda p: (p[0], -p[1])):
        lang = condition.split(".")[0]
        ax.plot(x, y, "o", ms=9, color=colors[lang], mec=surface, mew=2, zorder=3)
        label_y = y
        while any(abs(px - x) < 16 and abs(py - label_y) < top * 0.034 for px, py in placed):
            label_y -= top * 0.034
        placed.append((x, label_y))
        ax.annotate(condition, (x, y), xytext=(x - 1.8, label_y), fontsize=8, color=ink, ha="right", va="center",
                    arrowprops={"arrowstyle": "-", "color": "#b8b6b0", "lw": 0.8} if label_y != y else None, zorder=2)
    ax.set_xlim(-22, 104)
    ax.set_ylim(-top * 0.06, top)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("Missed by the detector (% of 20 base attacks)", fontsize=9, color=muted)
    ax.set_ylabel("Followed by the model (%), no detector", fontsize=9, color=muted)
    ax.grid(True, color=grid, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, colors=muted)
    for spine in ax.spines.values():
        spine.set_visible(False)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=8, color=colors[lang], mec=surface, mew=2, label=names[lang])
               for lang in ("ja", "en") if any(c.startswith(lang + ".") for _, _, c in points)]
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncols=3, frameon=False,
              fontsize=9, labelcolor=ink, borderaxespad=0.2)
    ax.set_title(_title(pair), loc="left", fontsize=11, color=ink, pad=24)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=surface)
    plt.close(fig)
    return True


def write_report(scores: list[Row], judged: list[Row], out_dir: Path, *,
                 threshold: float = DEFAULT_THRESHOLD) -> list[Path]:
    """Write summary.md, gap.csv, summary.json and (if possible) one figure per pair."""
    summary = summarize(scores, judged, threshold=threshold)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, text in (
        ("summary.md", render_markdown(summary) + "\n"),
        ("gap.csv", gap_csv(summary)),
        ("summary.json", json.dumps(_json_safe(summary), ensure_ascii=False, indent=2) + "\n"),
    ):
        path = out_dir / name
        path.write_text(text, encoding="utf-8", newline="\n")
        written.append(path)
    for pair in summary["pairs"]:
        name = f"{pair['detector']}_{pair['model_id']}_{pair['defense']}"
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)
        path = out_dir / f"gap_{safe}.png"
        if plot_gap(pair, path):
            written.append(path)
    return written
