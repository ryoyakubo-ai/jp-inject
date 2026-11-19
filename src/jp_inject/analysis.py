"""Turn recorded generations into success rates, comparisons and a report.

Everything here is recomputed from the raw result files, so the numbers in a
report can always be regenerated with ``jp-inject report``.
"""

from __future__ import annotations

import csv
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .dataset import CROSS_ENV, VARIANT_BY_ID, VARIANTS, load_cases
from .judge import judge, strip_reasoning
from .stats import bootstrap_mean, cluster_bootstrap_rate, wilson

Row = dict[str, Any]

# Report order: the two plain forms, then by family.
VARIANT_ORDER = [v.id for v in VARIANTS]
CROSS_IDS = [f"{variant}@{env}" for variant, env in CROSS_ENV]


# --------------------------------------------------------------------------
# Loading and judging
# --------------------------------------------------------------------------


def load_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Read result files; the last record wins if a generation appears twice."""
    latest: dict[tuple, dict[str, Any]] = {}
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (record["model"], record["defense"], record["case_id"], record["repeat"])
                latest[key] = record
    return list(latest.values())


def judged_rows(records: Iterable[dict[str, Any]], cases: Iterable[dict[str, Any]] | None = None) -> list[Row]:
    """Join each record with its test case and judge it."""
    by_id = {c["id"]: c for c in (cases if cases is not None else load_cases())}
    rows: list[Row] = []
    for record in records:
        case = by_id.get(record["case_id"])
        if case is None:
            continue  # recorded with a different dataset version
        verdict = judge(case, record.get("content"), record.get("finish_reason"))
        request = record.get("request") or {}
        settings = json.dumps(
            {
                "dataset": (record.get("dataset_sha256") or "")[:12],
                "temperature": request.get("temperature"),
                "max_tokens": request.get("max_tokens"),
                "extra": request.get("extra") or {},
            },
            sort_keys=True, ensure_ascii=False,
        )
        rows.append({
            "model": record["model"],
            "defense": record["defense"],
            "case_id": case["id"],
            "repeat": record["repeat"],
            "kind": case["kind"],
            "target": case["target"],
            "goal": case["goal"],
            "scenario": case["scenario"],
            "strategy": case["strategy"],
            "base": case["base"],
            "variant": case["variant"],
            "family": case["family"],
            "lang": case["lang"],
            "env": case["env"],
            "label": verdict.label,
            "success": verdict.success,
            "valid": verdict.valid,
            "token_present": verdict.token_present,
            "reason": verdict.reason,
            # The final answer exactly as the judge saw it (reasoning removed).
            "content": strip_reasoning(record.get("content") or ""),
            "settings": settings,
        })
    return rows


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def _clusters(rows: Iterable[Row], flag: str = "success") -> dict[str, tuple[int, int]]:
    acc: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        acc[row["base"]][0] += int(bool(row[flag]))
        acc[row["base"]][1] += 1
    return {base: (k, n) for base, (k, n) in acc.items()}


def _rate(rows: list[Row], flag: str = "success") -> dict[str, Any]:
    clusters = _clusters(rows, flag)
    rate, low, high = cluster_bootstrap_rate(clusters)
    return {
        "n": len(rows),
        "bases": len(clusters),
        "successes": sum(k for k, _ in clusters.values()),
        "rate": rate,
        "low": low,
        "high": high,
    }


def _per_base_rate(rows: Iterable[Row]) -> dict[str, float]:
    return {base: k / n for base, (k, n) in _clusters(rows).items() if n}


def _paired(a: dict[str, float], b: dict[str, float]) -> dict[str, Any]:
    """Mean of per-base differences a - b over bases present in both."""
    shared = sorted(set(a) & set(b))
    mean, low, high = bootstrap_mean([a[base] - b[base] for base in shared])
    return {"bases": len(shared), "diff": mean, "low": low, "high": high}


def summarize(rows: list[Row]) -> dict[str, Any]:
    """Aggregate judged rows of ONE model and defense profile."""
    valid = [r for r in rows if r["valid"]]
    attacks = [r for r in valid if r["kind"] == "attack"]
    by_variant: dict[str, list[Row]] = defaultdict(list)
    for row in attacks:
        by_variant[row["variant"]].append(row)
    per_base = {variant: _per_base_rate(items) for variant, items in by_variant.items()}

    variants: list[dict[str, Any]] = []
    for variant_id in VARIANT_ORDER:
        items = by_variant.get(variant_id)
        if not items:
            continue
        spec = VARIANT_BY_ID[variant_id]
        entry = {
            "variant": variant_id,
            "family": spec.family,
            "lang": spec.lang,
            "description": spec.description,
            **_rate(items),
            "strict": _rate([{**r, "success": r["label"] in ("full", "leak")} for r in items])["rate"],
            "loose": _rate(items, "token_present")["rate"],
        }
        if spec.reference and spec.reference in per_base:
            entry["vs_reference"] = {"reference": spec.reference, **_paired(per_base[variant_id], per_base[spec.reference])}
        if spec.also_compare and spec.also_compare in per_base:
            entry["vs_also"] = {"reference": spec.also_compare, **_paired(per_base[variant_id], per_base[spec.also_compare])}
        variants.append(entry)

    # Is an effect specific to Japanese? Compare each Japanese technique's
    # change from plain Japanese with its English counterpart's change from
    # plain English, base by base.
    interactions: list[dict[str, Any]] = []
    for spec in VARIANTS:
        needed = (spec.id, spec.reference, spec.counterpart)
        if spec.lang != "ja" or not spec.counterpart or not all(k in per_base for k in needed):
            continue
        other = VARIANT_BY_ID[spec.counterpart]
        if other.reference not in per_base:
            continue
        ja, ja0 = per_base[spec.id], per_base[spec.reference]
        en, en0 = per_base[other.id], per_base[other.reference]
        # All three columns use the same base attacks: those run under all
        # four conditions. Otherwise "JA change - EN change" would not equal
        # the difference of the two changes shown beside it.
        shared = sorted(set(ja) & set(ja0) & set(en) & set(en0))
        if not shared:
            continue
        ja_change = sum(ja[b] - ja0[b] for b in shared) / len(shared)
        en_change = sum(en[b] - en0[b] for b in shared) / len(shared)
        mean, low, high = bootstrap_mean([(ja[b] - ja0[b]) - (en[b] - en0[b]) for b in shared])
        interactions.append({
            "ja": spec.id, "en": other.id, "bases": len(shared),
            "ja_change": ja_change, "en_change": en_change,
            "difference": mean, "low": low, "high": high,
        })

    # Payload language x application language, plain payloads only.
    language_grid = []
    for variant_id in ("ja.plain", "en.plain@ja", "ja.plain@en", "en.plain"):
        items = by_variant.get(variant_id)
        if items:
            payload_lang = variant_id[:2]
            env = items[0]["env"]
            language_grid.append({"payload": payload_lang, "env": env, "variant": variant_id, **_rate(items)})

    def breakdown(field: str) -> list[dict[str, Any]]:
        out = []
        for variant_id in ("ja.plain", "en.plain"):
            groups: dict[str, list[Row]] = defaultdict(list)
            for row in by_variant.get(variant_id, []):
                groups[row[field]].append(row)
            for name, items in sorted(groups.items()):
                k = sum(r["success"] for r in items)
                low, high = wilson(k, len(items))
                out.append({"variant": variant_id, field: name, "n": len(items), "rate": k / len(items), "low": low, "high": high})
        return out

    comprehension = []
    comp_rows: dict[str, list[Row]] = defaultdict(list)
    for row in valid:
        if row["kind"] == "comprehension":
            comp_rows[row["variant"]].append(row)
    for variant_id in VARIANT_ORDER:
        items = comp_rows.get(variant_id)
        if items:
            k = sum(r["success"] for r in items)
            low, high = wilson(k, len(items))
            comprehension.append({"variant": variant_id, "n": len(items), "rate": k / len(items), "low": low, "high": high})

    benign = [r for r in valid if r["kind"] == "benign"]
    return {
        "generations": len(rows),
        "settings": sorted({r["settings"] for r in rows if "settings" in r}),
        "invalid": len(rows) - len(valid),
        "attack_generations": len(attacks),
        "benign": {"n": len(benign), "contaminated": sum(not r["success"] for r in benign)},
        "labels": dict(Counter(r["label"] for r in rows if r["kind"] == "attack")),
        "variants": variants,
        "interactions": interactions,
        "language_grid": language_grid,
        "by_target": breakdown("target"),
        "by_strategy": breakdown("strategy"),
        "comprehension": comprehension,
    }


def summarize_all(rows: list[Row]) -> dict[tuple[str, str], dict[str, Any]]:
    """Summaries keyed by ``(model, defense)``."""
    groups: dict[tuple[str, str], list[Row]] = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["defense"])].append(row)
    return {key: summarize(items) for key, items in sorted(groups.items())}


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _pct(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    return f"{value * 100:.0f}%"


def _ci(entry: dict[str, Any]) -> str:
    return f"{_pct(entry['rate'])} [{_pct(entry['low'])}, {_pct(entry['high'])}]"


def _pp(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "–"
    return f"{value * 100:+.0f}"


def _diff(entry: dict[str, Any] | None, key: str = "diff") -> str:
    if not entry:
        return ""
    value, low, high = entry[key], entry["low"], entry["high"]
    if math.isnan(value):
        return "–"
    star = ""
    if not math.isnan(low) and not math.isnan(high) and (low > 0 or high < 0):
        star = " *"
    return f"{_pp(value)} [{_pp(low)}, {_pp(high)}]{star}"


def _table(header: list[str], body: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(" --- " for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)


def render_markdown(summaries: dict[tuple[str, str], dict[str, Any]]) -> str:
    """A Markdown report with every table, one section per model."""
    out = [
        "# jp-inject results",
        "",
        "Generated by `jp-inject report`. Do not edit by hand.",
        "",
        "- **ASR** (attack success rate) is the share of valid generations in which the attack",
        "  succeeded. Brackets give a 95% interval from a cluster bootstrap over base attacks.",
        "- **Δ** columns are percentage-point differences, averaged over base attacks that were",
        "  run under both conditions. `*` marks intervals that exclude zero. Many comparisons",
        "  are made, so treat single starred results as leads to follow up, not as proof.",
        "- **strict / loose** bracket the hijack rate without the judge's mention heuristics:",
        "  strict counts only responses that are essentially just the code, loose counts every",
        "  response containing the code.",
        "",
    ]
    for (model, defense), s in summaries.items():
        comp = {c["variant"]: c for c in s["comprehension"]}
        out += [f"## {model} (defense profile: {defense})", ""]
        out += [
            f"- Generations: {s['generations']} ({s['invalid']} without a usable answer, excluded)",
            *(
                [f"- **Warning: these generations were recorded under {len(s['settings'])} different "
                 "settings or dataset versions. The rates below mix them and should not be used.**"]
                if len(s.get("settings", [])) > 1 else []
            ),
            f"- Benign controls: {s['benign']['contaminated']} of {s['benign']['n']} contained a code that should not appear",
            f"- Judge labels on attack cases: {', '.join(f'{k} {v}' for k, v in sorted(s['labels'].items()))}",
            "",
        ]
        if s["language_grid"]:
            out += ["### Payload language × application language (plain payloads)", ""]
            out.append(_table(
                ["Payload", "Application", "n", "ASR"],
                [[g["payload"], g["env"], str(g["n"]), _ci(g)] for g in s["language_grid"]],
            ))
            out.append("")
        out += ["### By technique", ""]
        body = []
        for v in s["variants"]:
            ref = v.get("vs_reference")
            also = v.get("vs_also")
            body.append([
                f"`{v['variant']}`", v["family"], str(v["n"]), _ci(v),
                _diff(ref) + (f" vs `{ref['reference']}`" if ref else ""),
                _diff(also) + (f" vs `{also['reference']}`" if also else ""),
                f"{_pct(v['strict'])} / {_pct(v['loose'])}",
                _pct(comp[v["variant"]]["rate"]) if v["variant"] in comp else "",
            ])
        out.append(_table(
            ["Variant", "Family", "n", "ASR", "Δ vs plain", "Δ vs closer baseline", "strict / loose", "Readable"],
            body,
        ))
        out += ["", "*Readable*: share of comprehension controls in which the model followed a harmless instruction written the same way.", ""]
        if s["interactions"]:
            out += ["### Is the effect specific to Japanese?", ""]
            out.append(_table(
                ["Japanese technique", "English counterpart", "Base attacks", "JA change", "EN change", "JA − EN"],
                [[f"`{i['ja']}`", f"`{i['en']}`", str(i["bases"]), _pp(i["ja_change"]), _pp(i["en_change"]),
                  _diff(i, "difference")]
                 for i in s["interactions"]],
            ))
            out.append("")
        for title, field, items in (("By target", "target", s["by_target"]), ("By strategy", "strategy", s["by_strategy"])):
            if items:
                out += [f"### {title} (plain payloads)", ""]
                out.append(_table(
                    ["Variant", field.capitalize(), "n", "ASR (Wilson 95%)"],
                    [[f"`{i['variant']}`", i[field], str(i["n"]), _ci(i)] for i in items],
                ))
                out.append("")
    return "\n".join(out).rstrip() + "\n"


def variants_csv(summaries: dict[tuple[str, str], dict[str, Any]]) -> str:
    """Per-variant numbers as CSV (one row per model x defense x variant)."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow([
        "model", "defense", "variant", "family", "lang", "n", "bases", "successes", "asr",
        "ci_low", "ci_high", "strict", "loose", "reference", "diff_vs_reference",
        "diff_low", "diff_high", "comprehension",
    ])
    for (model, defense), s in summaries.items():
        comp = {c["variant"]: c["rate"] for c in s["comprehension"]}
        for v in s["variants"]:
            ref = v.get("vs_reference") or {}
            writer.writerow([
                model, defense, v["variant"], v["family"], v["lang"], v["n"], v["bases"],
                v["successes"], f"{v['rate']:.4f}", f"{v['low']:.4f}", f"{v['high']:.4f}",
                f"{v['strict']:.4f}", f"{v['loose']:.4f}", ref.get("reference", ""),
                "" if not ref else f"{ref['diff']:.4f}",
                "" if not ref else f"{ref['low']:.4f}",
                "" if not ref else f"{ref['high']:.4f}",
                "" if v["variant"] not in comp else f"{comp[v['variant']]:.4f}",
            ])
    return buffer.getvalue()


def plot_variants(summary: dict[str, Any], title: str, path: Path) -> bool:
    """Dot plot of ASR by variant with 95% intervals. Needs matplotlib.

    Returns False (and writes nothing) if matplotlib is not installed.
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
    variants = summary["variants"]
    if not variants:
        return False

    # Lay rows out top to bottom, with a gap and a heading between families.
    rows: list[tuple[float, dict[str, Any]]] = []
    headings: list[tuple[float, str]] = []
    y, last_family = 0.0, None
    for v in variants:
        if v["family"] != last_family:
            y += 0.9 if last_family is not None else 0.0
            headings.append((y, v["family"]))
            y += 0.9
            last_family = v["family"]
        rows.append((y, v))
        y += 1.0

    fig, ax = plt.subplots(figsize=(8.2, 0.28 * y + 1.7), facecolor=surface)
    ax.set_facecolor(surface)
    for position, v in rows:
        color = colors[v["lang"]]
        ax.plot([v["low"] * 100, v["high"] * 100], [position, position], color=color, lw=2,
                solid_capstyle="round", zorder=2)
        ax.plot(v["rate"] * 100, position, "o", ms=9, color=color, mec=surface, mew=2, zorder=3)
    for position, family in headings:
        ax.text(-1.5, position, family.upper(), ha="right", va="center", fontsize=8, color=muted,
                fontweight="bold")

    ax.set_yticks([p for p, _ in rows])
    ax.set_yticklabels([v["variant"] for _, v in rows], fontsize=9, color=ink)
    ax.set_ylim(y - 0.4, -0.8)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Attack success rate (%), 95% interval", fontsize=9, color=muted)
    ax.xaxis.grid(True, color=grid, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, colors=muted)
    for spine in ax.spines.values():
        spine.set_visible(False)
    handles = [
        plt.Line2D([], [], marker="o", ls="-", lw=2, ms=8, color=colors[lang], mec=surface, mew=2, label=names[lang])
        for lang in ("ja", "en")
        if any(v["lang"] == lang for _, v in rows)
    ]
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncols=2, frameon=False,
              fontsize=9, labelcolor=ink, borderaxespad=0.2)
    ax.set_title(title, loc="left", fontsize=11, color=ink, pad=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=surface)
    plt.close(fig)
    return True


def _json_safe(value: Any) -> Any:
    """Replace NaN (not valid JSON) with null, recursively."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def write_report(rows: list[Row], out_dir: Path) -> list[Path]:
    """Write summary.md, CSVs, a JSON dump and (if possible) figures."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = summarize_all(rows)
    written: list[Path] = []

    def save(name: str, text: str) -> None:
        path = out_dir / name
        path.write_text(text, encoding="utf-8", newline="\n")
        written.append(path)

    save("summary.md", render_markdown(summaries))
    save("variants.csv", variants_csv(summaries))
    save(
        "summary.json",
        json.dumps(
            _json_safe([{"model": m, "defense": d, **s} for (m, d), s in summaries.items()]),
            ensure_ascii=False, indent=2,
        ) + "\n",
    )
    for (model, defense), summary in summaries.items():
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in f"{model}_{defense}")
        path = out_dir / f"asr_{safe}.png"
        if plot_variants(summary, f"{model} · defense: {defense}", path):
            written.append(path)
    return written
