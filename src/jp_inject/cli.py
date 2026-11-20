"""Command-line interface: ``jp-inject <command>``."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from . import __version__, analysis, dataset, detect, runner, transforms, validation

TRANSFORMS = {
    "fullwidth": transforms.to_fullwidth,
    "zero_width": transforms.insert_zero_width,
    "kyujitai": transforms.to_kyujitai,
    "lookalike": transforms.to_lookalike,
    "hiragana": transforms.reading_to_hiragana,
    "katakana": transforms.reading_to_katakana,
    "hankaku": transforms.reading_to_hankaku,
    "romaji": transforms.reading_to_romaji,
}
READING_TRANSFORMS = ("hiragana", "katakana", "hankaku", "romaji")


def _utf8_stdio() -> None:
    """Windows consoles often default to a legacy code page; force UTF-8."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def _select(cases: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    def keep(case: dict[str, Any]) -> bool:
        if args.kind and case["kind"] not in args.kind:
            return False
        if args.variant and case["variant"] not in args.variant:
            return False
        if args.target and case["target"] not in args.target:
            return False
        if args.family and case["family"] not in args.family:
            return False
        return True

    selected = [c for c in cases if keep(c)]
    if args.limit:
        selected = selected[: args.limit]
    return selected


def _add_filters(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("case selection (default: all cases)")
    group.add_argument("--kind", action="append", choices=["attack", "benign", "comprehension"])
    group.add_argument("--variant", action="append", help="e.g. ja.keigo (repeatable)")
    group.add_argument("--target", action="append", help="e.g. summarize.hijack (repeatable)")
    group.add_argument("--family", action="append",
                       choices=["language", "register", "framing", "orthographic", "cross_env", "control"])
    group.add_argument("--limit", type=int, help="only the first N selected cases")


# ---------------------------------------------------------------- commands


def cmd_build(args: argparse.Namespace) -> int:
    if args.check:
        if dataset.frozen_is_current():
            print("cases.jsonl is up to date")
            return 0
        print("cases.jsonl does not match the sources; run `jp-inject build`", file=sys.stderr)
        return 1
    cases = dataset.build_cases()
    target = Path(dataset.__file__).parent / "data" / dataset.CASES_FILE
    target.write_text(dataset.dump_cases(cases), encoding="utf-8", newline="\n")
    print(f"wrote {len(cases)} cases to {target}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    cases = _select(dataset.load_cases(), args)
    if args.ids:
        for case in cases:
            print(case["id"])
        return 0
    print(f"{len(cases)} cases (dataset sha256 {dataset.dataset_sha256()[:12]})")
    for title, field in (("kind", "kind"), ("family", "family"), ("target", "target"), ("variant", "variant")):
        counts = Counter(str(c[field]) for c in cases)
        print(f"\nby {title}:")
        for name, count in sorted(counts.items()):
            print(f"  {name:<28} {count}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    by_id = {c["id"]: c for c in dataset.load_cases()}
    case = by_id.get(args.case_id)
    if case is None:
        print(f"no such case: {args.case_id}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(case, ensure_ascii=False, indent=2))
        return 0
    print(f"# {case['id']}\n")
    print("## system\n" + dataset.case_system(case, args.defense) + "\n")
    print("## user\n" + case["user"])
    return 0


def cmd_transform(args: argparse.Namespace) -> int:
    text = args.text if args.text is not None else sys.stdin.read()
    print(TRANSFORMS[args.technique](text))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    cases = _select(dataset.load_cases(), args)
    try:
        extra = json.loads(args.extra) if args.extra else {}
    except json.JSONDecodeError as error:
        print(f"--extra is not valid JSON: {error}", file=sys.stderr)
        return 2
    for item in args.set or []:
        key, sep, value = item.partition("=")
        if not sep or not key:
            print(f"--set expects KEY=VALUE, got {item!r}", file=sys.stderr)
            return 2
        try:
            extra[key] = json.loads(value)
        except json.JSONDecodeError:
            extra[key] = value
    try:
        config = runner.RunConfig(
            model=args.model, out=Path(args.out), base_url=args.base_url, label=args.label,
            repeats=args.repeats, temperature=args.temperature, max_tokens=args.max_tokens,
            seed=args.seed, defense=args.defense, timeout=args.timeout, retries=args.retries,
            extra=extra, allow_remote=args.allow_remote, concurrency=args.concurrency,
        )
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    if args.dry_run:
        print(f"would send {len(cases)} cases x {config.repeats} repeats to {config.base_url}")
        if cases:
            print(json.dumps(runner.build_request(cases[0], config, 0), ensure_ascii=False, indent=2))
        return 0
    try:
        counts = runner.run(cases, config)
    except (runner.RemoteEndpointError, runner.RedirectError, runner.SettingsMismatchError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps(counts))
    return 1 if counts["failed"] else 0


def _expand(paths: Sequence[str]) -> list[Path]:
    """Expand wildcards ourselves: PowerShell and cmd.exe do not."""
    found: list[Path] = []
    for item in paths:
        matches = sorted(glob.glob(item)) if any(ch in item for ch in "*?[") else [item]
        found.extend(Path(m) for m in matches if not m.endswith(".meta.json"))
    return found


def _rows(paths: Sequence[str]) -> list[dict[str, Any]]:
    files = _expand(paths)
    missing = [str(f) for f in files if not f.is_file()]
    if missing or not files:
        raise SystemExit(f"result file(s) not found: {', '.join(missing) or ', '.join(paths)}")
    try:
        return analysis.judged_rows(analysis.load_records(files))
    except (analysis.MixedConditionsError, analysis.DatasetMismatchError) as error:
        raise SystemExit(str(error)) from error


def cmd_judge(args: argparse.Namespace) -> int:
    rows = _rows(args.results)
    out = sys.stdout if args.out == "-" else open(args.out, "w", encoding="utf-8", newline="\n")
    try:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        if out is not sys.stdout:
            out.close()
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    rows = _rows(args.results)
    if not rows:
        print("no usable records found", file=sys.stderr)
        return 1
    for path in analysis.write_report(rows, Path(args.out)):
        print(path)
    return 0


def cmd_sample(args: argparse.Namespace) -> int:
    rows = _rows(args.results)
    sample = validation.draw_sample(rows, n=args.n, seed=args.seed)
    if not sample:
        print("no valid attack generations to sample", file=sys.stderr)
        return 1
    cases = {c["id"]: c for c in dataset.load_cases()}
    sheet, key = validation.write_sheet(sample, cases, Path(args.out))
    print(f"{len(sample)} generations sampled\n  label these: {sheet}\n  answer key:  {key}")
    return 0


def cmd_agreement(args: argparse.Namespace) -> int:
    result = validation.agreement(Path(args.sheet), Path(args.key))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_detect(args: argparse.Namespace) -> int:
    detectors = args.detector or list(detect.DEFAULT_DETECTORS)

    def loader(model_id: str):
        return detect.load_hf_detector(model_id, device=args.device, batch_size=args.batch_size)

    result = detect.score_cases(detectors, Path(args.out), cases=_select(dataset.load_cases(), args), loader=loader)
    print(f"scored: {', '.join(result['scored']) or 'none'}")
    for item in result["skipped"]:
        print(f"skipped: {item['detector']}: {item['reason']}")
    print(f"written: {args.out}")
    return 0 if result["scored"] else 1


def cmd_detect_report(args: argparse.Namespace) -> int:
    try:
        rows = detect.load_scores(Path(args.scores))
    except detect.DatasetMismatchError as exc:
        print(exc, file=sys.stderr)
        return 2
    judged = _rows(args.runs) if args.runs else None
    for path in detect.write_report(rows, Path(args.out), threshold=args.threshold, judged=judged):
        print(path)
    return 0


# ------------------------------------------------------------------ parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jp-inject",
        description="Japanese-specific prompt-injection test cases and evaluation harness.",
    )
    parser.add_argument("--version", action="version", version=f"jp-inject {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list", help="summarise the test cases")
    _add_filters(p)
    p.add_argument("--ids", action="store_true", help="print case ids only")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("show", help="print one test case as the model will see it")
    p.add_argument("case_id")
    p.add_argument("--defense", choices=dataset.DEFENSE_PROFILES, default="basic")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("transform", help="apply one orthographic transform to your own text")
    p.add_argument("technique", choices=sorted(TRANSFORMS))
    p.add_argument("text", nargs="?", help="text to transform (default: read stdin). "
                   f"For {', '.join(READING_TRANSFORMS)} pass the reading: hiragana with spaces between words.")
    p.set_defaults(func=cmd_transform)

    p = sub.add_parser("run", help="send test cases to a local OpenAI-compatible endpoint")
    p.add_argument("--model", required=True, help="model identifier as the server knows it")
    p.add_argument("--out", required=True, help="result file (JSON Lines); appended to and resumed")
    p.add_argument("--base-url", default=runner.DEFAULT_BASE_URL)
    p.add_argument("--label", help="model name to store in results (default: --model)")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--defense", choices=dataset.DEFENSE_PROFILES, default="basic",
                   help="basic: system prompts include one defensive sentence; none: they do not")
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--concurrency", type=int, default=1,
                   help="requests in flight at once (default 1). Useful when the server decodes in "
                        "parallel, e.g. LM Studio's 'Max Concurrent Predictions'")
    p.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="extra field for each request, e.g. --set reasoning_effort=low (repeatable)")
    p.add_argument("--extra", help="extra fields as one JSON object (quoting is awkward in PowerShell; prefer --set)")
    p.add_argument("--allow-remote", action="store_true",
                   help="allow a non-local endpoint (only with the provider's permission)")
    p.add_argument("--dry-run", action="store_true", help="show what would be sent, send nothing")
    _add_filters(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("judge", help="judge recorded generations and print one JSON row each")
    p.add_argument("results", nargs="+")
    p.add_argument("--out", default="-")
    p.set_defaults(func=cmd_judge)

    p = sub.add_parser("report", help="write tables and figures from result files")
    p.add_argument("results", nargs="+")
    p.add_argument("--out", required=True, help="output directory")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("sample", help="draw a blind sample for checking the judge by hand")
    p.add_argument("results", nargs="+")
    p.add_argument("--n", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="output directory")
    p.set_defaults(func=cmd_sample)

    p = sub.add_parser("agreement", help="compare a labelled sheet with the judge")
    p.add_argument("sheet")
    p.add_argument("key")
    p.set_defaults(func=cmd_agreement)

    p = sub.add_parser("detect", help="score the test cases with prompt-injection detectors (needs jp-inject[detect])")
    p.add_argument("--out", required=True, help="score file (JSON Lines)")
    p.add_argument("--detector", action="append", metavar="HF_MODEL_ID",
                   help="Hugging Face text-classification model (repeatable); pin a version with "
                        f"ID@COMMIT. Default: {', '.join(detect.DEFAULT_DETECTORS)}")
    p.add_argument("--device", default="cpu", help="cpu (default) or cuda")
    p.add_argument("--batch-size", type=int, default=8)
    _add_filters(p)
    p.set_defaults(func=cmd_detect)

    p = sub.add_parser("detect-report", help="write detection-rate tables from a score file")
    p.add_argument("scores")
    p.add_argument("--runs", nargs="+", help="model result files, to combine detection with attack success")
    p.add_argument("--threshold", type=float, default=detect.DEFAULT_THRESHOLD)
    p.add_argument("--out", required=True, help="output directory")
    p.set_defaults(func=cmd_detect_report)

    p = sub.add_parser("build", help="(maintainers) regenerate cases.jsonl from the sources")
    p.add_argument("--check", action="store_true", help="only verify that cases.jsonl is current")
    p.set_defaults(func=cmd_build)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _utf8_stdio()
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except BrokenPipeError:  # e.g. `jp-inject list | head`
        import os

        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
