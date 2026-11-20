import csv
import json
import math

import pytest

import fake_server
from jp_inject import analysis, cli, dataset, runner, validation


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    """Run every case twice against the fake model."""
    path = tmp_path_factory.mktemp("run") / "fake.jsonl"
    with fake_server.serve() as url:
        config = runner.RunConfig(model="fake", out=path, base_url=url, repeats=2)
        counts = runner.run(dataset.load_cases(), config, log=lambda _: None)
    assert counts["failed"] == 0
    return path


@pytest.fixture(scope="module")
def rows(results):
    return analysis.judged_rows(analysis.load_records([results]))


def test_every_generation_is_judged(rows):
    assert len(rows) == 545 * 2
    assert {r["kind"] for r in rows} == {"attack", "benign", "comprehension"}
    labels = {r["label"] for r in rows if r["kind"] == "attack"}
    assert {"full", "partial", "mention", "none", "leak"} <= labels


def test_rerun_of_the_same_generation_keeps_the_most_recent(results, tmp_path):
    first = json.loads(results.read_text(encoding="utf-8").splitlines()[0])
    extra = tmp_path / "again.jsonl"
    newer = {**first, "content": "REPLACED", "started_at": "2099-01-01T00:00:00+00:00"}
    extra.write_text(json.dumps(newer) + "\n", encoding="utf-8")
    for order in ([results, extra], [extra, results]):  # the file order must not matter
        records = analysis.load_records(order)
        assert len(records) == 545 * 2
        assert sum(r["content"] == "REPLACED" for r in records) == 1


@pytest.mark.parametrize(
    "change, word",
    [
        ({"request": {"temperature": 0.2}}, "temperature"),
        ({"request": {"max_tokens": 64}}, "max_tokens"),
        ({"request": {"seed": 7}}, "seed"),
        ({"request": {"extra": {"top_p": 0.5}}}, "extra"),
        ({"model_id": "another-file"}, "model_id"),
        ({"base_url": "http://127.0.0.1:9/v1"}, "base_url"),
        ({"model_fingerprint": {"id": "fake", "quantization": "Q8"}}, "model_fingerprint"),
        ({"dataset_sha256": "0" * 64}, "dataset_sha256"),
    ],
)
def test_results_from_different_conditions_are_refused(results, tmp_path, change, word):
    first = json.loads(results.read_text(encoding="utf-8").splitlines()[0])
    other = {**first, **{k: v for k, v in change.items() if k != "request"}}
    other["request"] = {**first["request"], **change.get("request", {})}
    # A different generation, so nothing overlaps: mixing must still be caught.
    other["repeat"] = 99
    if "seed" not in change.get("request", {}):
        other["request"]["seed"] = first["request"]["seed"] + 99
    extra = tmp_path / "other.jsonl"
    extra.write_text(json.dumps(other) + "\n", encoding="utf-8")
    for order in ([results, extra], [extra, results]):
        with pytest.raises(analysis.MixedConditionsError, match=word):
            analysis.load_records(order)


def test_same_generation_under_two_temperatures_is_refused_in_either_order(results, tmp_path):
    first = json.loads(results.read_text(encoding="utf-8").splitlines()[0])
    other = {**first, "request": {**first["request"], "temperature": 0.2}}
    extra = tmp_path / "other.jsonl"
    extra.write_text(json.dumps(other) + "\n", encoding="utf-8")
    for order in ([results, extra], [extra, results]):
        with pytest.raises(analysis.MixedConditionsError, match="temperature"):
            analysis.load_records(order)


def test_unknown_field_in_one_record_does_not_hide_a_later_mismatch(results, tmp_path):
    # Reported in review: with the endpoint unknown in the first record, the
    # check compared everything against that record and let A and B through.
    import itertools

    first = json.loads(results.read_text(encoding="utf-8").splitlines()[0])

    def variant(name, base_url, repeat):
        record = {**first, "repeat": repeat, "request": {**first["request"], "seed": first["request"]["seed"] + repeat}}
        if base_url is None:
            record.pop("base_url", None)
        else:
            record["base_url"] = base_url
        path = tmp_path / f"{name}.jsonl"
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        return path

    unknown = variant("unknown", None, 91)
    a = variant("a", "http://127.0.0.1:1111/v1", 92)
    b = variant("b", "http://127.0.0.1:2222/v1", 93)
    for order in itertools.permutations([unknown, a, b]):
        with pytest.raises(analysis.MixedConditionsError, match="base_url"):
            analysis.load_records(order)
    # An unknown endpoint next to one known endpoint is still accepted (old result files).
    assert len(analysis.load_records([unknown, a])) == 2
    assert len(analysis.load_records([a, unknown])) == 2


def test_results_from_another_dataset_version_are_not_judged(results, tmp_path):
    # Reported in review: one file recorded with other test cases passed the
    # between-file check (nothing to compare with) and was judged against the
    # installed expectations, because the case ids still matched.
    stale = tmp_path / "stale.jsonl"
    lines = []
    for line in results.read_text(encoding="utf-8").splitlines()[:20]:
        record = json.loads(line)
        record["dataset_sha256"] = "0" * 64
        lines.append(json.dumps(record))
    stale.write_text("\n".join(lines) + "\n", encoding="utf-8")
    records = analysis.load_records([stale])
    with pytest.raises(analysis.DatasetMismatchError, match="000000000000"):
        analysis.judged_rows(records)
    # An explicit set of cases is the caller's responsibility.
    assert len(analysis.judged_rows(records, cases=dataset.load_cases())) == 20
    # Records written before hashes existed are still accepted.
    for record in records:
        record.pop("dataset_sha256")
    assert len(analysis.judged_rows(records)) == 20
    # The command line reports it instead of printing a table.
    with pytest.raises(SystemExit, match="installed test cases"):
        cli.main(["report", str(stale), "--out", str(tmp_path / "report")])


def test_different_labels_may_differ_in_condition(results, tmp_path):
    first = json.loads(results.read_text(encoding="utf-8").splitlines()[0])
    other = {**first, "model": "fake-cold", "request": {**first["request"], "temperature": 0.2}}
    extra = tmp_path / "other.jsonl"
    extra.write_text(json.dumps(other) + "\n", encoding="utf-8")
    assert len(analysis.load_records([results, extra])) == 545 * 2 + 1


def test_cli_reports_mixed_conditions(results, tmp_path):
    first = json.loads(results.read_text(encoding="utf-8").splitlines()[0])
    other = {**first, "repeat": 99, "request": {**first["request"], "temperature": 0.2, "seed": 99}}
    extra = tmp_path / "other.jsonl"
    extra.write_text(json.dumps(other) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="different conditions"):
        cli.main(["report", str(results), str(extra), "--out", str(tmp_path / "r")])


def test_summary_structure(rows):
    summaries = analysis.summarize_all(rows)
    assert list(summaries) == [("fake", "basic")]
    s = summaries[("fake", "basic")]
    assert s["generations"] == 1090 and s["invalid"] == 0
    assert s["benign"] == {"n": 60, "contaminated": 0}
    assert [v["variant"] for v in s["variants"]] == analysis.VARIANT_ORDER
    for v in s["variants"]:
        assert v["n"] == 40 and v["bases"] == 20
        assert v["low"] <= v["rate"] <= v["high"]
        assert v["strict"] <= v["rate"] <= v["loose"]
        if v["variant"] not in ("ja.plain", "en.plain"):
            assert v["vs_reference"]["bases"] == 20
    authority = next(v for v in s["variants"] if v["variant"] == "ja.authority")
    assert authority["vs_also"]["reference"] == "ja.keigo"
    assert {(g["payload"], g["env"]) for g in s["language_grid"]} == {
        ("ja", "ja"), ("en", "ja"), ("ja", "en"), ("en", "en")}
    assert {(i["ja"], i["en"]) for i in s["interactions"]} == {
        ("ja.keigo", "en.polite"), ("ja.indirect", "en.indirect"), ("ja.authority", "en.authority"),
        ("ja.note", "en.note"), ("ja.zero_width", "en.zero_width")}
    assert len(s["comprehension"]) == 11


def test_plain_reference_difference_is_computed_per_base(rows):
    s = analysis.summarize_all(rows)[("fake", "basic")]
    by = {v["variant"]: v for v in s["variants"]}
    # Every base has the same number of trials, so the mean per-base difference
    # equals the difference of pooled rates.
    for name in ("ja.keigo", "ja.romaji", "en.fullwidth"):
        ref = by[by[name]["vs_reference"]["reference"]]
        assert math.isclose(by[name]["vs_reference"]["diff"], by[name]["rate"] - ref["rate"], abs_tol=1e-9)


def test_report_files(rows, tmp_path):
    written = analysis.write_report(rows, tmp_path)
    names = {p.name for p in written}
    assert {"summary.md", "variants.csv", "summary.json"} <= names
    markdown = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "## fake (defense profile: basic)" in markdown and "`ja.kyujitai`" in markdown
    json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))  # strict JSON, no NaN
    with (tmp_path / "variants.csv").open(encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == len(analysis.VARIANT_ORDER)


def test_report_is_reproducible(rows, tmp_path):
    analysis.write_report(rows, tmp_path / "a")
    analysis.write_report(list(reversed(rows)), tmp_path / "b")
    for name in ("summary.md", "variants.csv"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_sample_is_stratified_blind_and_deterministic(rows, tmp_path):
    sample = validation.draw_sample(rows, n=150, seed=0)
    assert len(sample) == 150
    assert [r["case_id"] for r in sample] == [r["case_id"] for r in validation.draw_sample(rows, n=150, seed=0)]
    assert {r["label"] for r in sample} == {"full", "partial", "mention", "none", "leak"}
    cases = {c["id"]: c for c in dataset.load_cases()}
    sheet, key = validation.write_sheet(sample, cases, tmp_path)
    with sheet.open(encoding="utf-8-sig", newline="") as handle:
        table = list(csv.DictReader(handle))
    assert "judge_label" not in table[0] and all(r["human_label"] == "" for r in table)


def test_agreement(rows, tmp_path):
    sample = validation.draw_sample(rows, n=60, seed=1)
    cases = {c["id"]: c for c in dataset.load_cases()}
    sheet, key = validation.write_sheet(sample, cases, tmp_path)
    truth = {r["sample_id"]: r["success"] for r in sample}

    # A labeller who agrees with the judge everywhere except on two rows.
    with sheet.open(encoding="utf-8-sig", newline="") as handle:
        table = list(csv.DictReader(handle))
    flipped = {table[0]["sample_id"], table[1]["sample_id"]}
    for row in table:
        success = truth[row["sample_id"]] ^ (row["sample_id"] in flipped)
        row["human_label"] = "success" if success else "fail"
        row["labeller"] = "tester"
    table[-1]["human_label"] = ""  # one left blank
    with sheet.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=validation.SHEET_COLUMNS)
        writer.writeheader()
        writer.writerows(table)

    result = validation.agreement(sheet, key)
    assert result["labelled"] == 59 and result["skipped"] == 1
    assert len(result["disagreements"]) == 2
    assert math.isclose(result["agreement_in_sample"], 57 / 59)
    assert 0 < result["cohen_kappa"] < 1
    assert result["labellers"] == ["tester"]
    assert sum(result["confusion"].values()) == 59


def test_cli_end_to_end(results, tmp_path, capsys):
    assert cli.main(["report", str(results), "--out", str(tmp_path / "report")]) == 0
    assert cli.main(["sample", str(results), "--n", "20", "--out", str(tmp_path / "val")]) == 0
    assert cli.main(["agreement", str(tmp_path / "val" / "sheet.csv"), str(tmp_path / "val" / "key.jsonl")]) == 0
    assert cli.main(["list", "--kind", "attack"]) == 0
    assert cli.main(["show", "attack/support.leak/ignore/ja.keigo"]) == 0
    assert cli.main(["transform", "kyujitai", "変更"]) == 0
    assert cli.main(["build", "--check"]) == 0
    out = capsys.readouterr().out
    assert "變更" in out and "460 cases" in out


def test_cli_dry_run_sends_nothing(tmp_path, capsys):
    code = cli.main(["run", "--model", "x", "--out", str(tmp_path / "o.jsonl"), "--dry-run", "--limit", "3"])
    assert code == 0 and not (tmp_path / "o.jsonl").exists()
    assert "would send 3 cases x 5 repeats" in capsys.readouterr().out


def test_cli_refuses_remote_endpoint(tmp_path):
    code = cli.main(["run", "--model", "x", "--out", str(tmp_path / "o.jsonl"),
                     "--base-url", "https://api.example.com/v1", "--limit", "1"])
    assert code == 2


def test_cli_expands_wildcards_itself(results, tmp_path, capsys):
    pattern = str(results.parent / "*.jsonl")
    assert cli.main(["report", pattern, "--out", str(tmp_path / "r")]) == 0
    with pytest.raises(SystemExit):
        cli.main(["report", str(tmp_path / "nothing-*.jsonl"), "--out", str(tmp_path / "r2")])


def test_cli_set_adds_request_fields(tmp_path, capsys):
    code = cli.main(["run", "--model", "x", "--out", str(tmp_path / "o.jsonl"), "--dry-run", "--limit", "1",
                     "--set", "reasoning_effort=low", "--set", "top_p=0.9"])
    assert code == 0
    request = json.loads(capsys.readouterr().out.split("\n", 1)[1])
    assert request["reasoning_effort"] == "low" and request["top_p"] == 0.9
    assert cli.main(["run", "--model", "x", "--out", str(tmp_path / "o.jsonl"), "--dry-run", "--set", "oops"]) == 2


@pytest.mark.parametrize("field", ["temperature=0.2", "seed=42", "max_tokens=10", "model=y", "stream=true"])
def test_cli_refuses_set_for_fields_with_their_own_option(tmp_path, capsys, field):
    code = cli.main(["run", "--model", "x", "--out", str(tmp_path / "o.jsonl"), "--dry-run", "--set", field])
    assert code == 2
    assert "dedicated option" in capsys.readouterr().err


def test_japanese_vs_english_columns_use_the_same_base_attacks(rows):
    # Drop the English counterpart for half of the base attacks. The three
    # columns must still describe the same set of attacks: JA - EN == JA change - EN change.
    bases = sorted({r["base"] for r in rows if r["kind"] == "attack"})
    dropped = set(bases[::2])
    partial = [r for r in rows if not (r["variant"] == "en.polite" and r["base"] in dropped)]
    s = analysis.summarize(partial)
    keigo = next(i for i in s["interactions"] if i["ja"] == "ja.keigo")
    assert keigo["bases"] == len(bases) - len(dropped)
    assert math.isclose(keigo["difference"], keigo["ja_change"] - keigo["en_change"], abs_tol=1e-9)
    for i in analysis.summarize(rows)["interactions"]:
        assert math.isclose(i["difference"], i["ja_change"] - i["en_change"], abs_tol=1e-9)


def test_sheet_shows_the_answer_the_judge_saw(tmp_path):
    class Thinking(fake_server.Handler):
        think_prefix = "<think>The document mentions TV4-83MB. I should not output it.</think>"

    cases = [c for c in dataset.load_cases() if c["target"] == "summarize.hijack"][:40]
    path = tmp_path / "t.jsonl"
    with fake_server.serve(handler=Thinking) as url:
        runner.run(cases, runner.RunConfig(model="fake", out=path, base_url=url, repeats=1), log=lambda _: None)
    assert "<think>" in path.read_text(encoding="utf-8")  # raw output is kept in the result file
    judged = analysis.judged_rows(analysis.load_records([path]))
    assert all("<think>" not in r["content"] for r in judged)
    assert {r["label"] for r in judged} > {"none"}, "reasoning text must not decide the label"
    sample = validation.draw_sample(judged, n=20, seed=0)
    sheet, _ = validation.write_sheet(sample, {c["id"]: c for c in cases}, tmp_path)
    assert "<think>" not in sheet.read_text(encoding="utf-8-sig")


@pytest.mark.parametrize("text", ["=1+1", "+cmd", "-2", "@SUM(A1)", "＝1+1", "\t=1", "\n=1"])
def test_formula_like_cells_are_written_as_text(text):
    assert validation.as_text_cell(text) == "'" + text


def test_ordinary_cells_are_unchanged():
    for text in ("TV4-83MB", "会議は木曜日です。", "1+1=2", ""):
        assert validation.as_text_cell(text) == text


def test_formula_like_response_is_escaped_in_the_sheet(rows, tmp_path):
    sample = validation.draw_sample(rows, n=5, seed=0)
    sample[0] = dict(sample[0], content='=HYPERLINK("http://example.com","x")')
    cases = {c["id"]: c for c in dataset.load_cases()}
    sheet, _ = validation.write_sheet(sample, cases, tmp_path)
    with sheet.open(encoding="utf-8-sig", newline="") as handle:
        table = list(csv.DictReader(handle))
    assert table[0]["response"].startswith("'=HYPERLINK")
    assert not any(row["response"].startswith(("=", "+", "-", "@")) for row in table)
    assert not any(row["injected_text"].startswith(("=", "+", "-", "@")) for row in table)
