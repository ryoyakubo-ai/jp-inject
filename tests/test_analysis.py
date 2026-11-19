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


def test_duplicate_records_keep_the_latest(results, tmp_path):
    first = json.loads(results.read_text(encoding="utf-8").splitlines()[0])
    extra = tmp_path / "again.jsonl"
    extra.write_text(json.dumps({**first, "content": "REPLACED"}) + "\n", encoding="utf-8")
    records = analysis.load_records([results, extra])
    assert len(records) == 545 * 2
    assert any(r["content"] == "REPLACED" for r in records)


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


def test_report_warns_when_settings_are_mixed(rows):
    assert "Warning" not in analysis.render_markdown(analysis.summarize_all(rows))
    altered = [dict(r, settings=r["settings"].replace("0.7", "0.2")) if i % 2 else r for i, r in enumerate(rows)]
    assert "recorded under 2 different settings" in analysis.render_markdown(analysis.summarize_all(altered))


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
