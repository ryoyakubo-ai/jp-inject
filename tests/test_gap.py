"""Detector misses and model compliance side by side."""

import csv
import io
import json

import pytest

import fake_server
from jp_inject import analysis, cli, dataset, detect, gap, runner
from test_detect import keyword_loader


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    """Scores from the keyword stand-in (all views) and three runs of the fake model."""
    root = tmp_path_factory.mktemp("gap")
    scores = root / "scores.jsonl"
    detect.score_cases(["fake/keyword"], scores, loader=keyword_loader, log=lambda m: None,
                       normalize=["nfkc", "fold"])
    cases = [c for c in dataset.load_cases() if c["family"] in ("language", "orthographic", "control")]
    runs = []
    with fake_server.serve() as url:
        for level in (None, "nfkc"):
            path = root / f"fake-{level}.jsonl"
            config = runner.RunConfig(model="fake", out=path, base_url=url, repeats=2, normalize=level)
            assert runner.run(cases, config, log=lambda _: None)["failed"] == 0
            runs.append(path)
    return scores, runs


def _summary(inputs, runs=None):
    scores, all_runs = inputs
    judged = analysis.judged_rows(analysis.load_records(runs if runs is not None else all_runs))
    return gap.summarize(detect.load_scores(scores), judged)


def test_pipelines_are_ordered_as_a_filter_requires(inputs):
    pair = _summary(inputs)["pairs"][0]
    assert pair["detector"] == "fake/keyword" and pair["model_id"] == "fake"
    assert pair["levels"] == ["nfkc", "fold"] and pair["model_levels"] == ["nfkc"]
    assert pair["model_labels"] == {"none": "fake", "nfkc": "fake+nfkc"}
    for entry in pair["conditions"].values():
        assert entry["B"]["rate"] <= entry["A"]["rate"] + 1e-9  # a filter never adds successes
        for cell in entry["normalized"].values():
            assert cell["C"]["rate"] <= entry["A"]["rate"] + 1e-9


def test_normalising_for_the_detector_closes_the_fullwidth_gap(inputs):
    table = _summary(inputs)["pairs"][0]["conditions"]
    fullwidth, plain = table["en.fullwidth"], table["en.plain"]
    # The keyword filter cannot read full-width letters, and reads them after NFKC.
    assert fullwidth["miss"]["rate"] > plain["miss"]["rate"]
    assert fullwidth["normalized"]["nfkc"]["miss"]["rate"] == pytest.approx(plain["miss"]["rate"])
    assert fullwidth["normalized"]["nfkc"]["miss_vs_raw"]["diff"] < 0
    assert fullwidth["normalized"]["nfkc"]["C"]["rate"] <= fullwidth["B"]["rate"]
    # Hiragana is not changed by normalisation, so nothing changes for it.
    hiragana = table["ja.hiragana"]
    assert hiragana["normalized"]["fold"]["miss_vs_raw"]["diff"] == 0


def test_a_normalised_model_run_gives_pipeline_d(inputs):
    table = _summary(inputs)["pairs"][0]["conditions"]
    cell = table["en.fullwidth"]["normalized"]["nfkc"]
    # After NFKC the full-width message is the plain message, so the fake model
    # (a function of its input) behaves exactly as it does for en.plain.
    assert cell["asr"]["rate"] == pytest.approx(table["en.plain"]["normalized"]["nfkc"]["asr"]["rate"])
    assert {"D", "D_vs_C", "D_vs_B"} <= set(cell)
    assert "D" not in table["en.fullwidth"]["normalized"]["fold"]


def test_without_a_normalised_run_only_a_b_c_are_reported(inputs):
    pair = _summary(inputs, runs=inputs[1][:1])["pairs"][0]
    assert pair["model_levels"] == []
    assert "D" not in pair["conditions"]["en.fullwidth"]["normalized"]["nfkc"]
    assert set(pair["worst"]) == {"A", "B", "C:nfkc", "C:fold"}


def test_a_normalised_run_alone_is_refused(inputs):
    with pytest.raises(gap.GapInputError):
        _summary(inputs, runs=inputs[1][1:])


def test_text_identity_counts_what_each_level_undoes():
    identity = gap.text_identity()
    assert identity["nfkc"]["en.fullwidth"] == {"n": 20, "identical": 20}
    assert identity["nfkc"]["ja.zero_width"] == {"n": 20, "identical": 20}
    assert identity["nfkc"]["ja.kyujitai"]["identical"] == 0
    assert identity["fold"]["ja.kyujitai"]["identical"] == 20
    assert identity["fold"]["ja.lookalike"]["identical"] == 20
    assert identity["fold"]["ja.hiragana"]["identical"] == 0
    assert identity["fold"]["ja.romaji"]["identical"] == 0


def test_controls_are_counted_per_view(inputs):
    controls = _summary(inputs)["controls"]["fake/keyword"]
    assert controls["message"]["benign"]["n"] == 30
    assert controls["message_nfkc"]["benign"]["n"] == 30
    assert controls["message_fold"]["comprehension"]["n"] == 55


def test_cli_writes_the_report(inputs, tmp_path, capsys):
    scores, runs = inputs
    out = tmp_path / "gap"
    assert cli.main(["gap-report", str(scores), "--runs", *map(str, runs), "--out", str(out)]) == 0
    text = (out / "summary.md").read_text(encoding="utf-8")
    assert "fake/keyword in front of fake" in text and "After `nfkc` normalisation" in text
    rows = list(csv.DictReader(io.StringIO((out / "gap.csv").read_text(encoding="utf-8"))))
    assert {"miss", "asr", "B", "C", "D"} <= {row["measure"] for row in rows}
    json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert cli.main(["gap-report", str(scores), "--runs", str(runs[1]), "--out", str(out)]) == 2
    assert "without input normalisation" in capsys.readouterr().err


def test_runs_that_differ_in_more_than_normalisation_are_refused(inputs, tmp_path):
    scores, runs = inputs
    changed = tmp_path / "hotter+nfkc.jsonl"
    with changed.open("w", encoding="utf-8", newline="\n") as handle:
        for line in runs[1].read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            record["request"]["temperature"] = 1.3
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    judged = analysis.judged_rows(analysis.load_records([runs[0], changed]))
    with pytest.raises(gap.GapInputError, match="temperature"):
        gap.summarize(detect.load_scores(scores), judged)
    assert cli.main(["gap-report", str(scores), "--runs", str(runs[0]), str(changed),
                     "--out", str(tmp_path / "out")]) == 2
