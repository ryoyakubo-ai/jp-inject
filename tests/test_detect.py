"""Detector scoring and its analysis, with a stand-in for the real classifiers."""

import json

import pytest

from jp_inject import cli, dataset, detect


def keyword_loader(model_id):
    """A fake detector: flags text containing a plain override phrase."""
    if model_id == "gated/model":
        raise detect.DetectorUnavailable("gated")

    def score(texts):
        return [{"score": 0.99 if ("無視" in t or "Ignore" in t or "ignore" in t) else 0.01,
                 "tokens": len(t), "chunks": 1, "truncated": False} for t in texts]

    return score, {"detector": model_id, "revision": "test", "id2label": {"0": "SAFE", "1": "INJECTION"}}


@pytest.fixture()
def scores(tmp_path):
    out = tmp_path / "detectors" / "scores.jsonl"
    result = detect.score_cases(["fake/keyword", "gated/model"], out, loader=keyword_loader, log=lambda m: None)
    return out, result


def test_benign_index_reads_label_names():
    assert detect.benign_index({0: "SAFE", 1: "INJECTION"}) == 0
    assert detect.benign_index({0: "INJECTION", 1: "LEGIT"}) == 1
    assert detect.benign_index({0: "LABEL_0", 1: "LABEL_1"}) == 0
    assert detect.benign_index({0: "BENIGN", 1: "INJECTION", 2: "JAILBREAK"}) == 0
    assert detect.benign_index({0: "foo", 1: "bar"}) == 0


def test_pack_chunks_respects_budget_and_keeps_order():
    pieces = [("a", 3), ("b", 3), ("c", 3), ("huge", 20), ("d", 1)]
    assert detect.pack_chunks(pieces, 6) == ["a\nb", "c", "huge", "d"]
    assert detect.pack_chunks([], 6) == []


def test_every_case_is_scored_and_unavailable_detectors_are_reported(scores):
    out, result = scores
    assert result["scored"] == ["fake/keyword"]
    assert result["skipped"] == [{"detector": "gated/model", "reason": "gated"}]
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    cases = dataset.load_cases()
    documents = [r for r in records if r["view"] == "document"]
    assert {r["case_id"] for r in documents} == {c["id"] for c in cases}
    # the payload on its own is scored for every attack, but not where it is the whole document
    payloads = {r["case_id"] for r in records if r["view"] == "payload"}
    assert payloads == {c["id"] for c in cases if c["kind"] == "attack"}
    meta = json.loads((out.parent / "scores.jsonl.meta.json").read_text(encoding="utf-8"))
    assert meta["invocations"][0]["skipped"][0]["detector"] == "gated/model"


def test_rescoring_replaces_that_detector_only(scores):
    out, _ = scores
    detect.score_cases(["other/detector"], out, loader=keyword_loader, log=lambda m: None)
    detect.score_cases(["fake/keyword"], out, loader=keyword_loader, log=lambda m: None)
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    per_detector = {}
    for r in records:
        per_detector[r["detector"]] = per_detector.get(r["detector"], 0) + 1
    assert set(per_detector) == {"fake/keyword", "other/detector"}
    assert per_detector["fake/keyword"] == per_detector["other/detector"]


def test_summary_shows_what_a_keyword_filter_misses(scores):
    out, _ = scores
    summary = detect.summarize(detect.load_scores(out))["fake/keyword"]
    payload = summary["views"]["payload"]
    # Only the "ignore" strategy (4 of 20 base attacks) uses the phrase, and
    # only where it is spelled the standard way.
    assert payload["ja.plain"]["rate"] == pytest.approx(4 / 20)
    assert payload["en.plain"]["rate"] == pytest.approx(4 / 20)
    for respelled in ("ja.hiragana", "ja.katakana", "ja.hankaku", "ja.romaji", "ja.zero_width"):
        assert payload[respelled]["rate"] == 0.0, respelled
        assert payload[respelled]["vs_reference"]["diff"] == pytest.approx(-4 / 20)
        assert payload[respelled]["reference"] == "ja.plain"
    assert summary["controls"]["benign"] == {"n": 30, "flagged": 0}
    assert summary["controls"]["comprehension"]["n"] == 55


def test_threshold_is_applied_at_analysis_time(scores):
    out, _ = scores
    rows = detect.load_scores(out)
    everything = detect.summarize(rows, threshold=0.0)["fake/keyword"]
    assert everything["views"]["document"]["ja.plain"]["rate"] == 1.0
    assert everything["controls"]["benign"]["flagged"] == 30


def test_attack_success_behind_a_filter_needs_both_to_fail(scores):
    out, _ = scores
    rows = detect.load_scores(out)
    cases = [c for c in dataset.load_cases() if c["kind"] == "attack" and c["variant"] in ("ja.plain", "ja.katakana")]
    judged = [{"model": "m", "kind": "attack", "valid": True, "success": True,
               "variant": c["variant"], "base": c["base"]} for c in cases]
    behind = detect.summarize(rows, judged=judged)["fake/keyword"]["behind_filter"]["m"]
    assert behind["ja.plain"]["asr"]["rate"] == 1.0
    assert behind["ja.plain"]["asr_behind_filter"]["rate"] == pytest.approx(16 / 20)
    assert behind["ja.katakana"]["asr_behind_filter"]["rate"] == 1.0


def test_scores_from_another_dataset_version_are_refused(scores):
    out, _ = scores
    lines = out.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    record["dataset_sha256"] = "0" * 64
    out.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(detect.DatasetMismatchError):
        detect.load_scores(out)


def test_report_files_and_cli(scores, tmp_path, capsys):
    out, _ = scores
    assert cli.main(["detect-report", str(out), "--out", str(tmp_path / "report")]) == 0
    text = (tmp_path / "report" / "summary.md").read_text(encoding="utf-8")
    assert "## fake/keyword (threshold 0.5)" in text
    assert "`ja.katakana`" in text
    assert (tmp_path / "report" / "detectors.csv").read_text(encoding="utf-8").startswith("detector,threshold,view")
    json.loads((tmp_path / "report" / "summary.json").read_text(encoding="utf-8"))
