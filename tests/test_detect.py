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
    documents = [r for r in records if r["view"] == "message"]
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
    assert everything["views"]["message"]["ja.plain"]["rate"] == 1.0
    assert everything["controls"]["benign"]["flagged"] == 30


def test_attack_success_behind_a_filter_needs_both_to_fail(scores):
    out, _ = scores
    rows = detect.load_scores(out)
    cases = [c for c in dataset.load_cases() if c["kind"] == "attack" and c["variant"] in ("ja.plain", "ja.katakana")]
    judged = [{"model": "m", "defense": "basic", "kind": "attack", "valid": True, "success": True,
               "variant": c["variant"], "base": c["base"]} for c in cases]
    behind = detect.summarize(rows, judged=judged)["fake/keyword"]["behind_filter"]["m (defense profile: basic)"]
    assert behind["ja.plain"]["asr"]["rate"] == 1.0
    assert behind["ja.plain"]["asr_behind_filter"]["rate"] == pytest.approx(16 / 20)
    assert behind["ja.katakana"]["asr_behind_filter"]["rate"] == 1.0


def test_defense_profiles_of_one_model_are_not_pooled(scores):
    # Reported in review: the same model run with and without the defensive
    # sentence was averaged into one row.
    out, _ = scores
    rows = detect.load_scores(out)
    cases = [c for c in dataset.load_cases() if c["kind"] == "attack" and c["variant"] == "ja.plain"]
    judged = [{"model": "m", "defense": defense, "kind": "attack", "valid": True, "success": defense == "none",
               "variant": c["variant"], "base": c["base"]} for c in cases for defense in ("basic", "none")]
    behind = detect.summarize(rows, judged=judged)["fake/keyword"]["behind_filter"]
    assert set(behind) == {"m (defense profile: basic)", "m (defense profile: none)"}
    assert behind["m (defense profile: basic)"]["ja.plain"]["asr"]["rate"] == 0.0
    assert behind["m (defense profile: none)"]["ja.plain"]["asr"]["rate"] == 1.0


def test_a_detector_that_fails_to_load_keeps_its_earlier_scores(scores):
    # Reported in review: re-scoring a detector that could not be loaded
    # rewrote the file without it, down to zero bytes.
    out, _ = scores
    before = out.read_bytes()
    assert before

    def broken(model_id):
        raise detect.DetectorUnavailable("no network")

    result = detect.score_cases(["fake/keyword"], out, loader=broken, log=lambda m: None)
    assert result["scored"] == [] and result["skipped"][0]["detector"] == "fake/keyword"
    assert out.read_bytes() == before

    def half_broken(model_id):
        return broken(model_id) if model_id == "fake/keyword" else keyword_loader(model_id)

    detect.score_cases(["fake/keyword", "other/detector"], out, loader=half_broken, log=lambda m: None)
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [json.dumps(r, ensure_ascii=False) for r in records if r["detector"] == "fake/keyword"] == \
        before.decode("utf-8").splitlines()
    assert any(r["detector"] == "other/detector" for r in records)
    assert not out.with_name(out.name + ".partial").exists()


def test_a_scoring_error_leaves_the_file_as_it_was(scores):
    out, _ = scores
    before = out.read_bytes()

    def crashing(model_id):
        def score(texts):
            raise RuntimeError("out of memory")
        return score, {"detector": model_id}

    with pytest.raises(RuntimeError):
        detect.score_cases(["fake/keyword"], out, loader=crashing, log=lambda m: None)
    assert out.read_bytes() == before


def test_revision_pin_is_split_from_the_name_and_recorded(tmp_path):
    assert detect.split_revision("org/model@abc123") == ("org/model", "abc123")
    assert detect.split_revision("org/model") == ("org/model", None)
    local = tmp_path / "odd@name"
    local.mkdir()
    assert detect.split_revision(str(local)) == (str(local), None)

    seen = []

    def loader(spec):
        seen.append(spec)
        name, revision = detect.split_revision(spec)
        scorer, info = keyword_loader(name)
        return scorer, {**info, "revision": "f" * 40, "requested_revision": revision}

    out = tmp_path / "scores.jsonl"
    detect.score_cases(["org/model@abc123"], out, loader=loader, log=lambda m: None,
                       cases=dataset.load_cases()[:5])
    assert seen == ["org/model@abc123"]
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert {r["detector"] for r in records} == {"org/model"}
    meta = json.loads(out.with_name("scores.jsonl.meta.json").read_text(encoding="utf-8"))
    assert meta["invocations"][0]["scored"][0]["revision"] == "f" * 40
    assert meta["invocations"][0]["scored"][0]["requested_revision"] == "abc123"


def test_score_files_written_before_the_view_was_renamed_still_load(scores):
    out, _ = scores
    lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    for record in lines:
        if record["view"] == "message":
            record["view"] = "document"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in lines), encoding="utf-8")
    summary = detect.summarize(detect.load_scores(out))["fake/keyword"]
    assert set(summary["views"]) == {"message", "payload"}


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


def test_an_empty_selection_changes_nothing(scores, tmp_path, capsys):
    # Reported in review: a selection with no cases (a typo in --variant)
    # counted as a successful re-score and emptied the file.
    out, _ = scores
    before = out.read_bytes()
    with pytest.raises(detect.NothingToScore):
        detect.score_cases(["fake/keyword"], out, cases=[], loader=keyword_loader, log=lambda m: None)
    assert out.read_bytes() == before
    meta = out.with_name(out.name + ".meta.json")
    meta_before = meta.read_bytes()
    assert cli.main(["detect", "--out", str(out), "--variant", "ja.no-such-variant"]) == 2
    assert "no test cases selected" in capsys.readouterr().err
    assert out.read_bytes() == before and meta.read_bytes() == meta_before


def test_two_versions_of_one_detector_in_one_call_are_refused(scores):
    # Reported in review: both versions were stored under the same name, and
    # whichever came last silently won in the analysis.
    out, _ = scores
    before = out.read_bytes()
    with pytest.raises(detect.NothingToScore, match="fake/keyword"):
        detect.score_cases(["fake/keyword@a", "fake/keyword@b"], out, loader=keyword_loader, log=lambda m: None)
    with pytest.raises(detect.NothingToScore):
        detect.score_cases(["fake/keyword", "fake/keyword@b"], out, loader=keyword_loader, log=lambda m: None)
    assert out.read_bytes() == before


def test_normalize_folds_fullwidth_and_drops_zero_width_but_keeps_kana():
    assert detect.normalize_text("Ｉｇｎｏｒｅ　ａｌｌ") == "Ignore all"
    assert detect.normalize_text("無​視") == "無視"
    assert detect.normalize_text("むしして") == "むしして"


def test_normalized_view_lets_a_keyword_filter_catch_fullwidth(tmp_path):
    out = tmp_path / "scores.jsonl"
    detect.score_cases(["fake/keyword"], out, loader=keyword_loader, log=lambda m: None, normalize=True)
    rows = detect.load_scores(out)
    assert {row["view"] for row in rows} == {"message", "payload", "message_nfkc"}
    summary = detect.summarize(rows)["fake/keyword"]
    overview = summary["overview"]
    assert overview["en_fullwidth"]["flagged"] == 0
    assert overview["en_fullwidth_nfkc"]["flagged"] > 0
    assert overview["direct"]["n"] + overview["indirect"]["n"] == 460
    assert overview["benign"]["n"] == 30
    text = detect.render_markdown(detect.summarize(rows))
    assert "## Comparison" in text and "Flagged (after NFKC)" in text


def test_comparison_has_one_row_per_detector_and_a_csv(scores, tmp_path):
    out, _ = scores
    detect.score_cases(["other/detector"], out, loader=keyword_loader, log=lambda m: None)
    written = detect.write_report(detect.load_scores(out), tmp_path / "report")
    names = {path.name for path in written}
    assert "comparison.csv" in names
    table = detect.render_comparison(detect.summarize(detect.load_scores(out)))
    assert table.count("`fake/keyword`") == 1 and table.count("`other/detector`") == 1
    # without a normalised run, the NFKC columns are left out
    assert "NFKC" not in table
    lines = (tmp_path / "report" / "comparison.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "detector,threshold,column,flagged,n"


def test_generative_models_are_told_apart_from_classifiers():
    assert detect.is_generative(["LlamaForCausalLM"])
    assert detect.is_generative(["Llama4ForConditionalGeneration"])
    assert not detect.is_generative(["DebertaV2ForSequenceClassification"])
    assert not detect.is_generative(None)


def test_guard_score_reads_the_first_word_after_blank_lines():
    steps = [
        {"safe": 0.0, "unsafe": 0.0, "chosen_is_space": True},
        {"safe": 0.2, "unsafe": 0.6, "chosen_is_space": False},
        {"safe": 0.9, "unsafe": 0.0, "chosen_is_space": False},
    ]
    assert detect.guard_verdict_score(steps) == pytest.approx(0.75)
    assert detect.guard_verdict_score([{"safe": 0.0, "unsafe": 0.0, "chosen_is_space": False}]) == 0.0
    assert detect.guard_verdict_score([]) == 0.0
