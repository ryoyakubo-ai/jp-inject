import json

import pytest

import fake_server
from jp_inject import dataset, runner


@pytest.fixture()
def few_cases():
    cases = dataset.load_cases()
    return [c for c in cases if c["variant"] in ("ja.plain", "ja.keigo")][:6]


def _config(url, tmp_path, **overrides):
    defaults = dict(model="fake", out=tmp_path / "out.jsonl", base_url=url, repeats=2, retries=2)
    defaults.update(overrides)
    return runner.RunConfig(**defaults)


def _read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_run_records_everything_needed_to_reproduce(few_cases, tmp_path):
    with fake_server.serve() as url:
        config = _config(url, tmp_path, extra={"reasoning_effort": "low"})
        counts = runner.run(few_cases, config, log=lambda _: None)
        sent = list(fake_server.Handler.requests_seen)
    assert counts == {"planned": 12, "skipped": 0, "done": 12, "failed": 0}
    records = _read(config.out)
    assert len(records) == 12
    first = records[0]
    assert first["model"] == "fake" and first["defense"] == "basic"
    assert first["request"] == {"temperature": 0.7, "max_tokens": 1024, "seed": 0,
                                "extra": {"reasoning_effort": "low"}}
    assert first["dataset_sha256"] == dataset.dataset_sha256()
    assert first["finish_reason"] == "stop" and first["content"]
    assert {r["request"]["seed"] for r in records} == {0, 1}
    assert sent[0]["reasoning_effort"] == "low"
    assert sent[0]["messages"][0]["role"] == "system"
    meta = json.loads((tmp_path / "out.jsonl.meta.json").read_text(encoding="utf-8"))
    assert meta["invocations"][0]["model_info"]["quantization"] == "Q0_FAKE"


def test_run_goes_pass_by_pass(few_cases, tmp_path):
    with fake_server.serve() as url:
        config = _config(url, tmp_path)
        runner.run(few_cases, config, log=lambda _: None)
    repeats = [r["repeat"] for r in _read(config.out)]
    assert repeats == sorted(repeats)


def test_run_resumes_without_repeating_work(few_cases, tmp_path):
    with fake_server.serve() as url:
        config = _config(url, tmp_path, repeats=1)
        runner.run(few_cases[:2], config, log=lambda _: None)
        # A line cut short by an interruption must not break the next run.
        with config.out.open("a", encoding="utf-8") as handle:
            handle.write('{"case_id": "attack/trunc')
            handle.write("\n")
        before = len(fake_server.Handler.requests_seen)
        counts = runner.run(few_cases, config, log=lambda _: None)
        after = len(fake_server.Handler.requests_seen)
    assert counts["skipped"] == 2 and counts["done"] == 4
    assert after - before == 4


def test_defense_profile_changes_the_system_prompt(few_cases, tmp_path):
    with fake_server.serve() as url:
        runner.run(few_cases[:1], _config(url, tmp_path, repeats=1, defense="none"), log=lambda _: None)
        sent = fake_server.Handler.requests_seen[0]["messages"][0]["content"]
    assert sent == few_cases[0]["system_no_defense"]


def test_temporary_failures_are_retried(few_cases, tmp_path):
    with fake_server.serve() as url:
        fake_server.Handler.fail_first = 1
        counts = runner.run(few_cases[:1], _config(url, tmp_path, repeats=1), log=lambda _: None,
                            sleep=lambda _: None)
    assert counts["done"] == 1 and counts["failed"] == 0


def test_persistent_failures_are_counted_and_left_for_a_rerun(few_cases, tmp_path):
    with fake_server.serve() as url:
        fake_server.Handler.fail_first = 99
        config = _config(url, tmp_path, repeats=1)
        counts = runner.run(few_cases[:2], config, log=lambda _: None, sleep=lambda _: None)
    assert counts["failed"] == 2 and counts["done"] == 0
    assert _read(config.out) == []


def test_remote_endpoints_are_refused_by_default(few_cases, tmp_path):
    config = _config("https://api.example.com/v1", tmp_path)
    with pytest.raises(runner.RemoteEndpointError):
        runner.run(few_cases, config, log=lambda _: None)


@pytest.mark.parametrize(
    "url, local",
    [
        ("http://localhost:1234/v1", True),
        ("http://127.0.0.1:1234/v1", True),
        ("http://[::1]:8080/v1", True),
        ("http://192.168.1.20:1234/v1", True),
        ("http://my-pc.local:1234/v1", True),
        ("https://api.example.com/v1", False),
        ("https://8.8.8.8/v1", False),
    ],
)
def test_is_local_url(url, local):
    assert runner.is_local_url(url) is local


def test_api_key_is_sent_but_never_stored(few_cases, tmp_path, monkeypatch):
    monkeypatch.setenv(runner.API_KEY_ENV, "secret-key-123")
    with fake_server.serve() as url:
        config = _config(url, tmp_path, repeats=1)
        runner.run(few_cases[:1], config, log=lambda _: None)
    stored = config.out.read_text(encoding="utf-8") + (tmp_path / "out.jsonl.meta.json").read_text(encoding="utf-8")
    assert "secret-key-123" not in stored


def test_redirects_are_refused_and_the_api_key_never_leaves(few_cases, tmp_path, monkeypatch):
    # urllib would follow the redirect and repeat the Authorization header
    # against the new host. The runner must not.
    monkeypatch.setenv(runner.API_KEY_ENV, "secret-key-123")

    class Redirecting(fake_server.Handler):
        pass

    with fake_server.serve() as elsewhere:
        Redirecting.redirect_to = elsewhere.replace("127.0.0.1", "localhost") + "/chat/completions"
        with fake_server.serve(handler=Redirecting) as url:
            with pytest.raises(runner.RedirectError):
                runner.run(few_cases[:1], _config(url, tmp_path, repeats=1), log=lambda _: None)
            # the configured host was asked (metadata GET, then the chat POST) ...
            assert [method for method, _, _ in Redirecting.hits] == ["GET", "POST"]
        # ... and nothing at all reached the redirect target
        assert fake_server.Handler.hits == []
    assert not (tmp_path / "out.jsonl").exists() or _read(tmp_path / "out.jsonl") == []


@pytest.mark.parametrize("tail", ['{"case_id": "attack/trunc', '{"case_id": "x", "repeあ'])
def test_resume_after_a_line_cut_off_without_newline_loses_nothing(few_cases, tmp_path, tail):
    with fake_server.serve() as url:
        config = _config(url, tmp_path, repeats=1)
        runner.run(few_cases[:2], config, log=lambda _: None)
        with config.out.open("a", encoding="utf-8") as handle:
            handle.write(tail)  # interrupted mid-record: no newline at the end
        counts = runner.run(few_cases, config, log=lambda _: None)
    assert counts["done"] == 4 and counts["skipped"] == 2
    lines = config.out.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]  # every line is a complete record
    assert len(records) == 6
    assert {r["case_id"] for r in records} == {c["id"] for c in few_cases}


def test_repair_tail(tmp_path):
    path = tmp_path / "r.jsonl"
    assert runner.repair_tail(path) == "ok"  # missing file
    path.write_text('{"a": 1}\n', encoding="utf-8")
    assert runner.repair_tail(path) == "ok"
    path.write_text('{"a": 1}\n{"a": 2}', encoding="utf-8")  # complete record, newline lost
    assert runner.repair_tail(path) == "newline_added"
    assert path.read_text(encoding="utf-8") == '{"a": 1}\n{"a": 2}\n'
    path.write_text('{"a": 1}\n{"a":', encoding="utf-8")
    assert runner.repair_tail(path) == "fragment_removed"
    assert path.read_text(encoding="utf-8") == '{"a": 1}\n'
    path.write_text('{"a":', encoding="utf-8")  # nothing but a fragment
    assert runner.repair_tail(path) == "fragment_removed"
    assert path.read_text(encoding="utf-8") == ""


@pytest.mark.parametrize(
    "change, word",
    [({"temperature": 0.2}, "temperature"), ({"max_tokens": 64}, "max_tokens"), ({"seed": 7}, "seed"),
     ({"extra": {"top_p": 0.5}}, "extra"), ({"model": "other", "label": "fake"}, "model_id")],
)
def test_resuming_with_different_settings_is_refused(few_cases, tmp_path, change, word):
    with fake_server.serve() as url:
        runner.run(few_cases[:2], _config(url, tmp_path, repeats=1), log=lambda _: None)
        before = (tmp_path / "out.jsonl").read_text(encoding="utf-8")
        with pytest.raises(runner.SettingsMismatchError, match=word):
            runner.run(few_cases, _config(url, tmp_path, repeats=1, **change), log=lambda _: None)
    assert (tmp_path / "out.jsonl").read_text(encoding="utf-8") == before  # nothing was appended


def test_resuming_after_the_dataset_changed_is_refused(few_cases, tmp_path, monkeypatch):
    with fake_server.serve() as url:
        runner.run(few_cases[:2], _config(url, tmp_path, repeats=1), log=lambda _: None)
        monkeypatch.setattr(runner, "dataset_sha256", lambda: "0" * 64)
        with pytest.raises(runner.SettingsMismatchError, match="dataset_sha256"):
            runner.run(few_cases, _config(url, tmp_path, repeats=1), log=lambda _: None)


def test_other_models_and_defenses_may_share_a_file(few_cases, tmp_path):
    with fake_server.serve() as url:
        runner.run(few_cases[:2], _config(url, tmp_path, repeats=1), log=lambda _: None)
        runner.run(few_cases[:2], _config(url, tmp_path, repeats=1, defense="none", temperature=0.1),
                   log=lambda _: None)
        runner.run(few_cases[:2], _config(url, tmp_path, repeats=1, label="second", max_tokens=99),
                   log=lambda _: None)
    assert len(_read(tmp_path / "out.jsonl")) == 6


def test_reserved_fields_cannot_be_passed_as_extra(tmp_path):
    for key in runner.RESERVED_REQUEST_FIELDS:
        with pytest.raises(ValueError, match="dedicated option"):
            runner.RunConfig(model="m", out=tmp_path / "o.jsonl", extra={key: 1})


def test_same_command_resumes_with_options_and_extra_fields(few_cases, tmp_path):
    # Whatever was passed the first time, passing it again must resume, not be refused.
    settings = dict(repeats=1, temperature=0.2, seed=42, max_tokens=64, extra={"top_p": 0.9})
    with fake_server.serve() as url:
        runner.run(few_cases[:2], _config(url, tmp_path, **settings), log=lambda _: None)
        counts = runner.run(few_cases, _config(url, tmp_path, **settings), log=lambda _: None)
    assert counts["skipped"] == 2 and counts["done"] == 4
    records = _read(tmp_path / "out.jsonl")
    assert {r["request"]["temperature"] for r in records} == {0.2}
    assert {r["request"]["seed"] for r in records} == {42}


def test_resuming_against_another_server_is_refused(few_cases, tmp_path):
    class Other(fake_server.Handler):
        pass

    with fake_server.serve() as url:
        runner.run(few_cases[:2], _config(url, tmp_path, repeats=1), log=lambda _: None)
        with fake_server.serve(handler=Other) as elsewhere:
            with pytest.raises(runner.SettingsMismatchError, match="base_url"):
                runner.run(few_cases, _config(elsewhere, tmp_path, repeats=1), log=lambda _: None)
            assert Other.requests_seen == []  # nothing was generated on the other server


def test_resuming_after_the_server_swapped_the_model_file_is_refused(few_cases, tmp_path):
    with fake_server.serve() as url:
        runner.run(few_cases[:2], _config(url, tmp_path, repeats=1), log=lambda _: None)
        fake_server.Handler.quantization = "Q8_OTHER"  # same id, different file
        try:
            with pytest.raises(runner.SettingsMismatchError, match="model_fingerprint"):
                runner.run(few_cases, _config(url, tmp_path, repeats=1), log=lambda _: None)
        finally:
            fake_server.Handler.quantization = "Q0_FAKE"


def test_records_carry_the_full_condition(few_cases, tmp_path):
    with fake_server.serve() as url:
        runner.run(few_cases[:1], _config(url, tmp_path, repeats=1), log=lambda _: None)
        record = _read(tmp_path / "out.jsonl")[0]
        assert record["base_url"] == url
    assert record["model_id"] == "fake"
    # volatile metadata (load state, context length) is not part of the fingerprint
    assert record["model_fingerprint"] == {"id": "fake", "arch": "none", "quantization": "Q0_FAKE"}


def test_concurrent_run_records_the_same_generations(tmp_path):
    cases = dataset.load_cases()[:40]
    with fake_server.serve() as url:
        one = runner.RunConfig(model="fake", out=tmp_path / "one.jsonl", base_url=url, repeats=2)
        four = runner.RunConfig(model="fake", out=tmp_path / "four.jsonl", base_url=url, repeats=2, concurrency=4)
        assert runner.run(cases, one, log=lambda _: None)["done"] == 80
        assert runner.run(cases, four, log=lambda _: None)["done"] == 80
    def keyed(path):
        return {(r["case_id"], r["repeat"]): r["content"] for r in _read(path)}
    assert keyed(tmp_path / "one.jsonl") == keyed(tmp_path / "four.jsonl")
    repeats = [r["repeat"] for r in _read(tmp_path / "four.jsonl")]
    assert repeats == sorted(repeats), "passes must not interleave"
    meta = json.loads((tmp_path / "four.jsonl.meta.json").read_text(encoding="utf-8"))
    assert meta["invocations"][0]["concurrency"] == 4


def test_concurrent_run_resumes_and_counts_failures(few_cases, tmp_path):
    with fake_server.serve() as url:
        config = _config(url, tmp_path, repeats=1, concurrency=3, retries=1)
        fake_server.Handler.fail_first = 2
        counts = runner.run(few_cases, config, log=lambda _: None, sleep=lambda _: None)
        assert counts["failed"] == 2 and counts["done"] == 4
        again = runner.run(few_cases, config, log=lambda _: None, sleep=lambda _: None)
    assert again["skipped"] == 4 and again["done"] == 2
    assert len(_read(tmp_path / "out.jsonl")) == 6


def test_concurrent_run_stops_on_redirect(few_cases, tmp_path):
    class Redirecting(fake_server.Handler):
        redirect_to = "http://127.0.0.1:9/elsewhere"

    with fake_server.serve(handler=Redirecting) as url:
        with pytest.raises(runner.RedirectError):
            runner.run(few_cases, _config(url, tmp_path, repeats=1, concurrency=4), log=lambda _: None)


def test_concurrency_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="concurrency"):
        runner.RunConfig(model="m", out=tmp_path / "o.jsonl", concurrency=0)


def test_normalised_run_sends_normalised_text_under_its_own_label(tmp_path):
    cases = [c for c in dataset.load_cases() if c["variant"] == "en.fullwidth"][:2]
    with fake_server.serve() as url:
        config = _config(url, tmp_path, repeats=1, normalize="nfkc")
        runner.run(cases, config, log=lambda _: None)
        sent = list(fake_server.Handler.requests_seen)
    by_user = {request["messages"][1]["content"] for request in sent[-2:]}
    assert all("Ｉ" not in text and "\u3000" not in text for text in by_user)
    records = _read(config.out)
    assert {r["model"] for r in records} == {"fake+nfkc"}
    assert {r["model_id"] for r in records} == {"fake"}
    assert {r["input_normalization"] for r in records} == {"nfkc"}
    meta = json.loads((tmp_path / "out.jsonl.meta.json").read_text(encoding="utf-8"))
    assert meta["invocations"][0]["input_normalization"] == "nfkc"


def test_a_plain_run_cannot_continue_a_normalised_file(few_cases, tmp_path):
    with fake_server.serve() as url:
        runner.run(few_cases, _config(url, tmp_path, repeats=1, normalize="nfkc", label="same"),
                   log=lambda _: None)
        with pytest.raises(runner.SettingsMismatchError, match="input_normalization"):
            runner.run(few_cases, _config(url, tmp_path, repeats=1, label="same"), log=lambda _: None)


def test_unknown_normalisation_level_is_refused(tmp_path):
    with pytest.raises(ValueError, match="normalisation"):
        runner.RunConfig(model="fake", out=tmp_path / "out.jsonl", normalize="kana")
