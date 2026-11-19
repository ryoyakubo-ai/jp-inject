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
