"""Send test cases to an OpenAI-compatible chat endpoint and record the answers.

Designed for a model served on your own machine (LM Studio, Ollama,
llama.cpp server, vLLM). Only the standard library is used.

* Results are appended to a JSON Lines file, one generation per line, and
  flushed immediately, so a run can be interrupted and resumed at any time.
* Work is done pass by pass (every case once, then every case a second time,
  ...), so stopping early still leaves balanced data.
* By default only loopback and private addresses are accepted. Sending
  injection tests to a hosted API usually needs the provider's permission;
  pass ``allow_remote=True`` only when you have it.
"""

from __future__ import annotations

import ipaddress
import json
import os
import platform
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from . import __version__
from .dataset import case_system, dataset_sha256

DEFAULT_BASE_URL = "http://localhost:1234/v1"  # LM Studio's default
API_KEY_ENV = "JP_INJECT_API_KEY"


class RemoteEndpointError(ValueError):
    """Raised when the endpoint is not local and remote use was not allowed."""


@dataclass
class RunConfig:
    model: str
    out: Path
    base_url: str = DEFAULT_BASE_URL
    label: str | None = None  # name stored in results; defaults to ``model``
    repeats: int = 5
    temperature: float = 0.7
    max_tokens: int = 1024
    seed: int = 0
    defense: str = "basic"
    timeout: float = 600.0
    retries: int = 3
    extra: dict[str, Any] = field(default_factory=dict)
    allow_remote: bool = False

    @property
    def model_label(self) -> str:
        return self.label or self.model


def is_local_url(url: str) -> bool:
    """True for loopback, private-network and ``.local`` hosts."""
    host = urllib.parse.urlsplit(url).hostname or ""
    if host in ("localhost",) or host.endswith((".localhost", ".local")):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_loopback or address.is_private


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _post_json(url: str, body: dict[str, Any], timeout: float, api_key: str | None) -> dict[str, Any]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_json(url: str, timeout: float, api_key: str | None) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    request = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def build_request(case: dict[str, Any], config: RunConfig, repeat: int) -> dict[str, Any]:
    """The chat-completions request body for one generation."""
    body: dict[str, Any] = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": case_system(case, config.defense)},
            {"role": "user", "content": case["user"]},
        ],
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "seed": config.seed + repeat,
        "stream": False,
    }
    body.update(config.extra)
    return body


def _extract(response: dict[str, Any]) -> dict[str, Any]:
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return {
        "content": message.get("content") or "",
        "reasoning": message.get("reasoning_content") or message.get("reasoning") or "",
        "finish_reason": choice.get("finish_reason"),
        "server_model": response.get("model"),
        "usage": response.get("usage"),
    }


def done_keys(path: Path) -> set[tuple[str, int, str, str]]:
    """Keys of generations already recorded in ``path``."""
    keys: set[tuple[str, int, str, str]] = set()
    if not path.exists():
        return keys
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # a line cut short by an interruption; it will be redone
            keys.add((record["case_id"], record["repeat"], record["defense"], record["model"]))
    return keys


def _model_info(config: RunConfig, api_key: str | None) -> dict[str, Any] | None:
    """Best-effort model metadata (quantisation etc.) from LM Studio's REST API."""
    root = config.base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    url = f"{root}/api/v0/models/{urllib.parse.quote(config.model, safe='')}"
    try:
        return _get_json(url, timeout=10, api_key=api_key)
    except Exception:  # not LM Studio, or an older version: simply not available
        return None


def _write_meta(config: RunConfig, api_key: str | None, planned: int) -> None:
    meta_path = config.out.with_suffix(config.out.suffix + ".meta.json")
    runs: list[dict[str, Any]] = []
    if meta_path.exists():
        try:
            runs = json.loads(meta_path.read_text(encoding="utf-8")).get("invocations", [])
        except (json.JSONDecodeError, AttributeError):
            runs = []
    runs.append({
        "started_at": _now(),
        "jp_inject_version": __version__,
        "dataset_sha256": dataset_sha256(),
        "model": config.model,
        "label": config.model_label,
        "base_url": config.base_url,
        "defense": config.defense,
        "repeats": config.repeats,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "seed": config.seed,
        "extra": config.extra,
        "planned_generations": planned,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "model_info": _model_info(config, api_key),
    })
    meta_path.write_text(
        json.dumps({"invocations": runs}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _order(cases: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group cases that share a system prompt and carrier (helps prompt caching)."""
    return sorted(
        cases,
        key=lambda c: (c["kind"], c["scenario"], c["env"], -1 if c["carrier"] is None else c["carrier"], c["id"]),
    )


def run(
    cases: Iterable[dict[str, Any]],
    config: RunConfig,
    *,
    log: Callable[[str], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, int]:
    """Run ``cases`` against the endpoint. Returns counts of what happened."""
    if log is None:
        def log(message: str) -> None:
            print(message, file=sys.stderr, flush=True)

    if not config.allow_remote and not is_local_url(config.base_url):
        raise RemoteEndpointError(
            f"{config.base_url} is not a local address. jp-inject is meant for models you "
            "host yourself; many hosted APIs do not allow injection testing without prior "
            "permission. Pass --allow-remote if you have that permission."
        )

    api_key = os.environ.get(API_KEY_ENV)
    ordered = _order(cases)
    config.out.parent.mkdir(parents=True, exist_ok=True)
    already = done_keys(config.out)
    todo = [
        (repeat, case)
        for repeat in range(config.repeats)
        for case in ordered
        if (case["id"], repeat, config.defense, config.model_label) not in already
    ]
    total = len(ordered) * config.repeats
    log(f"{total} generations planned, {total - len(todo)} already recorded, {len(todo)} to run")
    counts = {"planned": total, "skipped": total - len(todo), "done": 0, "failed": 0}
    if not todo:
        return counts

    _write_meta(config, api_key, len(todo))
    url = config.base_url.rstrip("/") + "/chat/completions"
    sha = dataset_sha256()
    started = time.monotonic()

    with config.out.open("a", encoding="utf-8", newline="\n") as handle:
        for index, (repeat, case) in enumerate(todo, start=1):
            body = build_request(case, config, repeat)
            began_at, began = _now(), time.monotonic()
            result = None
            for attempt in range(1, config.retries + 1):
                try:
                    result = _extract(_post_json(url, body, config.timeout, api_key))
                    break
                except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as error:
                    log(f"  attempt {attempt}/{config.retries} failed for {case['id']}: {error}")
                    if attempt < config.retries:
                        sleep(min(30.0, 2.0 ** attempt))
            if result is None:
                counts["failed"] += 1
                continue

            record = {
                "case_id": case["id"],
                "repeat": repeat,
                "defense": config.defense,
                "model": config.model_label,
                **result,
                "request": {
                    "temperature": body["temperature"],
                    "max_tokens": body["max_tokens"],
                    "seed": body["seed"],
                    "extra": config.extra,
                },
                "elapsed_s": round(time.monotonic() - began, 3),
                "started_at": began_at,
                "jp_inject_version": __version__,
                "dataset_sha256": sha,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            counts["done"] += 1

            if index % 25 == 0 or index == len(todo):
                elapsed = time.monotonic() - started
                remaining = elapsed / index * (len(todo) - index)
                log(f"  {index}/{len(todo)} done, about {remaining / 60:.0f} min left")

    if counts["failed"]:
        log(f"{counts['failed']} generations failed; run the same command again to retry them")
    return counts
