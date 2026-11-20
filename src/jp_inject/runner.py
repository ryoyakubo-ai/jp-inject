"""Send test cases to an OpenAI-compatible chat endpoint and record the answers.

Designed for a model served on your own machine (LM Studio, Ollama,
llama.cpp server, vLLM). Only the standard library is used.

* Results are appended to a JSON Lines file, one generation per line, and
  flushed immediately, so a run can be interrupted and resumed at any time.
* Work is done pass by pass (every case once, then every case a second time,
  ...), so stopping early still leaves balanced data. Within a pass, several
  requests can be in flight at once (``concurrency``) for servers that decode
  in parallel.
* By default only loopback and private addresses are accepted. Sending
  injection tests to a hosted API usually needs the provider's permission;
  pass ``allow_remote=True`` only when you have it.
* HTTP redirects are never followed, and local endpoints are contacted
  directly (no proxy), so prompts and the optional API key go to the
  configured host and nowhere else.
* A result file holds one experimental condition per model: resuming with
  different settings or a different dataset version is refused.
* With ``normalize``, the user message is normalised before it is sent, as an
  application would do that normalises its input for the model too. Such a
  run is a different condition and is stored under its own model label.
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from . import __version__, conditions
from .dataset import case_system, dataset_sha256
from .normalize import LEVELS as NORMALIZE_LEVELS
from .normalize import normalize as normalize_input

DEFAULT_BASE_URL = "http://localhost:1234/v1"  # LM Studio's default
API_KEY_ENV = "JP_INJECT_API_KEY"

# Request fields that have their own option. They may not be passed through
# ``extra``: a value set there would bypass the option and the bookkeeping
# that keeps a result file to one condition.
RESERVED_REQUEST_FIELDS = ("model", "messages", "temperature", "max_tokens", "seed", "stream")


class RemoteEndpointError(ValueError):
    """Raised when the endpoint is not local and remote use was not allowed."""


class RedirectError(RuntimeError):
    """Raised when the endpoint answers with an HTTP redirect."""


class SettingsMismatchError(ValueError):
    """Raised when a result file was recorded under different settings."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirects instead of following them.

    urllib would otherwise repeat the request, Authorization header included,
    against whatever host the Location header names.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RedirectError(
            f"{req.full_url} answered with a redirect to {newurl}; redirects are not followed. "
            "Point --base-url at the final address."
        )


def _opener(base_url: str) -> urllib.request.OpenerDirector:
    handlers: list[urllib.request.BaseHandler] = [_NoRedirect()]
    if is_local_url(base_url):
        handlers.append(urllib.request.ProxyHandler({}))  # never route local traffic via a proxy
    return urllib.request.build_opener(*handlers)


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
    concurrency: int = 1  # requests in flight at once; match the server's parallel slots
    normalize: str | None = None  # a level of jp_inject.normalize applied to the user message

    def __post_init__(self) -> None:
        if self.concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        if self.normalize is not None and self.normalize not in NORMALIZE_LEVELS:
            raise ValueError(
                f"unknown normalisation level {self.normalize!r}; choose from {', '.join(NORMALIZE_LEVELS)}"
            )
        reserved = sorted(set(self.extra) & set(RESERVED_REQUEST_FIELDS))
        if reserved:
            raise ValueError(
                f"{', '.join(reserved)} cannot be set as an extra request field; "
                "use the dedicated option (--temperature, --max-tokens, --seed, --model)."
            )

    @property
    def model_label(self) -> str:
        """Name stored in results. A normalised run gets its own default name
        (``model+nfkc``), so that it cannot be pooled with the plain run."""
        if self.label:
            return self.label
        return f"{self.model}+{self.normalize}" if self.normalize else self.model


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


def _post_json(
    opener: urllib.request.OpenerDirector, url: str, body: dict[str, Any], timeout: float, api_key: str | None
) -> dict[str, Any]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with opener.open(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_json(
    opener: urllib.request.OpenerDirector, url: str, timeout: float, api_key: str | None
) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    request = urllib.request.Request(url, headers=headers, method="GET")
    with opener.open(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def build_request(case: dict[str, Any], config: RunConfig, repeat: int) -> dict[str, Any]:
    """The chat-completions request body for one generation."""
    body: dict[str, Any] = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": case_system(case, config.defense)},
            # The system prompt is the application's own text; only what
            # arrives from outside is normalised.
            {"role": "user", "content": normalize_input(case["user"], config.normalize)},
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


def _read_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a line cut short by an interruption; it will be redone
    return records


def done_keys(path: Path) -> set[tuple[str, int, str, str]]:
    """Keys of generations already recorded in ``path``."""
    return {(r["case_id"], r["repeat"], r["defense"], r["model"]) for r in _read_records(path)}


def repair_tail(path: Path) -> str:
    """Make sure ``path`` ends at a record boundary before appending to it.

    An interrupted run can leave a last line without its newline. Appending
    straight after it would glue the next record onto the fragment and lose
    both. A complete record just gets its newline; a fragment is removed.
    Returns what was done: ``"ok"``, ``"newline_added"`` or ``"fragment_removed"``.
    """
    if not path.exists() or path.stat().st_size == 0:
        return "ok"
    data = path.read_bytes()
    if data.endswith(b"\n"):
        return "ok"
    cut = data.rfind(b"\n") + 1  # 0 if the file holds a single unterminated line
    try:
        json.loads(data[cut:].decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        with path.open("r+b") as handle:
            handle.truncate(cut)
        return "fragment_removed"
    with path.open("ab") as handle:
        handle.write(b"\n")
    return "newline_added"


def _condition(config: RunConfig, model_fingerprint: dict[str, Any] | None) -> dict[str, Any]:
    """The condition this run would record (same shape as conditions.of_record)."""
    return {
        "dataset_sha256": dataset_sha256(),
        "base_url": conditions.normalize_base_url(config.base_url),
        "model_id": config.model,
        "model_fingerprint": model_fingerprint,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "seed": config.seed,
        "extra": config.extra,
        "input_normalization": config.normalize or conditions.NO_NORMALIZATION,
    }


def check_settings(
    path: Path, config: RunConfig, model_fingerprint: dict[str, Any] | None = None
) -> None:
    """Refuse to continue a result file recorded under another condition."""
    wanted = _condition(config, model_fingerprint)
    for record in _read_records(path):
        if record.get("model") != config.model_label or record.get("defense") != config.defense:
            continue
        differing = conditions.differences(conditions.of_record(record), wanted)
        if differing:
            raise SettingsMismatchError(
                f"{path} already holds generations for model {config.model_label!r} "
                f"(defense {config.defense!r}) recorded under a different condition "
                f"(recorded vs now: {conditions.describe(differing)}). Mixing them would make "
                "the results meaningless. Use a different --out file or a different --label."
            )


def _model_info(
    opener: urllib.request.OpenerDirector, config: RunConfig, api_key: str | None
) -> dict[str, Any] | None:
    """Best-effort model metadata (quantisation etc.) from LM Studio's REST API."""
    root = config.base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    url = f"{root}/api/v0/models/{urllib.parse.quote(config.model, safe='')}"
    try:
        return _get_json(opener, url, timeout=10, api_key=api_key)
    except (urllib.error.URLError, RedirectError, TimeoutError, OSError, ValueError):
        return None  # not LM Studio, or an older version: simply not available


def _write_meta(config: RunConfig, model_info: dict[str, Any] | None, planned: int) -> None:
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
        "input_normalization": config.normalize or conditions.NO_NORMALIZATION,
        "planned_generations": planned,
        "concurrency": config.concurrency,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "model_info": model_info,
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
    opener = _opener(config.base_url)
    ordered = _order(cases)
    config.out.parent.mkdir(parents=True, exist_ok=True)
    repaired = repair_tail(config.out)
    if repaired == "fragment_removed":
        log(f"removed an incomplete last line from {config.out} (left by an interrupted run)")
    model_info = _model_info(opener, config, api_key)
    model_fingerprint = conditions.fingerprint(model_info)
    check_settings(config.out, config, model_fingerprint)
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

    _write_meta(config, model_info, len(todo))
    base_url = conditions.normalize_base_url(config.base_url)
    url = config.base_url.rstrip("/") + "/chat/completions"
    sha = dataset_sha256()
    started = time.monotonic()

    def generate(repeat: int, case: dict[str, Any]) -> dict[str, Any] | None:
        """One generation, with retries. Runs in a worker thread."""
        body = build_request(case, config, repeat)
        began_at, began = _now(), time.monotonic()
        result = None
        for attempt in range(1, config.retries + 1):
            try:
                result = _extract(_post_json(opener, url, body, config.timeout, api_key))
                break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as error:
                log(f"  attempt {attempt}/{config.retries} failed for {case['id']}: {error}")
                if attempt < config.retries:
                    sleep(min(30.0, 2.0 ** attempt))
        if result is None:
            return None
        return {
            "case_id": case["id"],
            "repeat": repeat,
            "defense": config.defense,
            "model": config.model_label,
            "model_id": config.model,
            "base_url": base_url,
            "model_fingerprint": model_fingerprint,
            "input_normalization": config.normalize or conditions.NO_NORMALIZATION,
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

    # One pass at a time, so that stopping early leaves balanced data. Within a
    # pass, up to ``concurrency`` requests are in flight; only this thread
    # writes to the result file.
    passes: dict[int, list[dict[str, Any]]] = {}
    for repeat, case in todo:
        passes.setdefault(repeat, []).append(case)
    finished = 0
    with config.out.open("a", encoding="utf-8", newline="\n") as handle:
        for repeat, pass_cases in passes.items():
            executor = ThreadPoolExecutor(max_workers=config.concurrency)
            try:
                futures = [executor.submit(generate, repeat, case) for case in pass_cases]
                for future in as_completed(futures):
                    record = future.result()  # re-raises RedirectError and other fatal errors
                    finished += 1
                    if record is None:
                        counts["failed"] += 1
                    else:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        handle.flush()
                        counts["done"] += 1
                    if finished % 25 == 0 or finished == len(todo):
                        elapsed = time.monotonic() - started
                        remaining = elapsed / finished * (len(todo) - finished)
                        log(f"  {finished}/{len(todo)} done, about {remaining / 60:.0f} min left")
            finally:
                # On an error or Ctrl+C, do not start the requests still queued.
                executor.shutdown(wait=True, cancel_futures=True)

    if counts["failed"]:
        log(f"{counts['failed']} generations failed; run the same command again to retry them")
    return counts
