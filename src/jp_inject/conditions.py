"""What makes two generations comparable.

A success rate only means something if every generation behind it was
produced under the same condition: the same test cases, the same model file
on the same server, the same sampling settings. This module defines that
condition once, so that the runner (when resuming a file) and the analysis
(when reading files) apply exactly the same rule.
"""

from __future__ import annotations

import urllib.parse
from typing import Any

__all__ = [
    "CONDITION_FIELDS",
    "FINGERPRINT_KEYS",
    "describe",
    "differences",
    "fingerprint",
    "normalize_base_url",
    "of_record",
]

# Compared in this order when reporting a mismatch.
CONDITION_FIELDS = (
    "dataset_sha256",
    "base_url",
    "model_id",
    "model_fingerprint",
    "temperature",
    "max_tokens",
    "seed",
    "extra",
)

# Stable parts of the model metadata a server reports. Volatile fields (load
# state, loaded context length) are left out on purpose.
FINGERPRINT_KEYS = ("id", "publisher", "arch", "quantization", "compatibility_type", "type")


def normalize_base_url(url: str) -> str:
    """Scheme, host, port and path only: no credentials, query or trailing slash."""
    parts = urllib.parse.urlsplit(url)
    host = parts.hostname or ""
    if ":" in host:  # IPv6 literal
        host = f"[{host}]"
    netloc = f"{host}:{parts.port}" if parts.port else host
    return urllib.parse.urlunsplit((parts.scheme.lower(), netloc.lower(), parts.path.rstrip("/"), "", ""))


def fingerprint(model_info: dict[str, Any] | None) -> dict[str, Any] | None:
    """The identifying subset of a server's model metadata, or None if unknown."""
    if not isinstance(model_info, dict):
        return None
    found = {key: model_info[key] for key in FINGERPRINT_KEYS if model_info.get(key) is not None}
    return found or None


def of_record(record: dict[str, Any]) -> dict[str, Any]:
    """The condition a recorded generation was produced under.

    Fields a record does not carry come back as None, meaning "unknown".
    """
    request = record.get("request") or {}
    seed = request.get("seed")
    return {
        "dataset_sha256": record.get("dataset_sha256"),
        "base_url": record.get("base_url"),
        "model_id": record.get("model_id"),
        "model_fingerprint": record.get("model_fingerprint"),
        "temperature": request.get("temperature"),
        "max_tokens": request.get("max_tokens"),
        # Each pass uses seed + pass number; the run's base seed is what must match.
        "seed": None if seed is None else seed - record.get("repeat", 0),
        "extra": request.get("extra") or {},
    }


def differences(a: dict[str, Any], b: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    """Fields on which two conditions disagree.

    A field that is unknown (None) on either side is not counted: a server
    that reports no model metadata cannot be told apart by it.
    """
    return {
        field: (a.get(field), b.get(field))
        for field in CONDITION_FIELDS
        if a.get(field) is not None and b.get(field) is not None and a.get(field) != b.get(field)
    }


def describe(diff: dict[str, tuple[Any, Any]]) -> str:
    return "; ".join(f"{field}: {first!r} vs {second!r}" for field, (first, second) in diff.items())
