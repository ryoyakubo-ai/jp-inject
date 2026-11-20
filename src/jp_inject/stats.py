"""Small, dependency-free statistics helpers.

Repeated generations of the same attack are not independent observations: an
attack that works once tends to work again. The unit that was actually sampled
is the *base attack* (a target x strategy pair), so confidence intervals are
computed with a cluster bootstrap that resamples base attacks, not individual
generations.
"""

from __future__ import annotations

import math
import random
from typing import Mapping, Sequence

__all__ = ["bootstrap_mean", "cluster_bootstrap_rate", "percentile_interval", "wilson"]

DEFAULT_RESAMPLES = 2000
DEFAULT_SEED = 20261119


def wilson(successes: int, trials: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% by default)."""
    if trials <= 0:
        return (float("nan"), float("nan"))
    p = successes / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def percentile_interval(values: Sequence[float], level: float = 0.95) -> tuple[float, float]:
    """Percentile interval of a list of bootstrap estimates."""
    if not values:
        return (float("nan"), float("nan"))
    ordered = sorted(values)
    alpha = (1 - level) / 2

    def pick(q: float) -> float:
        position = q * (len(ordered) - 1)
        low, high = math.floor(position), math.ceil(position)
        if low == high:
            return ordered[low]
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

    return (pick(alpha), pick(1 - alpha))


def cluster_bootstrap_rate(
    clusters: Mapping[str, tuple[int, int]],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float, float]:
    """Success rate with a cluster-bootstrap 95% interval.

    ``clusters`` maps a cluster id to ``(successes, trials)``. Returns
    ``(rate, low, high)`` where the rate pools all trials.
    """
    # Sorted so that the result does not depend on the order rows were read in.
    items = [clusters[key] for key in sorted(clusters) if clusters[key][1] > 0]
    trials = sum(n for _, n in items)
    if not trials:
        return (float("nan"), float("nan"), float("nan"))
    rate = sum(k for k, _ in items) / trials
    if len(items) == 1:
        low, high = wilson(items[0][0], items[0][1])
        return (rate, low, high)
    rng = random.Random(seed)
    estimates = []
    for _ in range(resamples):
        sample = rng.choices(items, k=len(items))
        estimates.append(sum(k for k, _ in sample) / sum(n for _, n in sample))
    low, high = percentile_interval(estimates)
    return (rate, low, high)


def bootstrap_mean(
    values: Sequence[float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float, float]:
    """Mean of per-cluster values with a bootstrap 95% interval.

    Used for paired differences: ``values`` holds one difference per base
    attack (rate under one condition minus rate under another).
    """
    if not values:
        return (float("nan"), float("nan"), float("nan"))
    mean = sum(values) / len(values)
    if len(values) == 1:
        return (mean, float("nan"), float("nan"))
    rng = random.Random(seed)
    estimates = []
    for _ in range(resamples):
        sample = rng.choices(values, k=len(values))
        estimates.append(sum(sample) / len(sample))
    low, high = percentile_interval(estimates)
    return (mean, low, high)
