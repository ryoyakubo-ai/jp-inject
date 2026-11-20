import math

from jp_inject.stats import bootstrap_mean, cluster_bootstrap_rate, percentile_interval, wilson


def test_wilson_known_value():
    low, high = wilson(5, 10)
    assert math.isclose(low, 0.2366, abs_tol=1e-3)
    assert math.isclose(high, 0.7634, abs_tol=1e-3)
    assert wilson(0, 10)[0] == 0.0
    assert math.isclose(wilson(10, 10)[1], 1.0)
    assert all(math.isnan(v) for v in wilson(0, 0))


def test_percentile_interval():
    low, high = percentile_interval(list(range(101)), 0.9)
    assert math.isclose(low, 5.0) and math.isclose(high, 95.0)


def test_cluster_bootstrap_is_deterministic_and_brackets_the_rate():
    clusters = {f"b{i}": (i % 3, 5) for i in range(20)}
    first = cluster_bootstrap_rate(clusters)
    assert first == cluster_bootstrap_rate(clusters)
    rate, low, high = first
    assert low <= rate <= high


def test_clustered_data_gives_wider_interval_than_naive():
    # 10 bases that always succeed and 10 that never do: 100 generations look
    # like a lot of evidence, but there are only 20 independent units.
    clusters = {f"b{i}": (5 if i < 10 else 0, 5) for i in range(20)}
    _, low, high = cluster_bootstrap_rate(clusters)
    naive_low, naive_high = wilson(50, 100)
    assert (high - low) > (naive_high - naive_low)


def test_bootstrap_mean():
    mean, low, high = bootstrap_mean([0.2] * 10)
    assert math.isclose(mean, 0.2) and math.isclose(low, 0.2) and math.isclose(high, 0.2)
    mean, low, high = bootstrap_mean([0.0, 0.4, 0.2, 0.6, -0.2])
    assert low < mean < high
