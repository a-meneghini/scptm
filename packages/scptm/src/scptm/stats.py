"""
scptm/stats.py
--------------
Uncertainty and paired comparisons for topic-model benchmarks.

Deliberate duplicate of ``scptm_bow.stats`` (the packages stay independently
installable); verified by the shared conformance fixture.

``bootstrap_ci`` / ``paired_bootstrap_ci`` / ``wilcoxon_paired`` / ``cliffs_delta``
/ ``holm`` / ``aggregate``.

With three seeds a two-sided Wilcoxon cannot reach p < 0.05 (minimum 2/8 = 0.25):
``wilcoxon_paired`` warns rather than letting a reader over-interpret it.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as _scipy_stats

__all__ = [
    "Estimate",
    "TestResult",
    "bootstrap_ci",
    "paired_bootstrap_ci",
    "wilcoxon_paired",
    "cliffs_delta",
    "cohens_dz",
    "holm",
    "aggregate",
]

# Below this many pairs a two-sided Wilcoxon cannot reach alpha = 0.05.
MIN_PAIRS_FOR_SIGNIFICANCE = 6


@dataclass(frozen=True)
class Estimate:
    """A point estimate with a confidence interval."""

    estimate: float
    low: float
    high: float
    level: float
    n: int
    method: str

    @property
    def excludes_zero(self) -> bool:
        """True when the interval is entirely above or below zero."""
        return (self.low > 0.0 and self.high > 0.0) or (self.low < 0.0 and self.high < 0.0)

    def __str__(self) -> str:
        return (
            f"{self.estimate:.4f} [{self.low:.4f}, {self.high:.4f}] "
            f"({int(self.level * 100)}% CI, n={self.n}, {self.method})"
        )


@dataclass(frozen=True)
class TestResult:
    """Outcome of a paired comparison."""

    statistic: float
    p_value: float
    method: str
    n: int
    effect_size: float
    effect_name: str
    warning: Optional[str] = None

    def __str__(self) -> str:
        base = (
            f"{self.method}: stat={self.statistic:.4f} p={self.p_value:.4f} "
            f"n={self.n} {self.effect_name}={self.effect_size:.3f}"
        )
        return f"{base}  ! {self.warning}" if self.warning else base


def _as_paired(a: Sequence[float], b: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError(
            f"paired comparison needs equal-length vectors, got {x.shape} and {y.shape}"
        )
    if x.size == 0:
        raise ValueError("paired comparison needs at least one pair")
    return x, y


def bootstrap_ci(
    values: Sequence[float],
    n_resamples: int = 10_000,
    level: float = 0.95,
    statistic=np.mean,
    seed: int = 0,
) -> Estimate:
    """Percentile bootstrap CI for any statistic of ``values``."""
    x = np.asarray(values, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return Estimate(float("nan"), float("nan"), float("nan"), level, 0, "bootstrap")
    if x.size == 1:
        v = float(statistic(x))
        return Estimate(v, float("nan"), float("nan"), level, 1, "bootstrap")
    rng = np.random.default_rng(seed)
    draws = rng.choice(x, size=(n_resamples, x.size), replace=True)
    stats = np.apply_along_axis(statistic, 1, draws)
    alpha = (1.0 - level) / 2.0
    low, high = np.quantile(stats, [alpha, 1.0 - alpha])
    return Estimate(float(statistic(x)), float(low), float(high), level, int(x.size),
                    "bootstrap")


def paired_bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    n_resamples: int = 10_000,
    level: float = 0.95,
    seed: int = 0,
) -> Estimate:
    """
    Bootstrap CI for the mean paired difference ``a - b``.

    Pairing is what makes this valid: the same seed (or corpus split) must sit in
    the same position in both vectors.  A CI that excludes zero is the defensible
    claim "a beats b" for a small number of seeds.
    """
    x, y = _as_paired(a, b)
    diff = x - y
    est = bootstrap_ci(diff, n_resamples=n_resamples, level=level, seed=seed)
    return Estimate(est.estimate, est.low, est.high, est.level, est.n, "paired bootstrap")


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Cliff's delta: P(a > b) - P(a < b), in [-1, 1].

    Non-parametric, so it survives the small samples that topic-model benchmarks
    actually have.  |d| < 0.147 negligible, < 0.33 small, < 0.474 medium
    (Romano et al. 2006).
    """
    x, y = _as_paired(a, b)
    greater = sum(1 for xi in x for yi in y if xi > yi)
    less = sum(1 for xi in x for yi in y if xi < yi)
    total = x.size * y.size
    return float((greater - less) / total) if total else 0.0


def cohens_dz(a: Sequence[float], b: Sequence[float]) -> float:
    """Standardised mean difference for paired data (d_z)."""
    x, y = _as_paired(a, b)
    diff = x - y
    sd = float(np.std(diff, ddof=1)) if diff.size > 1 else 0.0
    return float(np.mean(diff) / sd) if sd > 0 else 0.0


def wilcoxon_paired(
    a: Sequence[float],
    b: Sequence[float],
    alpha: float = 0.05,
) -> TestResult:
    """
    Two-sided Wilcoxon signed-rank test on ``a - b``, with Cliff's delta.

    Warns when the sample is too small for the test to be able to reject at
    ``alpha`` — a three-seed design cannot, and silently reporting p > 0.05 as
    "no effect" would be a misreading of the evidence.
    """
    x, y = _as_paired(a, b)
    diff = x - y
    if np.allclose(diff, 0.0):
        # The warning is about the *design* (how many seeds), not the data, so it
        # applies here too: a reader must not read "p = 1" as evidence of no effect.
        min_p = 2.0 / (2 ** x.size)
        return TestResult(
            0.0, 1.0, "wilcoxon", int(x.size), 0.0, "cliffs_delta",
            warning=(
                f"n={x.size}: the smallest attainable two-sided p is {min_p:.3f}, "
                f"above alpha={alpha}. Use the paired bootstrap CI instead."
                if min_p > alpha else None
            ),
        )

    result = _scipy_stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
    warning = None
    min_p = 2.0 / (2 ** x.size)
    if min_p > alpha:
        warning = (
            f"n={x.size}: the smallest attainable two-sided p is {min_p:.3f}, "
            f"above alpha={alpha}. Use the paired bootstrap CI instead."
        )
    return TestResult(
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        method="wilcoxon",
        n=int(x.size),
        effect_size=cliffs_delta(x, y),
        effect_name="cliffs_delta",
        warning=warning,
    )


def holm(p_values: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni step-down adjustment, monotone and capped at 1."""
    p = np.asarray(p_values, dtype=np.float64).ravel()
    if p.size == 0:
        return p
    order = np.argsort(p)
    m = p.size
    adjusted = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, idx in enumerate(order):
        value = (m - rank) * p[idx]
        running = max(running, value)
        adjusted[idx] = min(running, 1.0)
    return adjusted


def aggregate(
    records: Sequence[Dict[str, object]],
    metric: str,
    group: Sequence[str] = ("corpus", "model"),
    n_resamples: int = 2_000,
    level: float = 0.95,
    seed: int = 0,
) -> List[Dict[str, object]]:
    """
    Mean / sd / bootstrap CI per group from per-run records.

    ``records`` is a list of dicts with the grouping keys plus one numeric field
    per metric — exactly what a benchmark runner emits, one row per fit.
    """
    buckets: Dict[Tuple, List[float]] = {}
    for record in records:
        if metric not in record:
            continue
        value = record[metric]
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(value):
            continue
        key = tuple(record.get(g) for g in group)
        buckets.setdefault(key, []).append(value)

    out: List[Dict[str, object]] = []
    for key, values in sorted(buckets.items(), key=lambda kv: tuple(map(str, kv[0]))):
        est = bootstrap_ci(values, n_resamples=n_resamples, level=level, seed=seed)
        row: Dict[str, object] = dict(zip(group, key))
        row.update(
            {
                "metric": metric,
                "n": len(values),
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "ci_low": est.low,
                "ci_high": est.high,
            }
        )
        out.append(row)
    return out
