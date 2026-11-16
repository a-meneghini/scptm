"""
scptm/selection.py
------------------
Automatic selection of the optimal number of topics K.
"""

import math
import warnings
from typing import Dict, List, Optional, Sequence, Tuple


def _last_history_value(history: dict, key: str) -> float:
    """Last recorded value for ``key``, or NaN when it was never recorded."""
    values = history.get(key) or []
    if not values:
        return float("nan")
    try:
        return float(values[-1])
    except (TypeError, ValueError):
        return float("nan")


def _best_finite_k(results: Dict[int, dict]) -> int:
    """
    K with the highest *finite* score.

    ``max()`` over a dict is order-dependent once a NaN is involved: NaN
    comparisons are always False, so a NaN entry wins whenever it comes first
    and is ignored when it comes later.  Non-finite scores are dropped here so
    the choice is deterministic.
    """
    finite = {k: r for k, r in results.items() if math.isfinite(r["score"])}
    if not finite:
        raise RuntimeError(
            "search_k: no K produced a finite score. Metrics were probably "
            "never recorded (check metrics_every_n_epochs) or the corpus is "
            "too small for NPMI/diversity to be defined."
        )
    return max(finite, key=lambda k: finite[k]["score"])


def search_k(
    docs: List[str],
    k_grid: Sequence[int] = (5, 10, 15, 20, 25, 30),
    edge_cache_path: Optional[str] = None,
    epochs: int = 50,
    finetune: bool = True,
    verbose: bool = True,
    **kwargs,
) -> Tuple[int, object, Dict]:
    """
    Grid search for the optimal number of topics K.

    Fits SCPTM for each K in k_grid, scores by NPMI × Diversity
    (composite metric: higher = better coherence AND diversity).
    If finetune=True, performs a finer search between the two grid
    neighbours of the best K.

    The edge cache (parsed graph, embeddings) is built once on the
    first fit and reused for all subsequent K values — no re-parsing.

    Parameters
    ----------
    docs : list of str
        Input corpus.
    k_grid : sequence of int
        Topic counts to evaluate in the coarse grid.
    edge_cache_path : str | None
        Path to the edge/embedding cache. Strongly recommended to
        avoid re-parsing across K values.
    epochs : int
        Training epochs per fit.
    finetune : bool
        If True, runs a finer search between the grid neighbours
        of the best K.
    verbose : bool
        Print progress table.
    **kwargs
        Additional parameters. SCPTMConfig parameters (lang, graph_mode,
        max_chunk_chars, …) are forwarded to the SCPTM constructor.
        ``covariate`` is forwarded to ``fit_transform()`` instead.

    Returns
    -------
    best_k : int
        K with the highest NPMI × Diversity score.
    best_model : SCPTM
        Fitted model for best_k.
    results : dict
        {K: {"model": SCPTM, "npmi": float, "div": float, "score": float}}
        for every K evaluated.  K values that failed to train carry
        ``model=None``, NaN metrics and an ``error`` string; they are skipped
        when picking the best K, and a K whose score is NaN/inf is never
        selected.

    Raises
    ------
    RuntimeError
        If no K produced a finite score.
    """
    from .model import SCPTM

    # Separate fit() kwargs from SCPTMConfig constructor kwargs
    covariate = kwargs.pop("covariate", None)

    k_grid = sorted(set(int(k) for k in k_grid))
    results: Dict = {}

    def _fit_k(K: int) -> dict:
        try:
            m = SCPTM(num_topics=K, epochs=epochs,
                      metrics_every_n_epochs=epochs, **kwargs)
            m.fit_transform(docs, edge_cache_path=edge_cache_path,
                            covariate=covariate)
        except Exception as exc:                      # noqa: BLE001
            # One bad K must not abort the whole sweep.
            warnings.warn(
                f"[search_k] K={K} failed ({type(exc).__name__}: {exc}) and was "
                "skipped.",
                stacklevel=2,
            )
            return {
                "model": None,
                "npmi": float("nan"),
                "div": float("nan"),
                "score": float("nan"),
                "error": f"{type(exc).__name__}: {exc}",
            }

        history = m._history or {}
        npmi  = _last_history_value(history, "coherence_npmi")
        div   = _last_history_value(history, "topic_diversity")
        score = npmi * div
        if not math.isfinite(score):
            warnings.warn(
                f"[search_k] K={K} produced a non-finite score "
                f"(npmi={npmi}, diversity={div}); it will not be selected.",
                stacklevel=2,
            )
        if verbose:
            print(f"  K={K:3d}  NPMI={npmi:+.3f}  Div={div:.3f}  Score={score:+.4f}")
        return {"model": m, "npmi": npmi, "div": div, "score": score}

    if verbose:
        print(f"\n[search_k] Coarse grid: {k_grid}")
    for K in k_grid:
        results[K] = _fit_k(K)

    best_K = _best_finite_k(results)

    if finetune and len(k_grid) >= 2:
        idx = k_grid.index(best_K)
        lo  = k_grid[idx - 1] if idx > 0 else max(2, best_K - 3)
        hi  = k_grid[idx + 1] if idx < len(k_grid) - 1 else best_K + 3
        step = max(1, (hi - lo) // 4)
        fine_grid = [k for k in range(lo + step, hi, step) if k not in results]
        if fine_grid:
            if verbose:
                print(f"\n[search_k] Fine-tuning around K={best_K}: {fine_grid}")
            for K in fine_grid:
                results[K] = _fit_k(K)
            best_K = _best_finite_k(results)

    if verbose:
        print(
            f"\n[search_k] Best K = {best_K}  "
            f"(NPMI={results[best_K]['npmi']:+.3f}, "
            f"Div={results[best_K]['div']:.3f}, "
            f"Score={results[best_K]['score']:+.4f})"
        )

    return best_K, results[best_K]["model"], results
