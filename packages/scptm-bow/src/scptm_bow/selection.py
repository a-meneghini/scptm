"""
scptm_bow/selection.py
----------------------
Automatic selection of the number of topics K.

Same composite criterion as the neural package (NPMI × diversity, higher is
better), but the sweep is cheap here: the parse artefacts are cached once and
reused for every K, so each fit is only the NMF factorisation.
"""

import math
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

from .model import BowSCPTM


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
            "search_k: no K produced a finite score. The corpus is probably too "
            "small for NPMI/diversity to be defined (try min_df=1 or more "
            "documents)."
        )
    return max(finite, key=lambda k: finite[k]["score"])


def _score_model(model) -> Dict[str, float]:
    """NPMI, diversity and their product for a fitted model."""
    metrics = model.evaluate()
    npmi = float(metrics.get("npmi", float("nan")))
    div = float(metrics.get("diversity", float("nan")))
    return {"npmi": npmi, "div": div, "score": npmi * div}


def search_k(
    docs: Sequence[str],
    k_grid: Sequence[int] = (5, 10, 15, 20, 25, 30),
    cache_path: Optional[str] = None,
    finetune: bool = True,
    verbose: bool = True,
    **kwargs,
) -> Tuple[int, BowSCPTM, Dict]:
    """
    Grid search for the optimal number of topics K.

    Fits :class:`BowSCPTM` for each K in ``k_grid``, scores by NPMI × diversity
    (higher = coherent *and* diverse topics), then optionally refines between the
    two grid neighbours of the best K.

    The parse cache (vocabulary, BoW, dependency graph, triples) is built once
    and reused for every K: it is signature-guarded on the corpus and the
    parsing/vocabulary settings, none of which depend on K.  Pass ``cache_path``
    to reuse it across runs too.

    Parameters
    ----------
    docs : sequence of str
        Input corpus.
    k_grid : sequence of int
        Topic counts to evaluate in the coarse grid.
    cache_path : str | None
        Path to the parse cache.  Strongly recommended: parsing dominates the
        cost and is identical for every K.
    finetune : bool
        If True, runs a finer search between the grid neighbours of the best K.
    verbose : bool
        Print the progress table and the final choice.
    **kwargs
        ``BowConfig`` parameters (``lang``, ``parser_backend``, ``min_df``,
        ``lambda_syntax``, ``vector_backend``, …).

    Returns
    -------
    best_k : int
        K with the highest NPMI × diversity score.
    best_model : BowSCPTM
        Fitted model for ``best_k``.
    results : dict
        ``{K: {"model": BowSCPTM, "npmi": float, "div": float, "score": float}}``
        for every K evaluated.  A K that failed to fit carries ``model=None``,
        NaN metrics and an ``error`` string; it is never selected, and a
        non-finite score is never selected either.

    Raises
    ------
    RuntimeError
        If no K produced a finite score.

    Examples
    --------
    >>> best_k, model, results = search_k(docs, k_grid=(5, 10, 20),
    ...                                   cache_path="corpus.pkl", lang="ita")
    >>> model.get_topic_info(top_k=8)
    """
    k_grid = sorted({int(k) for k in k_grid})
    if not k_grid:
        raise ValueError("k_grid is empty.")
    if any(k < 2 for k in k_grid):
        raise ValueError(f"every K in k_grid must be >= 2, got {k_grid}")

    config_kwargs = dict(kwargs)
    config_kwargs.pop("num_topics", None)   # K comes from the grid, not the caller
    results: Dict[int, dict] = {}

    def _fit_k(K: int) -> dict:
        try:
            model = BowSCPTM(num_topics=K, **config_kwargs)
            model.fit(docs, cache_path=cache_path, verbose=False)
        except Exception as exc:            # noqa: BLE001 - one bad K must not abort
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

        out = _score_model(model)
        if not math.isfinite(out["score"]):
            warnings.warn(
                f"[search_k] K={K} produced a non-finite score "
                f"(npmi={out['npmi']}, diversity={out['div']}); it will not be "
                "selected.",
                stacklevel=2,
            )
        if verbose:
            print(
                f"  K={K:3d}  NPMI={out['npmi']:+.3f}  Div={out['div']:.3f}  "
                f"Score={out['score']:+.4f}"
            )
        return {"model": model, **out}

    if verbose:
        print(f"\n[search_k] Coarse grid: {k_grid}")
    for K in k_grid:
        results[K] = _fit_k(K)

    best_k = _best_finite_k(results)

    if finetune and len(k_grid) >= 2:
        idx = k_grid.index(best_k)
        lo = k_grid[idx - 1] if idx > 0 else max(2, best_k - 3)
        hi = k_grid[idx + 1] if idx < len(k_grid) - 1 else best_k + 3
        step = max(1, (hi - lo) // 4)
        fine_grid = [k for k in range(lo + step, hi, step) if k not in results]
        if fine_grid:
            if verbose:
                print(f"\n[search_k] Fine-tuning around K={best_k}: {fine_grid}")
            for K in fine_grid:
                results[K] = _fit_k(K)
            best_k = _best_finite_k(results)

    if verbose:
        print(
            f"\n[search_k] Best K = {best_k}  "
            f"(NPMI={results[best_k]['npmi']:+.3f}, "
            f"Div={results[best_k]['div']:.3f}, "
            f"Score={results[best_k]['score']:+.4f})"
        )

    return best_k, results[best_k]["model"], results
