"""
scptm/selection.py
------------------
Automatic selection of the optimal number of topics K.
"""

from typing import Dict, List, Optional, Sequence, Tuple


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
        for every K evaluated.
    """
    from .model import SCPTM

    # Separate fit() kwargs from SCPTMConfig constructor kwargs
    covariate = kwargs.pop("covariate", None)

    k_grid = sorted(set(int(k) for k in k_grid))
    results: Dict = {}

    def _fit_k(K: int) -> dict:
        m = SCPTM(num_topics=K, epochs=epochs,
                  metrics_every_n_epochs=epochs, **kwargs)
        m.fit_transform(docs, edge_cache_path=edge_cache_path,
                        covariate=covariate)
        history = m._history or {}
        npmi  = history.get("coherence_npmi",  [0.0])[-1]
        div   = history.get("topic_diversity", [0.0])[-1]
        score = npmi * div
        if verbose:
            print(f"  K={K:3d}  NPMI={npmi:+.3f}  Div={div:.3f}  Score={score:+.4f}")
        return {"model": m, "npmi": npmi, "div": div, "score": score}

    if verbose:
        print(f"\n[search_k] Coarse grid: {k_grid}")
    for K in k_grid:
        results[K] = _fit_k(K)

    best_K = max(results, key=lambda k: results[k]["score"])

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
            best_K = max(results, key=lambda k: results[k]["score"])

    if verbose:
        print(
            f"\n[search_k] Best K = {best_K}  "
            f"(NPMI={results[best_K]['npmi']:+.3f}, "
            f"Div={results[best_K]['div']:.3f}, "
            f"Score={results[best_K]['score']:+.4f})"
        )

    return best_K, results[best_K]["model"], results
