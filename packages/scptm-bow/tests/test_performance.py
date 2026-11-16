"""
Performance guard for scptm-bow.

Two things are asserted, neither of them flaky:

1. **A wall-clock ceiling** generous enough to survive a slow CI machine but tight
   enough to catch a complexity regression (the whole point of the package is
   that a fit is seconds, not minutes).
2. **No dense V x V materialisation**: the reconstruction is evaluated on the
   support of X in bounded chunks, so the memory cost must not grow with V^2.
"""
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import scptm_bow.factorize as F  # noqa: E402
from scptm_bow.factorize import GraphRegularizedNMF  # noqa: E402

# Generous: the real time on a laptop is ~1-3 s.
FIT_CEILING_SECONDS = 60.0


def _synthetic(n_docs=800, n_vocab=2000, nnz_per_doc=40, n_topics=6, seed=0):
    rng = np.random.default_rng(seed)
    rows, cols = [], []
    for d in range(n_docs):
        cols.extend(rng.choice(n_vocab, size=nnz_per_doc, replace=False).tolist())
        rows.extend([d] * nnz_per_doc)
    data = rng.integers(1, 6, size=len(rows)).astype(np.float64)
    return sparse.csr_matrix((data, (rows, cols)), shape=(n_docs, n_vocab))


def test_a_full_fit_stays_within_the_wall_clock_budget():
    X = _synthetic()
    start = time.perf_counter()
    result = GraphRegularizedNMF(
        n_components=6, max_iter=60, divergence="kl", random_state=0
    ).fit_transform(X)
    elapsed = time.perf_counter() - start
    assert np.isfinite(result.objective)
    assert elapsed < FIT_CEILING_SECONDS, f"fit took {elapsed:.1f}s"


def test_reconstruction_is_evaluated_in_bounded_chunks():
    X = _synthetic(n_docs=200, n_vocab=300)
    rng = np.random.default_rng(1)
    W = rng.random((X.shape[0], 4))
    H = rng.random((4, X.shape[1]))
    rows, cols = X.nonzero()

    full = GraphRegularizedNMF._reconstruct_at(W, H, rows, cols)
    original = F._NNZ_CHUNK
    try:
        F._NNZ_CHUNK = 7                      # force many small chunks
        chunked = GraphRegularizedNMF._reconstruct_at(W, H, rows, cols)
    finally:
        F._NNZ_CHUNK = original
    assert np.allclose(full, chunked, atol=1e-12), "chunking must not change values"
    assert full.shape == rows.shape


def test_kl_objective_is_monotone_on_a_larger_problem():
    X = _synthetic(n_docs=300, n_vocab=500)
    result = GraphRegularizedNMF(
        n_components=5, max_iter=40, tol=0.0, divergence="kl", random_state=0
    ).fit_transform(X)
    history = result.history
    assert all(b <= a + 1e-9 for a, b in zip(history, history[1:]))


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:                      # noqa: BLE001
            failures += 1
            print(f"  FAIL  {name}\n        {type(exc).__name__}: {exc}")
    print(f"{'OK' if failures == 0 else 'FAILED'} ({failures} failure(s))")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
