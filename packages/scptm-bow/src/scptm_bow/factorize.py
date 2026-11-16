"""
scptm_bow/factorize.py
----------------------
Graph-regularised non-negative matrix factorisation.

Objective
---------
    min_{W,H >= 0}  D(X || W H)
                    + lambda_s * tr(H L_s H^T)
                    + lambda_e * tr(H L_e H^T)

``D`` is the generalised Kullback-Leibler divergence (**default**) or the squared
Frobenius norm, selected by ``divergence``.

Why KL is the default
---------------------
``X`` holds counts, and the Frobenius objective corresponds to Gaussian noise,
which is misspecified for count data.  The KL objective corresponds to a
Poisson/multinomial likelihood, and it is the same optimisation problem that
LDA's mean-field variational inference solves.  So KL is both the statistically
appropriate choice and the LDA-equivalent one — while keeping the deterministic
multiplicative updates and, crucially, the graph penalty that would be
non-conjugate under an explicit Dirichlet prior.

Both divergences use the same positive/negative gradient split, which is why the
Laplacian terms slot in identically:

    H <- H * (num_H + lambda * H A) / (den_H + lambda * H D)

Everything stays sparse: ``X / (W H)`` is non-zero only on the support of ``X``,
so the ratio matrix keeps ``X``'s sparsity pattern and no ``n_docs x n_terms``
dense array is ever materialised.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds

from .graphs import Graph, empty_graph

_EPS = 1e-10

DIVERGENCES = ("kl", "frobenius")

# Rows of X processed at once when evaluating W H on the non-zero pattern;
# keeps the temporary at ~chunk x K floats instead of nnz x K.
_NNZ_CHUNK = 200_000


@dataclass
class NMFResult:
    W: np.ndarray                     # (n_docs, K) document-topic
    H: np.ndarray                     # (K, V) topic-term
    n_iter: int
    objective: float
    divergence: str = "kl"
    history: List[float] = field(default_factory=list)

    @property
    def n_topics(self) -> int:
        return self.H.shape[0]


class GraphRegularizedNMF:
    """Multiplicative-update NMF (KL or Frobenius) with graph Laplacian penalties."""

    def __init__(
        self,
        n_components: int = 10,
        lambda_syntax: float = 0.0,
        lambda_embed: float = 0.0,
        max_iter: int = 200,
        tol: float = 1e-4,
        random_state: int = 42,
        divergence: str = "kl",
    ):
        if n_components < 1:
            raise ValueError("n_components must be >= 1")
        if divergence not in DIVERGENCES:
            raise ValueError(
                f"divergence must be one of {list(DIVERGENCES)}, got '{divergence}'"
            )
        self.n_components = n_components
        self.lambda_syntax = float(lambda_syntax)
        self.lambda_embed = float(lambda_embed)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = int(random_state)
        self.divergence = divergence

    # -- initialisation ----------------------------------------------------

    def _init_factors(self, X: sparse.csr_matrix):
        """Deterministic SVD-based init (NNDSVD flavour)."""
        n, v = X.shape
        k = min(self.n_components, min(n, v) - 1)
        if k >= 1:
            try:
                u, s, vt = svds(X.astype(np.float64), k=k)
                order = np.argsort(-s)
                u, s, vt = u[:, order], s[order], vt[order]
                W = np.abs(u) * np.sqrt(s)
                H = np.abs(vt) * np.sqrt(s)[:, None]
                if self.n_components > k:      # pad if rank was capped
                    pad = self.n_components - k
                    W = np.hstack([W, np.zeros((n, pad))])
                    H = np.vstack([H, np.zeros((pad, v))])
                rng = np.random.default_rng(self.random_state)
                W = W + rng.random(W.shape) * 1e-4
                H = H + rng.random(H.shape) * 1e-4
                return W, H
            except Exception:                  # pragma: no cover - tiny inputs
                pass
        rng = np.random.default_rng(self.random_state)
        return (
            rng.random((n, self.n_components)),
            rng.random((self.n_components, v)),
        )

    # -- reconstruction on the support of X --------------------------------

    @staticmethod
    def _reconstruct_at(
        W: np.ndarray, H: np.ndarray, rows: np.ndarray, cols: np.ndarray
    ) -> np.ndarray:
        """Evaluate (W H)[rows, cols] without materialising the full product."""
        out = np.empty(rows.shape[0], dtype=np.float64)
        for start in range(0, rows.shape[0], _NNZ_CHUNK):
            r = rows[start:start + _NNZ_CHUNK]
            c = cols[start:start + _NNZ_CHUNK]
            out[start:start + _NNZ_CHUNK] = np.einsum("ij,ij->i", W[r], H[:, c].T)
        return out

    # -- objective ---------------------------------------------------------

    @staticmethod
    def _penalty(H: np.ndarray, graphs, lambdas) -> float:
        penalty = 0.0
        for graph, lam in zip(graphs, lambdas):
            if lam <= 0.0 or graph is None or graph.adj.nnz == 0:
                continue
            a_term = float(np.sum(H * (H @ graph.adj)))
            d_term = float(np.sum((H ** 2) * graph.degree[None, :]))
            penalty += lam * (d_term - a_term)
        return penalty

    def _objective(self, X, W, H, graphs, lambdas, rows, cols, x_data,
                   x_squared_sum=None) -> float:
        if self.divergence == "kl":
            wh = np.maximum(self._reconstruct_at(W, H, rows, cols), _EPS)
            # sum X log(X/WH) - sum X + sum WH  (X = 0 entries contribute 0)
            value = float(
                np.sum(x_data * (np.log(x_data) - np.log(wh)))
                - x_data.sum()
                + float(W.sum(axis=0) @ H.sum(axis=1))
            )
        else:
            xh_t = np.asarray(X @ H.T)
            if x_squared_sum is None:          # only on a standalone call
                x_squared_sum = float(X.multiply(X).sum())
            value = (
                x_squared_sum
                - 2.0 * float(np.sum(W * xh_t))
                + float(np.sum((W.T @ W) * (H @ H.T)))
            )
        return value + self._penalty(H, graphs, lambdas)

    # -- fitting -----------------------------------------------------------

    def fit_transform(
        self,
        X,
        syntax_graph: Optional[Graph] = None,
        embed_graph: Optional[Graph] = None,
    ) -> NMFResult:
        X = sparse.csr_matrix(X, dtype=np.float64)
        X.eliminate_zeros()
        X.sort_indices()
        n, v = X.shape

        graphs = [syntax_graph or empty_graph(v), embed_graph or empty_graph(v)]
        lambdas = [self.lambda_syntax, self.lambda_embed]

        rows, cols = X.nonzero()
        x_data = np.asarray(X[rows, cols]).ravel().astype(np.float64)
        # ||X||_F^2 is constant across iterations; computing it inside the loop
        # would repeat a full sparse pass every time.
        x_squared_sum = float(X.multiply(X).sum())

        W, H = self._init_factors(X)
        if self.divergence == "kl":
            # KL is scale-sensitive: start the factors on X's total mass.
            mass = float(W.sum(axis=0) @ H.sum(axis=1))
            if mass > 0:
                W *= float(X.sum()) / mass

        history: List[float] = []
        objective = np.inf
        iteration = 0

        for iteration in range(1, self.max_iter + 1):
            if self.divergence == "kl":
                # H <- H * (W^T R + lam H A) / (W^T 1 + lam H D),
                # with R = X / (W H) evaluated at the *current* factors.
                ratio = x_data / np.maximum(
                    self._reconstruct_at(W, H, rows, cols), _EPS
                )
                R = sparse.csr_matrix((ratio, (rows, cols)), shape=X.shape)

                numerator = np.asarray(R.T @ W).T          # (K, V)
                denominator = np.broadcast_to(
                    W.sum(axis=0)[:, None], (self.n_components, v)
                ).copy()
                for graph, lam in zip(graphs, lambdas):
                    if lam > 0.0 and graph.adj.nnz:
                        numerator = numerator + lam * (H @ graph.adj)
                        denominator = denominator + lam * (H * graph.degree[None, :])
                H = H * numerator / (denominator + _EPS)

                # W <- W * (R' H^T) / (1 H^T) with H now updated, so the ratio
                # must be recomputed at (W_old, H_new).  Reusing the ratio from
                # before the H update breaks the monotonicity guarantee and makes
                # the factors oscillate in scale.
                ratio_w = x_data / np.maximum(
                    self._reconstruct_at(W, H, rows, cols), _EPS
                )
                R_w = sparse.csr_matrix((ratio_w, (rows, cols)), shape=X.shape)

                w_num = np.asarray(R_w @ H.T)              # (n, K)
                w_den = np.broadcast_to(
                    H.sum(axis=1)[None, :], (n, self.n_components)
                ).copy()
                W = W * w_num / (w_den + _EPS)
            else:
                numerator = np.asarray(X.T @ W).T          # (K, V)
                denominator = (W.T @ W) @ H
                for graph, lam in zip(graphs, lambdas):
                    if lam > 0.0 and graph.adj.nnz:
                        numerator = numerator + lam * (H @ graph.adj)
                        denominator = denominator + lam * (H * graph.degree[None, :])
                H = H * numerator / (denominator + _EPS)

                w_num = np.asarray(X @ H.T)                # (n, K)
                w_den = W @ (H @ H.T)
                W = W * w_num / (w_den + _EPS)

            objective = self._objective(
                X, W, H, graphs, lambdas, rows, cols, x_data,
                x_squared_sum=x_squared_sum,
            )
            history.append(objective)
            if iteration > 1 and history[-2] != 0:
                rel = abs(history[-2] - objective) / abs(history[-2])
                if rel < self.tol:
                    break

        # Row-normalise H so it reads as a topic-term distribution; push the
        # scale into W so that W H is (approximately) unchanged.
        row_sums = H.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        H = H / row_sums
        W = W * row_sums.T

        return NMFResult(
            W=W, H=H, n_iter=iteration, objective=objective,
            divergence=self.divergence, history=history,
        )

    # -- out-of-sample -----------------------------------------------------

    def transform(self, X_new, H: np.ndarray, n_iter: int = 100) -> np.ndarray:
        """
        Fold new documents in: non-negative least squares for W with H fixed,
        solved with W-only multiplicative updates (no scipy optimiser needed).
        """
        X_new = sparse.csr_matrix(X_new, dtype=np.float64)
        X_new.eliminate_zeros()
        n = X_new.shape[0]
        rng = np.random.default_rng(self.random_state)
        W = rng.random((n, H.shape[0])) + 1e-3

        if self.divergence == "kl":
            w_den = np.broadcast_to(H.sum(axis=1)[None, :], W.shape).copy()
            X_dense = np.asarray(X_new.todense())
            for _ in range(n_iter):
                wh = np.maximum(W @ H, _EPS)
                ratio = np.divide(
                    X_dense, wh, out=np.zeros_like(X_dense), where=wh > 0
                )
                W = W * (ratio @ H.T) / (w_den + _EPS)
        else:
            hh_t = H @ H.T
            for _ in range(n_iter):
                W = W * np.asarray(X_new @ H.T) / (W @ hh_t + _EPS)
        return W
