"""
scptm_bow/graphs.py
--------------------
The two word-word graphs used to regularise NMF.

* **syntactic** — aggregated dependency arcs, symmetrised and top-k pruned.
* **semantic**  — cosine k-NN graph over fastText (or PPMI-SVD) word vectors.

Both are returned as an adjacency matrix plus its degree vector, which is all
the multiplicative updates need (the Laplacian L = D - A never has to be
materialised).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import sparse


@dataclass
class Graph:
    """Symmetric non-negative adjacency plus its degree vector."""

    adj: sparse.csr_matrix
    degree: np.ndarray

    @property
    def n_nodes(self) -> int:
        return self.adj.shape[0]

    @property
    def n_edges(self) -> int:
        return int(self.adj.nnz // 2)


def empty_graph(n_nodes: int) -> Graph:
    return Graph(adj=sparse.csr_matrix((n_nodes, n_nodes), dtype=np.float32),
                 degree=np.zeros(n_nodes, dtype=np.float32))


def _mutual(adj: sparse.csr_matrix) -> sparse.csr_matrix:
    """
    Keep only mutually chosen neighbours (mutual k-NN).

    Symmetrising a per-row top-k with ``maximum`` would let a node accumulate
    every edge that *others* kept for it, so the degree bound would not hold.
    Intersecting instead (``minimum``) keeps the graph symmetric **and** bounded:
    an edge survives only if both endpoints rank each other.  The two sides carry
    the same weight because the underlying affinities are symmetric.
    """
    return adj.minimum(adj.T).tocsr()


def _top_k_per_row(adj: sparse.csr_matrix, top_k: int) -> sparse.csr_matrix:
    """Keep only the ``top_k`` strongest entries of each row."""
    adj = adj.tocsr()
    indptr, indices, data = adj.indptr, adj.indices, adj.data
    keep_i, keep_j, keep_v = [], [], []
    for row in range(adj.shape[0]):
        start, end = indptr[row], indptr[row + 1]
        if end - start <= top_k:
            keep_i.extend([row] * (end - start))
            keep_j.extend(indices[start:end].tolist())
            keep_v.extend(data[start:end].tolist())
            continue
        width = end - start
        order = np.argpartition(-data[start:end], top_k - 1)[:top_k]
        keep_i.extend([row] * top_k)
        keep_j.extend(indices[start:end][order].tolist())
        keep_v.extend(data[start:end][order].tolist())
    return sparse.csr_matrix(
        (np.asarray(keep_v, dtype=np.float32),
         (np.asarray(keep_i), np.asarray(keep_j))),
        shape=adj.shape,
    )


def syntax_graph(
    dep_src: np.ndarray,
    dep_dst: np.ndarray,
    dep_weight: np.ndarray,
    n_vocab: int,
    top_k: int = 15,
) -> Graph:
    """
    Build the syntactic graph from aggregated dependency pairs.

    The graph is symmetrised (a dependency arc is evidence of relatedness in both
    directions), row-normalised so a hub word cannot dominate, and pruned to a
    **mutual** top-``top_k`` neighbourhood.
    """
    if len(dep_src) == 0:
        return empty_graph(n_vocab)

    w = np.asarray(dep_weight, dtype=np.float32)
    adj = sparse.coo_matrix(
        (np.concatenate([w, w]),
         (np.concatenate([dep_src, dep_dst]), np.concatenate([dep_dst, dep_src]))),
        shape=(n_vocab, n_vocab),
    ).tocsr()
    adj.sum_duplicates()
    adj.setdiag(0.0)
    adj.eliminate_zeros()

    # Row-max normalisation keeps a hub word from dominating its neighbours.
    row_max = np.asarray(adj.max(axis=1).todense()).ravel()
    row_max[row_max == 0] = 1.0
    inv = sparse.diags((1.0 / row_max).astype(np.float32))
    adj = inv @ adj

    adj = _mutual(_top_k_per_row(adj, top_k))
    return Graph(adj=adj.astype(np.float32), degree=_degree(adj))


def similarity_graph(vectors: np.ndarray, top_k: int = 15) -> Graph:
    """
    Cosine k-NN graph over word vectors.

    Uses sklearn's neighbour search rather than materialising the full V x V
    similarity matrix (which would be V^2 floats).
    """
    n = vectors.shape[0]
    if n == 0 or vectors.shape[1] == 0:
        return empty_graph(n)

    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import normalize

    unit = normalize(np.asarray(vectors, dtype=np.float32), norm="l2", axis=1)
    k = min(top_k + 1, n)
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
    nn.fit(unit)
    _, indices = nn.kneighbors(unit)

    rows = np.repeat(np.arange(n), k - 1)
    cols = indices[:, 1:].ravel()          # drop self
    sims = np.einsum("ij,ij->i", unit[rows], unit[cols])
    sims = np.clip(sims, 0.0, None)        # keep it non-negative for the Laplacian

    adj = sparse.csr_matrix((sims.astype(np.float32), (rows, cols)), shape=(n, n))
    adj = _mutual(adj)                     # mutual k-NN: symmetric and bounded
    return Graph(adj=adj, degree=_degree(adj))


def _degree(adj: sparse.csr_matrix) -> np.ndarray:
    return np.asarray(adj.sum(axis=1)).ravel().astype(np.float32)


# ---------------------------------------------------------------------------
# Controls for the "does syntax matter?" question
# ---------------------------------------------------------------------------
#
# The claim "syntactic edges add information" is only testable against a control
# with the *same graph statistics* but no linguistic content.  Two controls,
# answering two different questions:
#
# * ``random_graph_matched`` — configuration model: the same degree sequence and
#   the same weight distribution, random topology.  Asks "is it the *content* of
#   the edges, or just having more regularisation?"
# * ``permuted_graph`` — node relabeling: literally the same graph, attached to
#   the wrong words.  Asks "is it the *pairing with words*, or any smoothing?"


def _undirected_edges(adj: sparse.csr_matrix):
    coo = sparse.triu(adj, k=1).tocoo()
    return coo.row.astype(np.int64), coo.col.astype(np.int64), coo.data.astype(np.float32)


def random_graph_matched(
    graph: Graph, seed: int = 0, n_swaps: Optional[int] = None
) -> Graph:
    """
    Degree-preserving random control, by 2-edge swap randomisation.

    Repeatedly picks two edges ``(u,v)`` and ``(x,y)`` and proposes
    ``(u,y)``, ``(x,v)``, rejecting any swap that would create a self-loop or a
    duplicate edge.  Every node keeps **exactly** its original degree, the edge
    count and the weight multiset are unchanged, and the only thing destroyed is
    the topology — which is precisely the nuisance variable to control for when
    asking whether the *content* of syntactic edges matters.
    """
    n = graph.n_nodes
    src, dst, w = _undirected_edges(graph.adj)
    if src.size == 0:
        return empty_graph(n)

    rng = np.random.default_rng(seed)
    edges = [(int(a), int(b), float(c)) for a, b, c in zip(src, dst, w)]
    present = {(min(a, b), max(a, b)) for a, b, _ in edges}
    swaps = 10 * len(edges) if n_swaps is None else int(n_swaps)

    for _ in range(swaps):
        i, j = rng.integers(0, len(edges), size=2)
        if i == j:
            continue
        u, v, wu = edges[i]
        x, y, wy = edges[j]
        if len({u, v, x, y}) < 4:            # need four distinct endpoints
            continue
        new_i, new_j = (u, y), (x, v)
        if new_i[0] == new_i[1] or new_j[0] == new_j[1]:
            continue
        key_i = (min(new_i), max(new_i))
        key_j = (min(new_j), max(new_j))
        if key_i == key_j or key_i in present or key_j in present:
            continue
        present.discard((min(u, v), max(u, v)))
        present.discard((min(x, y), max(x, y)))
        present.add(key_i)
        present.add(key_j)
        edges[i] = (u, y, wu)
        edges[j] = (x, v, wy)

    a = np.array([e[0] for e in edges], dtype=np.int64)
    b = np.array([e[1] for e in edges], dtype=np.int64)
    vals = np.array([e[2] for e in edges], dtype=np.float32)
    adj = sparse.coo_matrix(
        (np.concatenate([vals, vals]), (np.concatenate([a, b]), np.concatenate([b, a]))),
        shape=(n, n),
    ).tocsr()
    adj.setdiag(0.0)
    adj.eliminate_zeros()
    return Graph(adj=adj.astype(np.float32), degree=_degree(adj))


def permuted_graph(graph: Graph, seed: int = 0) -> Graph:
    """
    Node-relabeled control: identical topology, attached to shuffled words.

    This preserves degrees, weights and every graph statistic exactly; only the
    correspondence between graph nodes and vocabulary entries is destroyed.
    """
    n = graph.n_nodes
    src, dst, w = _undirected_edges(graph.adj)
    if src.size == 0:
        return empty_graph(n)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    adj = sparse.coo_matrix(
        (np.concatenate([w, w]),
         (np.concatenate([perm[src], perm[dst]]),
          np.concatenate([perm[dst], perm[src]]))),
        shape=(n, n),
    ).tocsr()
    adj.setdiag(0.0)
    adj.eliminate_zeros()
    return Graph(adj=adj.astype(np.float32), degree=_degree(adj))
