"""
scptm_bow/metrics.py
--------------------
Topic-quality metrics, computed with numpy/scipy only (no gensim, no torch).

Canonical keys — identical to ``scptm.metrics`` so results from the two packages
are directly comparable:

* ``c_v``         — C_V coherence (Röder et al. 2015): sliding-window NPMI
                    context vectors, cosine similarity to the topic centroid.
* ``npmi``        — mean NPMI over document co-occurrence of the top words.
* ``diversity``   — unique words across all top-word lists.
* ``exclusivity`` — how exclusively a topic's words belong to it.
* ``redundancy``  — mean pairwise Jaccard overlap between topics.

``c_v`` and ``npmi`` are *different* estimators on purpose: ``npmi`` uses
document co-occurrence (cheap, robust), ``c_v`` uses a token sliding window
(the standard from the coherence literature).  Reporting both is what makes a
comparison against gensim's ``CoherenceModel`` meaningful.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy import sparse

METRIC_KEYS = ("c_v", "npmi", "diversity", "exclusivity", "redundancy")


def _binary(bow: sparse.csr_matrix) -> sparse.csr_matrix:
    out = bow.copy()
    out.data = np.ones_like(out.data, dtype=np.float32)
    return out


# ---------------------------------------------------------------------------
# C_V
# ---------------------------------------------------------------------------

def _cooccurrence_window(
    tokenized: Sequence[Sequence[str]],
    index: Dict[str, int],
    n_vocab: int,
    window: int,
) -> sparse.csr_matrix:
    """Symmetric sliding-window co-occurrence counts over tokenised documents."""
    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []
    for doc in tokenized:
        ids = np.fromiter((index[w] for w in doc if w in index), dtype=np.int64)
        if ids.size < 2:
            continue
        length = ids.size
        for offset in range(1, window + 1):
            if length <= offset:
                break
            left, right = ids[: length - offset], ids[offset:]
            keep = left != right
            rows.append(left[keep])
            cols.append(right[keep])

    if not rows:
        return sparse.csr_matrix((n_vocab, n_vocab), dtype=np.float64)
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    data = np.ones(r.shape[0], dtype=np.float64)
    cooc = sparse.csr_matrix((data, (r, c)), shape=(n_vocab, n_vocab))
    return (cooc + cooc.T).tocsr()


def _cooccurrence_documents(bow: sparse.csr_matrix) -> sparse.csr_matrix:
    """Whole-document co-occurrence, used when no token order is available."""
    binary = _binary(bow).astype(np.float64).tocsr()
    cooc = (binary.T @ binary).tocsr()
    cooc.setdiag(0.0)
    cooc.eliminate_zeros()
    return cooc


def _npmi_matrix(cooc: sparse.csr_matrix) -> sparse.csr_matrix:
    """
    NPMI over a co-occurrence matrix.

        NPMI(i,j) = log( C_ij * N / (C_i * C_j) ) / -log( C_ij / N )

    Absent pairs stay implicit zeros (their NPMI is undefined and they carry no
    context information).
    """
    cooc = cooc.tocoo()
    total = float(cooc.data.sum())
    if total <= 0.0:
        return sparse.csr_matrix(cooc.shape, dtype=np.float64)

    counts_i = np.asarray(cooc.sum(axis=1)).ravel()
    counts_j = np.asarray(cooc.sum(axis=0)).ravel()

    eps = 1e-12
    c_ij = cooc.data
    p_ij = c_ij / total
    p_i = counts_i[cooc.row]
    p_j = counts_j[cooc.col]
    denom = -np.log(np.maximum(p_ij, eps))
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.log(c_ij * total / np.maximum(p_i * p_j, eps)) / denom
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return sparse.csr_matrix((values, (cooc.row, cooc.col)), shape=cooc.shape)


def c_v_coherence(
    top_words: Sequence[Sequence[str]],
    bow: Optional[sparse.csr_matrix] = None,
    vocab: Optional[Sequence[str]] = None,
    tokenized: Optional[Sequence[Sequence[str]]] = None,
    top_k: int = 10,
    window: int = 10,
    n_context: int = 10,
) -> float:
    """
    C_V coherence (Röder, Both & Hinneburg 2015).

    For every topic: build the context set ``S`` as the union of the ``n_context``
    most NPMI-similar words of each top word, aggregate the top words' NPMI rows
    into a topic vector, and average the cosine similarity between each word's
    context vector and the topic vector.  Higher is better.

    Parameters
    ----------
    tokenized : sequence of sequences of str, optional
        Lemmatised documents **with token order** — gives the canonical
        sliding-window estimator.  When omitted, ``bow`` is used with
        whole-document co-occurrence, which is a coarser but still usable
        approximation (documented, not silent).
    window : int
        Sliding-window width in tokens.
    n_context : int
        Size of each word's context set.
    """
    if not top_words:
        return float("nan")
    if tokenized is not None:
        if vocab is None:
            vocab = sorted({w for doc in tokenized for w in doc})
        index = {w: i for i, w in enumerate(vocab)}
        cooc = _cooccurrence_window(tokenized, index, len(vocab), window)
    elif bow is not None:
        if vocab is None:
            raise ValueError("c_v_coherence needs `vocab` together with `bow`")
        index = {w: i for i, w in enumerate(vocab)}
        cooc = _cooccurrence_documents(bow)
    else:
        raise ValueError("c_v_coherence needs either `tokenized` or `bow` + `vocab`")

    npmi = _npmi_matrix(cooc)
    n_vocab = npmi.shape[1]

    scores: List[float] = []
    for words in top_words:
        idx = [index[w] for w in words[:top_k] if w in index]
        idx = [i for i in idx if 0 <= i < n_vocab]
        if len(idx) < 2:
            continue

        rows = npmi[idx, :].toarray()                # (n_words, V), |rows| <= 10
        # Context set: the n_context strongest NPMI partners of each top word.
        context: List[int] = []
        seen = set()
        for row in rows:
            order = np.argsort(-row)[:n_context]
            for j in order:
                if row[j] > 0.0 and j not in seen:
                    seen.add(int(j))
                    context.append(int(j))
        if not context:
            continue

        block = rows[:, context]                      # (n_words, |S|)
        topic_vec = block.sum(axis=0)                 # v(t)

        def _cos(a: np.ndarray, b: np.ndarray) -> float:
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na == 0.0 or nb == 0.0:
                return 0.0
            return float(np.dot(a, b) / (na * nb))

        scores.append(float(np.mean([_cos(row, topic_vec) for row in block])))

    return float(np.mean(scores)) if scores else float("nan")


# ---------------------------------------------------------------------------
# NPMI, diversity, exclusivity, redundancy
# ---------------------------------------------------------------------------

def npmi_coherence(
    top_words: Sequence[Sequence[str]],
    bow: sparse.csr_matrix,
    vocab: Sequence[str],
    top_k: int = 10,
) -> float:
    """
    Mean NPMI over the top-``top_k`` words of every topic.

    Document co-occurrence.  Pairs that never co-occur contribute -1 (the usual
    NPMI convention), so the metric punishes topics whose words are unrelated.
    """
    index = {w: i for i, w in enumerate(vocab)}
    n_docs = float(bow.shape[0])
    if n_docs == 0:
        return float("nan")
    binary = _binary(bow).tocsr()
    doc_freq = np.asarray(binary.sum(axis=0)).ravel()

    per_topic: List[float] = []
    for words in top_words:
        idx = [index[w] for w in words[:top_k] if w in index]
        if len(idx) < 2:
            continue
        pair_scores: List[float] = []
        for a_pos in range(len(idx)):
            for b_pos in range(a_pos + 1, len(idx)):
                a, b = idx[a_pos], idx[b_pos]
                co = float(binary[:, a].multiply(binary[:, b]).sum())
                if co <= 0.0:
                    pair_scores.append(-1.0)
                    continue
                p_ij = co / n_docs
                p_i = doc_freq[a] / n_docs
                p_j = doc_freq[b] / n_docs
                denom = -np.log(p_ij)
                pair_scores.append(
                    0.0 if denom == 0 else float(np.log(p_ij / (p_i * p_j)) / denom)
                )
        if pair_scores:
            per_topic.append(float(np.mean(pair_scores)))
    return float(np.mean(per_topic)) if per_topic else float("nan")


def topic_diversity(top_words: Sequence[Sequence[str]]) -> float:
    """Unique words across all topics / total words."""
    flat = [w for words in top_words for w in words]
    return float(len(set(flat)) / len(flat)) if flat else float("nan")


def topic_exclusivity(H: np.ndarray, top_words: Sequence[Sequence[str]],
                      vocab: Sequence[str], top_n: int = 10) -> float:
    """
    Mean exclusivity of the top words, from the topic-term matrix ``H`` (K x V).

    Exclusivity of word w for topic k is p(w|k) / sum_k' p(w|k'): 1.0 means the
    word belongs only to that topic.
    """
    H = np.asarray(H, dtype=np.float64)
    col = H.sum(axis=0, keepdims=True)
    col[col == 0] = 1.0
    exclusivity = H / col
    index = {w: i for i, w in enumerate(vocab)}
    values: List[float] = []
    for k, words in enumerate(top_words):
        idx = [index[w] for w in words[:top_n] if w in index]
        if idx:
            values.append(float(exclusivity[k, idx].mean()))
    return float(np.mean(values)) if values else float("nan")


def redundancy(top_words: Sequence[Sequence[str]]) -> float:
    """
    Mean pairwise Jaccard overlap between topics' top words.

    0.0 = no topic shares a word with another; higher means near-duplicate
    topics, which is precisely what NPMI x diversity is supposed to avoid.
    """
    sets = [set(words) for words in top_words if words]
    if len(sets) < 2:
        return 0.0
    overlaps: List[float] = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = sets[i] | sets[j]
            if union:
                overlaps.append(len(sets[i] & sets[j]) / len(union))
    return float(np.mean(overlaps)) if overlaps else 0.0


def summary(
    top_words: Sequence[Sequence[str]],
    bow: sparse.csr_matrix,
    vocab: Sequence[str],
    H: Optional[np.ndarray] = None,
    top_k: int = 10,
    tokenized: Optional[Sequence[Sequence[str]]] = None,
) -> Dict[str, float]:
    """
    Every metric in one dict, under the canonical keys in ``METRIC_KEYS``.

    ``c_v`` is NaN when neither ``tokenized`` nor a usable co-occurrence matrix
    is available — never silently omitted, so the key set is stable.
    """
    out: Dict[str, float] = {
        "c_v": c_v_coherence(top_words, bow, vocab, tokenized=tokenized, top_k=top_k),
        "npmi": npmi_coherence(top_words, bow, vocab, top_k=top_k),
        "diversity": topic_diversity(top_words),
        "exclusivity": float("nan"),
        "redundancy": redundancy(top_words),
    }
    if H is not None:
        out["exclusivity"] = topic_exclusivity(H, top_words, vocab, top_n=top_k)
    return out
