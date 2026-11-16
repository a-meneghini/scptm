"""
scptm_bow/vectors.py
---------------------
Cheap word vectors — fastText when ``gensim`` is available, PPMI+SVD otherwise.

Both produce a plain ``(V, dim)`` float array over the vocabulary, which is all
the semantic graph needs.  No transformer, no download, no GPU.

fastText is trained **on the corpus**, and the dependency pairs collected during
parsing are fed in as extra two-token sentences: that nudges the vectors toward
syntactic neighbours, so the semantic graph partly absorbs the structure too.
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse

from .config import BowConfig
from .parse import ParsedCorpus


def build_word_vectors(parsed: ParsedCorpus, cfg: BowConfig) -> Optional[np.ndarray]:
    """
    Return a ``(V, cfg.vector_dim)`` array of word vectors, or ``None`` when
    ``vector_backend="none"``.
    """
    if cfg.vector_backend == "none":
        return None

    if cfg.vector_backend == "fasttext":
        try:
            return _fasttext_vectors(parsed, cfg)
        except ImportError:
            warnings.warn(
                "[vectors] 'gensim' is not installed; falling back to the "
                "PPMI+SVD backend. Install gensim for fastText, or set "
                "vector_backend='ppmi_svd' to silence this warning.",
                stacklevel=2,
            )
    return _ppmi_svd_vectors(parsed, cfg)


# ---------------------------------------------------------------------------
# fastText
# ---------------------------------------------------------------------------

def _dependency_sentences(parsed: ParsedCorpus) -> List[List[str]]:
    """Two-token 'sentences' built from aggregated dependency pairs."""
    pairs = {(head, dep) for (head, _, dep) in parsed.dep_counts}
    return [[a, b] for a, b in pairs if a != b]


def _fasttext_vectors(parsed: ParsedCorpus, cfg: BowConfig) -> np.ndarray:
    from gensim.models import FastText

    sentences = [doc.split() for doc in parsed.lemma_docs]
    sentences.extend(_dependency_sentences(parsed))
    print(
        f"  word vectors: fastText on {len(sentences):,} sentences "
        f"(corpus + dependency pairs)"
    )

    model = FastText(
        sentences=sentences,
        vector_size=cfg.vector_dim,
        window=cfg.fasttext_window,
        min_count=1,                 # vocabulary pruning already happened
        epochs=cfg.fasttext_epochs,
        sg=1,                        # skip-gram: better for small corpora
        seed=cfg.random_state,
        workers=1,                   # deterministic
    )

    dim = cfg.vector_dim
    out = np.zeros((parsed.n_vocab, dim), dtype=np.float32)
    missing = 0
    for i, word in enumerate(parsed.vocab):
        try:
            out[i] = model.wv[word]
        except KeyError:
            missing += 1
    if missing:
        warnings.warn(
            f"[vectors] {missing}/{parsed.n_vocab} vocabulary words had no "
            "fastText vector (left as zeros).",
            stacklevel=2,
        )
    return out


# ---------------------------------------------------------------------------
# PPMI + truncated SVD (pure numpy/scipy fallback)
# ---------------------------------------------------------------------------

def _cooccurrence(parsed: ParsedCorpus, window: int) -> Tuple[sparse.csr_matrix, Dict[str, int]]:
    """Symmetric co-occurrence counts over vocabulary terms, within a window."""
    index = {w: i for i, w in enumerate(parsed.vocab)}
    rows: List[int] = []
    cols: List[int] = []
    for doc in parsed.lemma_docs:
        ids = [index[t] for t in doc.split() if t in index]
        for pos, src in enumerate(ids):
            for other in ids[pos + 1: pos + 1 + window]:
                rows.append(src)
                cols.append(other)
                rows.append(other)
                cols.append(src)
    if not rows:
        return sparse.csr_matrix((parsed.n_vocab, parsed.n_vocab), dtype=np.float32), index
    mat = sparse.coo_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(parsed.n_vocab, parsed.n_vocab),
    ).tocsr()
    mat.sum_duplicates()
    mat.setdiag(0.0)
    return mat, index


def _ppmi_svd_vectors(parsed: ParsedCorpus, cfg: BowConfig) -> np.ndarray:
    cooc, _ = _cooccurrence(parsed, cfg.fasttext_window)
    total = float(cooc.sum())
    if total <= 0:
        return np.zeros((parsed.n_vocab, cfg.vector_dim), dtype=np.float32)

    row_sums = np.asarray(cooc.sum(axis=1)).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = cooc.tocoo()
        expected = np.outer(row_sums, row_sums) / total
        mask = expected[pmi.row, pmi.col] > 0
        vals = np.zeros_like(pmi.data)
        vals[mask] = np.log(
            (pmi.data[mask] * total) / expected[pmi.row[mask], pmi.col[mask]]
        )
        ppmi = sparse.csr_matrix(
            (np.clip(vals, 0.0, None), (pmi.row, pmi.col)), shape=cooc.shape
        )

    k = min(cfg.vector_dim, min(ppmi.shape) - 1)
    if k < 1:
        return np.zeros((parsed.n_vocab, cfg.vector_dim), dtype=np.float32)
    print(f"  word vectors: PPMI + SVD (k={k})")
    u, s, _ = sparse.linalg.svds(ppmi.astype(np.float64), k=k)
    order = np.argsort(-s)
    vectors = u[:, order] * np.sqrt(s[order])

    if vectors.shape[1] < cfg.vector_dim:      # pad if k was capped
        pad = np.zeros((vectors.shape[0], cfg.vector_dim - vectors.shape[1]), dtype=np.float64)
        vectors = np.hstack([vectors, pad])
    return vectors.astype(np.float32)


# ---------------------------------------------------------------------------
# Document vectors
# ---------------------------------------------------------------------------

def document_vectors(bow: sparse.csr_matrix, vectors: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Mean of the word vectors present in each document.

    Used for visualisation / semantic metrics — cheap and embedding-free.
    """
    if vectors is None:
        return None
    dense = np.asarray(bow.todense(), dtype=np.float32)
    counts = dense.sum(axis=1, keepdims=True)
    counts[counts == 0] = 1.0
    return (dense @ vectors) / counts
