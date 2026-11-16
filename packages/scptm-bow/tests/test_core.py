"""
tests/test_core.py
==================
Hermetic tests for the pure components: config, cache store, graphs, metrics
and MWE extraction.  No parser and no model download required.

Run with pytest::

    pytest tests/test_core.py -v

or directly::

    python tests/test_core.py
"""

import os
import tempfile
import traceback
import warnings
from pathlib import Path

import numpy as np
from scipy import sparse

from scptm_bow.config import BowConfig
from scptm_bow.explain import MWE_RELATIONS, extract_mwes, topic_labels
from scptm_bow.graphs import similarity_graph, syntax_graph
from scptm_bow.parse import chunk_documents
from scptm_bow.metrics import (
    npmi_coherence,
    redundancy,
    topic_diversity,
    topic_exclusivity,
)
from scptm_bow.store import (
    cache_signature,
    corpus_hash,
    load_cache,
    save_cache,
    signature_diff,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_config_defaults_and_validation():
    cfg = BowConfig()
    assert cfg.num_topics == 10 and cfg.parser_backend == "spacy"

    for bad in (
        {"lang": "deu"},
        {"parser_backend": "nltk"},
        {"vector_backend": "word2vec"},
        {"num_topics": 1},
        {"min_df": 0},
        {"graph_top_k": 0},
        {"lambda_syntax": -1.0},
        {"nmf_max_iter": 0},
        {"divergence": "braycurtis"},
    ):
        raised = False
        try:
            BowConfig(**bad)
        except ValueError:
            raised = True
        assert raised, f"BowConfig(**{bad}) must raise ValueError"


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def test_chunking_drops_nothing_and_adds_no_periods():
    long_doc = "Prima parte molto lunga. " * 4 + "Coda breve."
    segments = chunk_documents(["Breve.", long_doc], max_chunk_chars=60)

    # Every document contributes at least one segment, and the short tail of a
    # long document survives too.
    joined = " ".join(segments)
    assert "Breve." in joined
    assert "Prima parte molto lunga." in joined
    assert "Coda breve." in joined
    assert ".." not in joined, "chunks must not get doubled periods"


def test_chunking_is_in_the_cache_signature():
    docs = ["alpha beta gamma. " * 10]
    plain = cache_signature(BowConfig(apply_chunking=False), docs)
    chunked = cache_signature(
        BowConfig(apply_chunking=True, max_chunk_chars=100), docs
    )
    assert plain["apply_chunking"] is False
    assert chunked["apply_chunking"] is True
    assert any("apply_chunking" in r for r in signature_diff(plain, chunked))


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

def test_corpus_hash_detects_edits_at_constant_length():
    a = ["alpha beta gamma", "delta epsilon"]
    b = ["alpha beta gamma", "delta zeta"]
    assert corpus_hash(a) == corpus_hash(list(a))
    assert corpus_hash(a) != corpus_hash(b)
    assert len(a) == len(b)


def test_signature_diff_names_the_changed_field():
    cfg = BowConfig(min_df=2)
    docs = ["alpha beta", "beta gamma"]
    base = cache_signature(cfg, docs)

    assert signature_diff(base, cache_signature(BowConfig(min_df=2), docs)) == []
    assert any(
        "min_df" in r for r in signature_diff(base, cache_signature(BowConfig(min_df=5), docs))
    )
    assert any(
        "content" in r for r in signature_diff(base, cache_signature(cfg, ["alpha beta", "beta delta"]))
    )


def test_cache_roundtrip_and_invalidation(tmpdir=None):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "c.pkl"
        cfg = BowConfig(min_df=2)
        docs = ["alpha beta", "beta gamma"]
        sig = cache_signature(cfg, docs)
        payload = {"vocab": np.array(["alpha", "beta"]), "n_docs": 2}

        save_cache(path, payload, sig)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            got = load_cache(path, sig)
        assert got is not None and got["n_docs"] == 2
        assert not [w for w in caught if "cache" in str(w.message)]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert load_cache(path, cache_signature(BowConfig(min_df=9), docs)) is None
        assert any("min_df" in str(w.message) for w in caught)


def test_cache_rejects_unsigned_payload():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "c.pkl"
        save_cache(path, {"vocab": np.array(["a"])}, {"unused": True})
        # Simulate a legacy payload: strip the signature.
        import pickle

        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        payload.pop("_signature")
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert load_cache(path, {"n_docs": 1}) is None
        assert any("no signature" in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------

def test_syntax_graph_is_symmetric_and_pruned():
    src = np.array([0, 0, 0, 1, 2], dtype=np.int32)
    dst = np.array([1, 2, 3, 2, 3], dtype=np.int32)
    weight = np.array([5.0, 3.0, 1.0, 2.0, 1.0], dtype=np.float32)
    g = syntax_graph(src, dst, weight, n_vocab=4, top_k=2)

    dense = g.adj.toarray()
    assert np.allclose(dense, dense.T), "adjacency must be symmetric"
    assert np.allclose(np.diag(dense), 0.0)
    assert np.all(dense >= 0)
    assert int((dense > 0).sum(axis=1).max()) <= 2, "top-k pruning failed"
    assert np.allclose(g.degree, dense.sum(axis=1))


def test_similarity_graph_keeps_semantic_neighbours():
    # Three vectors: 0 and 1 are close, 2 is orthogonal to both.
    vecs = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)
    g = similarity_graph(vecs, top_k=1)
    dense = g.adj.toarray()
    assert dense[0, 1] > 0, "the close pair must be connected"
    assert np.allclose(dense, dense.T)
    assert np.allclose(np.diag(dense), 0.0)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _toy_bow():
    # doc 0-1: alpha/beta ; doc 2-3: gamma/delta
    rows = [0, 0, 1, 1, 2, 2, 3, 3]
    cols = [0, 1, 0, 1, 2, 3, 2, 3]
    data = [1] * 8
    return sparse.csr_matrix((data, (rows, cols)), shape=(4, 4))


def test_npmi_rewards_coherent_topics():
    bow = _toy_bow()
    vocab = ["alpha", "beta", "gamma", "delta"]
    coherent = [["alpha", "beta"], ["gamma", "delta"]]
    incoherent = [["alpha", "gamma"], ["beta", "delta"]]
    assert npmi_coherence(coherent, bow, vocab) > npmi_coherence(incoherent, bow, vocab)
    # Pairs that never co-occur give the -1 convention.
    assert npmi_coherence([["alpha", "gamma"]], bow, vocab) == -1.0


def test_diversity_and_redundancy():
    assert topic_diversity([["a", "b"], ["c", "d"]]) == 1.0
    assert topic_diversity([["a", "b"], ["a", "b"]]) == 0.5
    assert redundancy([["a", "b"], ["c", "d"]]) == 0.0
    assert redundancy([["a", "b"], ["a", "b"]]) == 1.0


def test_exclusivity_from_topic_term_matrix():
    H = np.array([[0.9, 0.1, 0.0], [0.1, 0.9, 0.0]])
    vocab = ["a", "b", "c"]
    words = [["a"], ["b"]]
    assert topic_exclusivity(H, words, vocab) > 0.8


# ---------------------------------------------------------------------------
# MWE extraction
# ---------------------------------------------------------------------------

def test_extract_mwes_prefers_on_topic_pairs():
    triple_counts = {
        ("spirito", "amod", "santo"): 30,     # both on topic
        ("spirito", "amod", "santo2"): 5,
        ("cane", "amod", "gatto"): 40,        # off topic
        ("chiesa", "nsubj", "pregare"): 9,    # relation not lexical
    }
    topic_words = [["spirito", "santo", "chiesa"], ["cane", "gatto"]]
    mwes = extract_mwes(topic_words, triple_counts, top_k=5)

    phrases = [m["phrase"] for m in mwes["Topic_1"]]
    assert "spirito santo" in phrases
    assert "cane gatto" not in phrases          # off topic for Topic_1
    assert all(m["relation"] in MWE_RELATIONS for m in mwes["Topic_1"])
    # A relation outside the lexical set is ignored.
    assert not any(m["relation"] == "nsubj" for m in mwes["Topic_1"])

    labels = topic_labels(topic_words, mwes, n=3)
    assert len(labels) == 2 and "spirito santo" in labels[0]


# ---------------------------------------------------------------------------

if __name__ == "__main__":          # pragma: no cover - manual runner
    import sys

    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception:               # noqa: BLE001
            failures += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
    print(f"\n{'FAILED' if failures else 'OK'} ({failures} failure(s))")
    sys.exit(1 if failures else 0)
