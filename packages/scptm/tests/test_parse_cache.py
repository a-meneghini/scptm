"""
tests/test_parse_cache.py
=========================
Tests for the parse/embedding cache signature.

The cache must be invalidated whenever anything the cached artifacts depend on
changes — corpus content, language, lemmatiser, graph mode, ``min_df`` or
``max_features`` — and only then.  In particular, a vocabulary built with a
different ``min_df`` must never be reused silently (that was a real bug).

These tests are hermetic: no model, download or network access is needed.  The
graph-level test uses a tiny fake parser built from the token classes shipped
in :mod:`scptm.nlp`.

Run with pytest::

    pytest tests/test_parse_cache.py -v

or directly (no pytest required)::

    python tests/test_parse_cache.py
"""

import os
import tempfile
import traceback
import warnings
from pathlib import Path

import numpy as np

from scptm.config import SCPTMConfig
from scptm.graph import (
    VOCAB_TOKEN_PATTERN,
    _cache_signature,
    _corpus_hash,
    _load_parse_cache,
    _save_parse_cache,
    _signature_diff,
    build_hetero_graph,
    load_ctx_embs_from_cache,
    save_ctx_embs_to_cache,
)
from scptm.nlp import UDPipeDoc, UDPipeToken

DOCS_A = ["alpha beta gamma delta",
          "alpha beta epsilon zeta",
          "alpha eta theta iota"]
DOCS_B = ["alpha beta gamma delta",
          "alpha beta epsilon zeta",
          "alpha eta theta kappa"]        # same length/content shape, one word differs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _StubSBERT:
    """Deterministic stand-in for SentenceTransformer (no downloads)."""

    def __init__(self, dim: int = 16):
        self.dim = dim
        self.device = "cpu"

    def encode(self, texts, show_progress_bar=False):
        rng = np.random.default_rng(0)
        if isinstance(texts, str):
            return rng.standard_normal(self.dim).astype("float32")
        return rng.standard_normal((len(texts), self.dim)).astype("float32")


def _fake_nlp(text):
    """Whitespace tokeniser returning spaCy-compatible token objects."""
    words = [w for w in text.split() if w.isalpha()]
    tokens = [
        UDPipeToken(i, w, w.lower(), "NOUN", "NN", "nsubj", frozenset())
        for i, w in enumerate(words, start=1)
    ]
    for token in tokens:
        token.head = tokens[0]
    return UDPipeDoc(tokens)


def _save_dummy(path, signature, n_docs=3):
    _save_parse_cache(
        path,
        vocab_arr=np.array(["alpha", "beta"]),
        bow_sparse=None,
        doc_word_src=[0], doc_word_dst=[0],
        word_word_src=[0], word_word_dst=[1],
        n_docs=n_docs,
        signature=signature,
    )


# ---------------------------------------------------------------------------
# Corpus hash
# ---------------------------------------------------------------------------

def test_corpus_hash_is_stable():
    assert _corpus_hash(DOCS_A) == _corpus_hash(list(DOCS_A))


def test_corpus_hash_detects_content_change_same_n_docs():
    assert _corpus_hash(DOCS_A) != _corpus_hash(DOCS_B)
    assert len(DOCS_A) == len(DOCS_B)        # n_docs alone would miss this


# ---------------------------------------------------------------------------
# Signature
# ---------------------------------------------------------------------------

def test_cache_signature_fields():
    sig = _cache_signature(SCPTMConfig(lang="ita", min_df=7), DOCS_A)
    assert sig["n_docs"] == 3
    assert sig["lang"] == "ita"
    assert sig["lemmatizer"] == "spacy"
    assert sig["graph_mode"] == "filtered"
    assert sig["min_df"] == 7
    assert "max_features" in sig
    assert "content_hash" in sig
    # The vocabulary tokenizer rule must be part of the signature too: an old
    # ASCII-only pattern produced a vocabulary without accented words.
    assert sig["token_pattern"] == VOCAB_TOKEN_PATTERN


def test_signature_diff_reports_each_change():
    cfg = SCPTMConfig(min_df=1)
    base = _cache_signature(cfg, DOCS_A)

    same = _signature_diff(base, _cache_signature(SCPTMConfig(min_df=1), DOCS_A))
    assert same == []

    min_df = _signature_diff(base, _cache_signature(SCPTMConfig(min_df=5), DOCS_A))
    assert any("min_df" in r for r in min_df)

    mode = _signature_diff(base, _cache_signature(SCPTMConfig(graph_mode="none"), DOCS_A))
    assert any("graph_mode" in r for r in mode)

    content = _signature_diff(base, _cache_signature(cfg, DOCS_B))
    assert any("content" in r for r in content)

    # A vocabulary tokenizer change must invalidate the cache as well.
    legacy_pattern = dict(base, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    pattern = _signature_diff(legacy_pattern, base)
    assert any("token_pattern" in r for r in pattern)


# ---------------------------------------------------------------------------
# Save / load round-trip and invalidation
# ---------------------------------------------------------------------------

def test_load_accepts_matching_signature():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "c.pkl"
        sig = _cache_signature(SCPTMConfig(min_df=1), DOCS_A)
        _save_dummy(path, sig)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cache = _load_parse_cache(path, sig)
        assert cache is not None
        assert not [w for w in caught if "EdgeCache" in str(w.message)]


def test_load_rejects_min_df_change():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "c.pkl"
        _save_dummy(path, _cache_signature(SCPTMConfig(min_df=1), DOCS_A))
        new_sig = _cache_signature(SCPTMConfig(min_df=5), DOCS_A)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cache = _load_parse_cache(path, new_sig)
        assert cache is None
        assert any("min_df" in str(w.message) for w in caught)


def test_load_rejects_graph_mode_change():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "c.pkl"
        _save_dummy(path, _cache_signature(SCPTMConfig(graph_mode="filtered"), DOCS_A))
        new_sig = _cache_signature(SCPTMConfig(graph_mode="none"), DOCS_A)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cache = _load_parse_cache(path, new_sig)
        assert cache is None
        assert any("graph_mode" in str(w.message) for w in caught)


def test_load_rejects_content_change_with_same_n_docs():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "c.pkl"
        _save_dummy(path, _cache_signature(SCPTMConfig(), DOCS_A))
        new_sig = _cache_signature(SCPTMConfig(), DOCS_B)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cache = _load_parse_cache(path, new_sig)
        assert cache is None
        assert any("content" in str(w.message) for w in caught)


def test_load_rejects_legacy_cache_without_signature():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "c.pkl"
        _save_dummy(path, signature=None)          # pre-signature cache layout
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cache = _load_parse_cache(path, _cache_signature(SCPTMConfig(), DOCS_A))
        assert cache is None
        assert any("predates" in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# Contextual-embeddings cache
# ---------------------------------------------------------------------------

def test_ctx_cache_rejects_max_occurrences_change():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "c.pkl"
        _save_dummy(path, _cache_signature(SCPTMConfig(), DOCS_A))
        save_ctx_embs_to_cache(path, [None, None], max_occurrences=50)

        assert load_ctx_embs_from_cache(
            path, 2, lemmatizer="spacy", lang="eng", max_occurrences=50
        ) is not None

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert load_ctx_embs_from_cache(
                path, 2, lemmatizer="spacy", lang="eng", max_occurrences=10
            ) is None
        assert any("max_ctx_occurrences" in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# End-to-end: a min_df change must not reuse the stale vocabulary
# ---------------------------------------------------------------------------

def test_build_graph_does_not_reuse_stale_vocab_after_min_df_change():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "c.pkl"
        sbert = _StubSBERT()

        def build(min_df, cache):
            cfg = SCPTMConfig(graph_mode="filtered", min_df=min_df,
                              max_features=100, apply_chunking=False)
            _, vocab, _, _, _, _ = build_hetero_graph(
                DOCS_A, sbert, _fake_nlp, set(), cfg, edge_cache_path=cache
            )
            return vocab

        vocab_min1 = build(1, str(path))                      # builds + caches
        reference = build(2, None)                            # fresh, no cache

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            vocab_min2 = build(2, str(path))                  # must NOT reuse min_df=1
        assert any("min_df" in str(w.message) for w in caught), "must warn about min_df"
        assert vocab_min2 == reference, "min_df change must rebuild the vocabulary"
        assert len(vocab_min2) < len(vocab_min1)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            vocab_again = build(2, str(path))                 # now it may be reused
        assert vocab_again == vocab_min2
        assert not [w for w in caught if "EdgeCache" in str(w.message)]


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
