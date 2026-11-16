"""
Performance and caching tests for scptm.

Reconstruction note
-------------------
This file was accidentally overwritten during a refactor and its original source
was unrecoverable (it had never been committed to git).  It was rebuilt from the
compiled bytecode cache (``tests/__pycache__/test_performance.cpython-313.pyc``),
which preserved the exact list of test and helper names, their docstrings, and
their literal constants.  The bodies are reconstructed against the current API
and verified to pass, but they are **not** byte-identical to the original.
If an older copy exists elsewhere, prefer it.

What is guarded here: the expensive things must be computed once and reused —
pipelines are cached, SBERT is loaded from the local cache without a Hub
round-trip, and the vectorised beta must match a straightforward per-word
reference on ragged input.
"""
import os
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import scptm.nlp as nlp_mod  # noqa: E402
from scptm.nn import VariationalGraphTopicModel  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Tok:
    """Minimal stand-in for a spaCy token."""

    def __init__(self, lemma: str, text: str):
        self.lemma_ = lemma
        self.text = text
        self.is_alpha = True
        self.is_stop = False
        self.pos_ = "NOUN"
        self.tag_ = "NN"
        self.dep_ = "nsubj"
        self.head = self


def _fake_nlp():
    """A lemmatiser-shaped object that never touches the network."""
    class _Doc(list):
        pass

    class _NLP:
        def __call__(self, text):
            return _Doc(_Tok(w.lower(), w) for w in text.split())

        def pipe(self, texts, **kwargs):
            for text in texts:
                yield self(text)

    return _NLP()


class _ExplodingSBERT:
    """Any construction of this must fail loudly."""

    device = "cpu"

    def __init__(self, *args, **kwargs):
        raise AssertionError(
            "a cached SBERT must be reused, not constructed again (device/model "
            "configuration did not hit the pipeline cache)"
        )


class _FakeST:
    """Records how SentenceTransformer was constructed."""

    calls = []

    def __init__(self, name, device="cpu", **kwargs):
        self.name = name
        self.device = device
        _FakeST.calls.append(kwargs)


class _FakeSTNotCached(_FakeST):
    """Fails the local-only load, so the network path is the only way through."""

    calls = []

    def __init__(self, name, device="cpu", **kwargs):
        _FakeSTNotCached.calls.append(kwargs)
        if kwargs.get("local_files_only"):
            raise OSError("model is not in the local cache")

    def encode(self, texts, **kwargs):
        return np.zeros((len(texts), 8), dtype=np.float32)


def _with_fake_st(fake=None):
    _FakeST.calls = []
    _FakeSTNotCached.calls = []
    return mock.patch.object(nlp_mod, "SentenceTransformer", fake or _FakeST)


def _reference_beta(encoder, ctx_embs_list, static_word_embs):
    """
    Straightforward per-word implementation (the previous behaviour).

    Same mathematics as the batched version: attention over each word's contexts
    (logits = context . topic, softmax over the contexts), then cosine similarity
    between the attended representation and each topic.
    """
    K = encoder.num_topics
    V = encoder.vocab_size
    topic = encoder.topic_embeddings.detach()
    beta = torch.zeros(K, V)
    for v in range(V):
        ctx = ctx_embs_list[v]
        if ctx is None or len(ctx) == 0:
            emb = static_word_embs[v].detach()
            beta[:, v] = torch.nn.functional.cosine_similarity(
                topic, emb.unsqueeze(0), dim=1
            )
            continue
        ctx = ctx.detach()
        logits = ctx @ topic.T                       # (N, K)
        attn = torch.softmax(logits, dim=0)          # over contexts
        repr_k = attn.T @ ctx                        # (K, D)
        beta[:, v] = torch.nn.functional.cosine_similarity(repr_k, topic, dim=1)
    return torch.softmax(beta / encoder.beta_temperature, dim=1)


def _build_encoder(num_topics=3, vocab_size=7, dim=8, seed=0):
    torch.manual_seed(seed)
    encoder = VariationalGraphTopicModel(
        in_channels=dim,
        hidden_channels=8,
        num_topics=num_topics,
        vocab_size=vocab_size,
        graph_mode="filtered",
    )
    return encoder


# ---------------------------------------------------------------------------
# Pipeline cache
# ---------------------------------------------------------------------------

def test_pipeline_cache_avoids_reload():
    nlp_mod.clear_pipeline_cache()
    with mock.patch.object(nlp_mod, "_load_sentence_transformer", return_value=_FakeST("cached")):
        first = nlp_mod.setup_nlp_pipeline(lang="eng", verbose=False, device="cpu")
        second = nlp_mod.setup_nlp_pipeline(lang="eng", verbose=False, device="cpu")

    sbert_a, nlp_a, stop_a = first
    sbert_b, nlp_b, stop_b = second
    assert sbert_a is sbert_b, "cached SBERT was not reused"
    assert nlp_a is nlp_b, "cached lemmatiser was not reused"
    assert stop_a is stop_b


def test_clear_pipeline_cache_reports_and_empties():
    nlp_mod.clear_pipeline_cache()
    with mock.patch.object(nlp_mod, "_load_sentence_transformer", return_value=_FakeST("cached")):
        nlp_mod.setup_nlp_pipeline(lang="eng", verbose=False, device="cpu")
    freed = nlp_mod.clear_pipeline_cache()
    assert freed >= 1
    assert nlp_mod.clear_pipeline_cache() == 0


# ---------------------------------------------------------------------------
# collect_contextual_embeddings
# ---------------------------------------------------------------------------

def test_collect_reuses_given_doc_embeddings():
    from scptm.graph import collect_contextual_embeddings

    documents = ["alpha beta gamma", "alpha beta delta"]
    vocab = {"alpha": 0, "beta": 1, "gamma": 2, "delta": 3}
    doc_embs = np.zeros((2, 8), dtype=np.float32)

    class _SBERT:
        device = "cpu"

        def __init__(self):
            self.encode_calls = 0

        def encode(self, texts, **kwargs):
            self.encode_calls += 1
            return np.zeros((len(texts), 8), dtype=np.float32)

    sbert = _SBERT()
    collect_contextual_embeddings(
        documents, _fake_nlp(), sbert, vocab, doc_embeddings=doc_embs
    )
    assert sbert.encode_calls == 0, (
        "doc_embeddings were supplied, so SBERT must not re-encode the corpus"
    )


def test_collect_rejects_mismatched_doc_embeddings():
    from scptm.graph import collect_contextual_embeddings

    documents = ["alpha beta gamma", "alpha beta delta"]
    vocab = {"alpha": 0, "beta": 1, "gamma": 2, "delta": 3}
    doc_embs = np.zeros((1, 8), dtype=np.float32)      # one row for two documents

    class _SBERT:
        device = "cpu"

        def encode(self, texts, **kwargs):
            return np.zeros((len(texts), 8), dtype=np.float32)

    raised = False
    try:
        collect_contextual_embeddings(
            documents, _fake_nlp(), _SBERT(), vocab, doc_embeddings=doc_embs
        )
    except ValueError:
        raised = True
    assert raised, "a row/document mismatch must raise, not be silently accepted"


# ---------------------------------------------------------------------------
# Vectorised beta
# ---------------------------------------------------------------------------

def test_contextual_beta_matches_reference_on_ragged_input():
    encoder = _build_encoder(num_topics=3, vocab_size=7, dim=8)
    torch.manual_seed(1)
    static = torch.randn(7, 8)

    # ragged: words 0,2,3 have context, 1,4 have none, 5 has one, 6 has five
    sizes = [4, 0, 1, 7, 0, 1, 5]
    ctx = [torch.randn(n, 8) if n else None for n in sizes]

    got = encoder.compute_contextual_beta(ctx, static)
    expected = _reference_beta(encoder, ctx, static)
    assert got.shape == (3, 7)
    assert torch.allclose(got, expected, atol=1e-5), (
        f"vectorised beta diverged from the reference by "
        f"{(got - expected).abs().max().item():.2e}"
    )
    assert torch.allclose(got.sum(dim=1), torch.ones(3), atol=1e-4)


# ---------------------------------------------------------------------------
# SBERT loading: local cache first, network only if missing, never offline
# ---------------------------------------------------------------------------

def test_sbert_is_loaded_from_cache_without_network():
    with _with_fake_st():
        nlp_mod._load_sentence_transformer("some-model", "cpu")
    assert _FakeST.calls, "SentenceTransformer was never constructed"
    kwargs = _FakeST.calls[0]
    assert kwargs.get("local_files_only") is True, (
        "a cached model must be loaded with local_files_only=True"
    )


def test_sbert_falls_back_to_network_only_when_missing():
    with _with_fake_st(_FakeSTNotCached):
        nlp_mod._load_sentence_transformer("some-model", "cpu")
    calls = _FakeSTNotCached.calls
    assert len(calls) == 2, f"expected local then network, got {calls}"
    assert calls[0].get("local_files_only") is True, "an uncached model should try local first"
    assert not calls[1].get("local_files_only"), "then it should fall back to the network"


def test_hf_offline_env_never_downloads():
    with _with_fake_st(_FakeSTNotCached), mock.patch.dict(
        os.environ, {"SCPTM_HF_OFFLINE": "1"}
    ):
        raised = False
        try:
            nlp_mod._load_sentence_transformer("some-model", "cpu")
        except OSError:
            raised = True
    assert raised, "SCPTM_HF_OFFLINE=1 must raise instead of downloading"
    for kwargs in _FakeSTNotCached.calls:
        assert kwargs.get("local_files_only") is True, (
            "no network attempt is allowed when offline"
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:                            # noqa: BLE001
            failures += 1
            print(f"  FAIL  {name}\n        {type(exc).__name__}: {exc}")
    print(f"{'OK' if failures == 0 else 'FAILED'} ({failures} failure(s))")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
