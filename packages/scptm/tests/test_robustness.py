"""
tests/test_robustness.py
========================
Regression tests for the robustness fixes:

1. ``prepare_corpus`` used to drop every document whose final chunk was <= 50
   characters — including *whole* documents shorter than that.
2. ``search_k`` scored with ``max()`` over possibly-NaN values (order-dependent)
   and let a single failing K abort the whole sweep.
3. ``fit()`` mutated the caller's ``SCPTMConfig`` (and the mutated value was
   then pickled by ``save()``).
4. Neighbour sampling was auto-enabled without checking that pyg-lib /
   torch-sparse are installed, crashing mid-epoch with ImportError.

Hermetic: no model download, no network.  The config-immutability test replaces
the NLP pipeline and ``train`` with stubs.

Run with pytest::

    pytest tests/test_robustness.py -v

or directly (no pytest required)::

    python tests/test_robustness.py
"""

import traceback

import numpy as np

import scptm.model as model_mod
from scptm.config import SCPTMConfig, neighbor_sampling_available
from scptm.graph import MIN_DOC_CHARS, _split_sentences, prepare_corpus
from scptm.nlp import UDPipeDoc, UDPipeToken
from scptm.selection import _best_finite_k, _last_history_value

DOCS = [
    "Machine learning is transforming healthcare diagnostics rapidly.",
    "Deep neural networks achieve state of the art performance in natural language processing.",
    "Climate change accelerates biodiversity loss in tropical regions worldwide.",
    "Neural networks learn representations from large text corpora of documents.",
    "Hospitals deploy machine learning models to diagnose diseases earlier.",
    "Tropical rainforests harbour a large share of global biodiversity.",
]


# ---------------------------------------------------------------------------
# 1. prepare_corpus must not drop documents
# ---------------------------------------------------------------------------

def test_prepare_corpus_keeps_documents_above_min_length():
    raw = [
        "Breve.",                                  # below MIN_DOC_CHARS -> dropped
        "Una frase di lunghezza media.",           # 29 chars -> used to be dropped
        "Questo documento è abbastanza lungo da superare la soglia dei cinquanta caratteri.",
    ]
    docs, idx = prepare_corpus(raw, source_type="list", apply_chunking=True,
                               max_chunk_chars=800, return_doc_indices=True)
    assert len(docs) == 2, f"expected 2 segments, got {len(docs)}"
    assert idx == [1, 2], f"doc_indices must point at the kept documents, got {idx}"
    assert len(raw[0]) <= MIN_DOC_CHARS          # the dropped one really is tiny


def test_prepare_corpus_keeps_trailing_chunk():
    # One long document whose remainder after the first split is short: the tail
    # used to be discarded by `if len(chunk) > 50`.
    raw = ["Prima parte molto lunga. " * 4 + "Coda breve."]
    docs = prepare_corpus(raw, source_type="list", apply_chunking=True,
                          max_chunk_chars=80)
    joined = " ".join(docs)
    assert "Coda breve." in joined, "the trailing chunk was dropped"
    assert ".." not in joined, "chunks must not get doubled periods"


def test_prepare_corpus_no_chunking_matches_min_length():
    raw = ["Breve.", "Una frase di lunghezza media."]
    docs = prepare_corpus(raw, source_type="list", apply_chunking=False)
    assert len(docs) == 1
    assert docs[0] == "Una frase di lunghezza media."


def test_split_sentences_handles_all_terminators():
    assert _split_sentences("Ciao! Come stai? Bene. Ok") == [
        "Ciao!", "Come stai?", "Bene.", "Ok"
    ]


# ---------------------------------------------------------------------------
# 2. search_k helpers: NaN-safe and failure-safe
# ---------------------------------------------------------------------------

def test_last_history_value_is_nan_when_missing_or_empty():
    assert not np.isfinite(_last_history_value({}, "coherence_npmi"))
    assert not np.isfinite(_last_history_value({"coherence_npmi": []}, "coherence_npmi"))
    assert _last_history_value({"coherence_npmi": [0.1, 0.3]}, "coherence_npmi") == 0.3


def test_best_finite_k_ignores_nan_regardless_of_order():
    nan = float("nan")
    first = {5: {"score": nan}, 10: {"score": 0.42}, 15: {"score": 0.10}}
    second = {5: {"score": 0.42}, 10: {"score": nan}, 15: {"score": 0.10}}
    assert _best_finite_k(first) == 10      # NaN first must not win
    assert _best_finite_k(second) == 5


def test_best_finite_k_raises_when_everything_is_nan():
    raised = False
    try:
        _best_finite_k({5: {"score": float("nan")}, 10: {"score": float("inf")}})
    except RuntimeError:
        raised = True
    assert raised, "an all-non-finite sweep must fail loudly, not pick a K at random"


# ---------------------------------------------------------------------------
# 3./4. neighbour sampling availability and config immutability
# ---------------------------------------------------------------------------

def test_neighbor_sampling_available_returns_bool():
    assert isinstance(neighbor_sampling_available(), bool)


class _StubSBERT:
    device = "cpu"

    def __init__(self, dim: int = 8):
        self.dim = dim

    def encode(self, texts, show_progress_bar=False):
        rng = np.random.default_rng(0)
        if isinstance(texts, str):
            return rng.standard_normal(self.dim).astype("float32")
        return rng.standard_normal((len(texts), self.dim)).astype("float32")


def _fake_nlp(text):
    words = [w for w in text.split() if w.isalpha()]
    tokens = [
        UDPipeToken(i, w, w.lower(), "NOUN", "NN", "nsubj", frozenset())
        for i, w in enumerate(words, start=1)
    ]
    for token in tokens:
        token.head = tokens[0]
    return UDPipeDoc(tokens)


def _fit_with_stubs(cfg, capture):
    """Run fit() with the NLP pipeline and train() stubbed out."""
    originals = (
        model_mod.setup_nlp_pipeline,
        model_mod.train,
        model_mod.neighbor_sampling_available,
    )

    def fake_setup(*args, **kwargs):
        return _StubSBERT(), _fake_nlp, set()

    def fake_train(*args, **kwargs):
        capture["train_cfg"] = args[5]          # (graph, bow, nn, ctx, static, cfg, device)
        return {"loss": [], "recon": [], "kl": [], "kl_weight": [],
                "coherence_npmi": [], "topic_diversity": [], "metric_epochs": []}

    model_mod.setup_nlp_pipeline = fake_setup
    model_mod.train = fake_train
    model_mod.neighbor_sampling_available = lambda: True    # pretend pyg-lib is installed
    try:
        model_mod.SCPTM(config=cfg).fit_transform(DOCS)
    finally:
        (model_mod.setup_nlp_pipeline,
         model_mod.train,
         model_mod.neighbor_sampling_available) = originals


def _base_cfg(**over):
    base = dict(device="cpu", lang="eng", num_topics=2, epochs=1, min_df=1,
                max_features=100, apply_chunking=False,
                metrics_every_n_epochs=100, beta_refresh_epochs=100,
                neighbor_sampling_edge_threshold=0)
    base.update(over)
    return SCPTMConfig(**base)


def test_fit_does_not_mutate_user_config_when_auto_enabling():
    cfg = _base_cfg(use_neighbor_sampling=False)
    capture = {}
    _fit_with_stubs(cfg, capture)

    assert cfg.use_neighbor_sampling is False, "fit() mutated the caller's config"
    assert capture["train_cfg"] is not cfg, "training must use a private copy"
    assert capture["train_cfg"].use_neighbor_sampling is True, (
        "the auto-enable must still reach train()"
    )


def test_fit_does_not_mutate_user_config_when_present():
    cfg = _base_cfg(use_neighbor_sampling=True)
    capture = {}
    _fit_with_stubs(cfg, capture)

    assert cfg.use_neighbor_sampling is True
    assert capture["train_cfg"] is cfg, "no override needed -> same object is fine"


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
