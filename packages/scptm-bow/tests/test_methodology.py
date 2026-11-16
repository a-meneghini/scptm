"""
Methodological test suite for scptm-bow.

Covers the pieces that make a result *reportable*: the shared conformance fixture
(the same numbers as scptm), C_V behaving like a coherence measure, the matched
control graphs, held-out perplexity, bootstrap stability, covariate effects and
temporal drift.

Run:  cd scptm-bow && ../.venv/bin/python tests/test_methodology.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

HERE = Path(__file__).resolve().parent

def _find_conformance():
    """Locate conformance/ by walking up: the fixture belongs to the monorepo,
    not to the package, so it is not in the sdist."""
    for parent in (HERE, *HERE.parents):
        candidate = parent / "conformance" / "metrics_fixture.json"
        if candidate.is_file():
            return candidate
    return HERE / "conformance" / "metrics_fixture.json"   # absent -> test skips


sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

from test_pipeline import _cfg, _corpus, _install_fake_parser, _restore  # noqa: E402

import scptm_bow.metrics as M  # noqa: E402
import scptm_bow.stats as S  # noqa: E402
from scptm_bow import BowSCPTM  # noqa: E402
from scptm_bow.graphs import (  # noqa: E402
    permuted_graph,
    random_graph_matched,
    syntax_graph,
)

FIXTURE = _find_conformance()

# ---------------------------------------------------------------------------
# Conformance with scptm
# ---------------------------------------------------------------------------

def _load_fixture():
    if not FIXTURE.is_file():
        return None
    return json.loads(FIXTURE.read_text())


def _fixture_bow(payload):
    rows = [r for r, _, _ in payload["bow_nnz"]]
    cols = [c for _, c, _ in payload["bow_nnz"]]
    vals = [v for _, _, v in payload["bow_nnz"]]
    return sparse.csr_matrix(
        (np.asarray(vals), (np.asarray(rows), np.asarray(cols))),
        shape=tuple(payload["bow_shape"]),
    )


def test_metrics_reproduce_the_conformance_fixture():
    payload = _load_fixture()
    if payload is None:
        print("    (fixture assente, salto)")
        return
    bow = _fixture_bow(payload)
    got = {
        "c_v": M.c_v_coherence(payload["topics"], tokenized=payload["tokens"], window=5),
        "npmi": M.npmi_coherence(payload["topics"], bow, payload["vocab"]),
        "diversity": M.topic_diversity(payload["topics"]),
        "redundancy": M.redundancy(payload["topics"]),
    }
    for key, expected in payload["metrics"].items():
        assert abs(got[key] - expected) < 1e-10, (key, got[key], expected)


def test_stats_reproduce_the_conformance_fixture():
    payload = _load_fixture()
    if payload is None:
        return
    st = payload["stats"]
    a, b = st["a"], st["b"]
    assert abs(S.cliffs_delta(a, b) - st["cliffs_delta"]) < 1e-10
    assert abs(S.cohens_dz(a, b) - st["cohens_dz"]) < 1e-10
    assert abs(S.wilcoxon_paired(a, b).p_value - st["wilcoxon_p"]) < 1e-10
    ci = S.paired_bootstrap_ci(a, b, n_resamples=4000, seed=3)
    assert abs(ci.estimate - st["paired_ci_estimate"]) < 1e-10
    assert abs(ci.low - st["paired_ci_low"]) < 1e-10
    assert abs(ci.high - st["paired_ci_high"]) < 1e-10
    assert np.allclose(S.holm([0.01, 0.04, 0.03]), st["holm"], atol=1e-10)


# ---------------------------------------------------------------------------
# C_V behaves like a coherence measure
# ---------------------------------------------------------------------------

def _two_pool_corpus(n=40, seed=0):
    rng = np.random.default_rng(seed)
    a = "rocket orbit launch satellite mission".split()
    b = "hockey puck goal playoff team".split()
    docs = []
    for _ in range(n):
        docs.append([str(rng.choice(a)) for _ in range(8)])
        docs.append([str(rng.choice(b)) for _ in range(8)])
    return docs


def test_c_v_ranks_coherent_above_mixed():
    docs = _two_pool_corpus()
    coherent = [["rocket", "orbit", "launch", "satellite"], ["hockey", "puck", "goal", "playoff"]]
    mixed = [["rocket", "puck", "launch", "goal"], ["orbit", "hockey", "satellite", "playoff"]]
    good = float(np.mean([M.c_v_coherence([t], tokenized=docs) for t in coherent]))
    bad = float(np.mean([M.c_v_coherence([t], tokenized=docs) for t in mixed]))
    assert good > bad + 0.05, (good, bad)


def test_c_v_degenerate_cases():
    docs = _two_pool_corpus(n=10)
    assert np.isnan(M.c_v_coherence([["rocket"]], tokenized=docs))
    assert np.isnan(M.c_v_coherence([["zzz", "yyy"]], tokenized=docs))
    assert np.isnan(M.c_v_coherence([], tokenized=docs))
    raised = False
    try:
        M.c_v_coherence([["rocket", "orbit"]])
    except ValueError:
        raised = True
    assert raised, "C_V must ask for either tokenized or bow+vocab"


def test_summary_has_a_stable_key_set():
    docs = _two_pool_corpus(n=10)
    vocab = sorted({w for d in docs for w in d})
    index = {w: i for i, w in enumerate(vocab)}
    rows, cols = [], []
    for r, d in enumerate(docs):
        for w in d:
            rows.append(r)
            cols.append(index[w])
    bow = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(docs), len(vocab)))
    out = M.summary([vocab[:4], vocab[4:8]], bow, vocab)
    assert set(out) == set(M.METRIC_KEYS), (set(out), M.METRIC_KEYS)


# ---------------------------------------------------------------------------
# Matched control graphs
# ---------------------------------------------------------------------------

def _sample_graph(seed=0, n_vocab=200, n_edges=800, top_k=8):
    rng = np.random.default_rng(seed)
    return syntax_graph(
        rng.integers(0, n_vocab, n_edges),
        rng.integers(0, n_vocab, n_edges),
        rng.random(n_edges).astype(np.float32),
        n_vocab=n_vocab,
        top_k=top_k,
    )


def test_matched_control_preserves_degrees_and_breaks_topology():
    g = _sample_graph()
    ctrl = random_graph_matched(g, seed=1)
    d0 = np.asarray((g.adj > 0).sum(axis=1)).ravel()
    d1 = np.asarray((ctrl.adj > 0).sum(axis=1)).ravel()
    assert np.array_equal(d0, d1), "2-edge swaps must preserve every degree"
    assert ctrl.n_edges == g.n_edges
    assert abs(ctrl.adj.sum() - g.adj.sum()) < 1e-3
    shared = set(zip(*g.adj.nonzero())) & set(zip(*ctrl.adj.nonzero()))
    assert len(shared) < 0.3 * g.adj.nnz, "topology must actually change"


def test_permuted_control_preserves_the_degree_multiset():
    g = _sample_graph()
    ctrl = permuted_graph(g, seed=2)
    d0 = np.asarray((g.adj > 0).sum(axis=1)).ravel()
    d1 = np.asarray((ctrl.adj > 0).sum(axis=1)).ravel()
    assert np.array_equal(np.sort(d0), np.sort(d1))
    assert ctrl.n_edges == g.n_edges


# ---------------------------------------------------------------------------
# Held-out, bootstrap, covariates, time
# ---------------------------------------------------------------------------

def _fitted():
    originals = _install_fake_parser()
    model = BowSCPTM(config=_cfg()).fit(_corpus(), verbose=False)
    return model, originals


def test_encode_documents_uses_the_fitted_vocabulary():
    model, originals = _fitted()
    try:
        bow = model.encode_documents(_corpus()[:3])
        assert bow.shape == (3, model.parsed.n_vocab)
        assert bow.nnz > 0
    finally:
        _restore(originals)


def test_heldout_perplexity_is_finite_and_better_than_uniform():
    model, originals = _fitted()
    try:
        ppl = model.heldout_perplexity(_corpus()[:4])
        assert np.isfinite(ppl) and ppl > 1.0
        assert ppl < model.parsed.n_vocab, (
            f"perplexity {ppl:.2f} should beat the uniform baseline "
            f"{model.parsed.n_vocab}"
        )
    finally:
        _restore(originals)


def test_bootstrap_reports_per_topic_stability():
    model, originals = _fitted()
    try:
        out = model.bootstrap(n_resamples=3, seed=0)
        k = model.config.num_topics
        assert len(out["per_topic_stability"]) == k
        assert 0.0 <= out["topic_stability"] <= 1.0
        assert len(out["prevalence_sd"]) == k
    finally:
        _restore(originals)


def test_estimate_effect_returns_one_row_per_topic_and_term():
    model, originals = _fitted()
    try:
        n = model.theta.shape[0]
        cov = np.linspace(0.0, 1.0, n)
        rows = model.estimate_effect(cov, names=["time"], n_bootstrap=20)
        assert len(rows) == model.config.num_topics * 2      # intercept + time
        for r in rows:
            assert r["ci_low"] <= r["coef"] <= r["ci_high"]
            assert 0.0 <= r["p"] <= 1.0
        # a wrongly sized covariate must fail loudly
        raised = False
        try:
            model.estimate_effect(np.zeros(n + 1), n_bootstrap=2)
        except ValueError:
            raised = True
        assert raised
    finally:
        _restore(originals)


def test_temporal_prevalence_and_drift():
    model, originals = _fitted()
    try:
        n = model.theta.shape[0]
        stamps = np.arange(n)
        prev = model.temporal_prevalence(stamps, n_slices=3)
        assert len(prev) == 3 * model.config.num_topics
        assert all(0.0 <= r["mean"] <= 1.0 for r in prev)
        drift = model.temporal_drift(stamps, n_slices=3)
        assert drift[0]["drift"] == 0.0, "the first slice is the reference"
        assert all(0.0 <= r["drift"] <= 2.0 for r in drift)
    finally:
        _restore(originals)


# ---------------------------------------------------------------------------
# sklearn-style protocol and stats helpers
# ---------------------------------------------------------------------------

def test_get_set_params_roundtrip():
    model = BowSCPTM(num_topics=4, min_df=2)
    assert set(model.get_params()) == {"config"}
    model.set_params(num_topics=7)
    assert model.config.num_topics == 7
    raised = False
    try:
        model.set_params(not_a_field=1)
    except ValueError:
        raised = True
    assert raised


def test_holm_is_monotone_and_bounded():
    adjusted = S.holm([0.001, 0.02, 0.04, 0.5])
    assert np.all(adjusted <= 1.0)
    assert np.all(np.diff(adjusted) >= -1e-12)
    assert adjusted[0] >= 0.001


def test_wilcoxon_warns_when_the_sample_is_too_small():
    result = S.wilcoxon_paired([1.0, 2.0, 3.0], [0.5, 1.5, 2.5])
    assert result.warning is not None and "smallest attainable" in result.warning
    assert 2.0 / 2 ** 3 == 0.25


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:                      # noqa: BLE001
            failures += 1
            print(f"  FAIL  {name}")
            print(f"        {type(exc).__name__}: {exc}")
    print(f"{'OK' if failures == 0 else 'FAILED'} ({failures} failure(s))")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
