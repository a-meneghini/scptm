"""
tests/test_pipeline.py
======================
Factorisation and end-to-end pipeline tests.

The parser is replaced by a deterministic fake, so the whole pipeline runs in
milliseconds with no spaCy/UDPipe model and no network.

Run with pytest::

    pytest tests/test_pipeline.py -v

or directly::

    python tests/test_pipeline.py
"""

import pickle
import shutil
import tempfile
import traceback
import warnings
from pathlib import Path

import numpy as np

import scptm_bow.model as model_mod
import scptm_bow.parse as parse_mod
from scptm_bow import SAVE_FORMAT_VERSION, BowConfig, BowSCPTM
from scptm_bow.factorize import GraphRegularizedNMF
from scptm_bow.graphs import syntax_graph
from scptm_bow.parse import Tok

# Four well-separated word pools, one topic each.
POOLS = [
    "rocket orbit launch satellite mission astronaut nasa".split(),
    "hockey puck goal playoff team player score".split(),
    "faith prayer church god belief scripture sacred".split(),
    "graphics render pixel shader texture polygon image".split(),
]


def _corpus(n_per_pool: int = 12, rng=None) -> list:
    rng = rng or np.random.default_rng(0)
    docs = []
    for _ in range(n_per_pool):
        for pool in POOLS:
            docs.append(" ".join(rng.choice(pool, size=25)))
    return docs


class _FakeParser:
    """Whitespace lemmatiser with a fixed dependency structure."""

    name = "fake"

    def pipe(self, texts, batch_size: int = 64):
        for text in texts:
            words = [w for w in text.split() if w.isalpha()]
            tokens = []
            for i, w in enumerate(words):
                tokens.append(
                    Tok(
                        lemma=w.lower(),
                        dep="nmod" if i else "root",
                        head_lemma=words[0].lower() if i else "",
                        is_stop=False,
                        is_alpha=True,
                    )
                )
            yield tokens


def _install_fake_parser():
    original_parse = parse_mod.build_parser
    original_model = model_mod.build_parser
    parse_mod.build_parser = lambda cfg: _FakeParser()
    model_mod.build_parser = lambda cfg: _FakeParser()
    return original_parse, original_model


def _restore(originals):
    parse_mod.build_parser, model_mod.build_parser = originals


def _cfg(**over):
    base = dict(num_topics=4, min_df=2, max_features=200, nmf_max_iter=80,
                vector_backend="ppmi_svd", vector_dim=20, random_state=0)
    base.update(over)
    return BowConfig(**base)


# ---------------------------------------------------------------------------
# NMF
# ---------------------------------------------------------------------------

def test_plain_nmf_recovers_block_structure():
    rng = np.random.default_rng(0)
    n_docs, n_blocks = 60, 4
    blocks = np.zeros((n_docs, n_blocks), dtype=np.float32)
    for i in range(n_docs):
        blocks[i, i % n_blocks] = 1.0
    X = blocks @ rng.random((n_blocks, 30)).astype(np.float32) + 0.01

    from scipy import sparse

    res = GraphRegularizedNMF(n_components=n_blocks, random_state=0).fit_transform(
        sparse.csr_matrix(X)
    )
    assert res.W.shape == (n_docs, n_blocks)
    assert res.H.shape == (n_blocks, 30)
    assert np.allclose(res.H.sum(axis=1), 1.0, atol=1e-6), "H rows must sum to 1"

    # NMF factor labels are arbitrary, so test the *partition* it induces rather
    # than the labels: a block must land on one factor, and blocks must differ.
    dominant = res.W.argmax(axis=1)
    per_block = [set(dominant[b::n_blocks].tolist()) for b in range(n_blocks)]
    assert all(len(s) == 1 for s in per_block), f"a block was split: {per_block}"
    assert len({next(iter(s)) for s in per_block}) == n_blocks, (
        f"blocks collapsed onto the same factor: {per_block}"
    )


def test_graph_regulariser_changes_the_solution():
    from scipy import sparse

    rng = np.random.default_rng(1)
    X = sparse.csr_matrix(rng.random((40, 20)).astype(np.float32))
    src = np.arange(19, dtype=np.int32)
    dst = src + 1
    weight = np.ones(19, dtype=np.float32)
    graph = syntax_graph(src, dst, weight, n_vocab=20, top_k=3)

    plain = GraphRegularizedNMF(n_components=3, lambda_syntax=0.0, random_state=0).fit_transform(X)
    reg = GraphRegularizedNMF(n_components=3, lambda_syntax=5.0, random_state=0).fit_transform(
        X, syntax_graph=graph
    )
    assert not np.allclose(plain.H, reg.H), "the Laplacian term must affect H"
    # Nearby words (adjacent indices) should be more similar in topic space.
    def neighbour_similarity(H):
        Hn = H / (np.linalg.norm(H, axis=0, keepdims=True) + 1e-10)
        return float(np.mean([np.dot(Hn[:, i], Hn[:, i + 1]) for i in range(19)]))
    assert neighbour_similarity(reg.H) > neighbour_similarity(plain.H)


# ---------------------------------------------------------------------------
# Divergence (KL is the default; Frobenius kept for comparison)
# ---------------------------------------------------------------------------

def _toy_counts(n=30, v=12, k=3, seed=0):
    from scipy import sparse

    rng = np.random.default_rng(seed)
    base = np.zeros((n, k), dtype=np.float64)
    for i in range(n):
        base[i, i % k] = 1.0
    return sparse.csr_matrix((base @ rng.random((k, v)) * 10).round() + 0.0)


def test_kl_objective_matches_brute_force():
    from scptm_bow.factorize import GraphRegularizedNMF as NMF
    from scptm_bow.graphs import empty_graph

    X = _toy_counts()
    rng = np.random.default_rng(3)
    W = rng.random((X.shape[0], 3)) + 0.1
    H = rng.random((3, X.shape[1])) + 0.1

    model = NMF(n_components=3, divergence="kl")
    rows, cols = X.nonzero()
    x_data = np.asarray(X[rows, cols]).ravel().astype(np.float64)
    got = model._objective(X, W, H, [empty_graph(X.shape[1])] * 2, [0.0, 0.0],
                           rows, cols, x_data)

    dense = np.asarray(X.todense())
    wh = W @ H
    reference = float(
        np.sum(dense * (np.log(dense + 1e-300) - np.log(wh)))
        - dense.sum() + wh.sum()
    )
    assert abs(got - reference) < 1e-6, (got, reference)


def test_kl_objective_decreases_monotonically():
    X = _toy_counts()
    result = GraphRegularizedNMF(n_components=3, max_iter=60, tol=0.0,
                                 divergence="kl").fit_transform(X)
    history = result.history
    assert len(history) > 5
    # Strict monotonicity: this is what catches an inconsistent multiplicative
    # update (reusing X/(W H) after H changed makes the factors oscillate in
    # scale and the objective tick back up).
    assert all(b <= a + 1e-9 for a, b in zip(history, history[1:])), (
        "multiplicative updates must not increase the KL objective"
    )


def test_frobenius_still_available_and_differs_from_kl():
    X = _toy_counts()
    kl = GraphRegularizedNMF(n_components=3, random_state=0, divergence="kl").fit_transform(X)
    eu = GraphRegularizedNMF(n_components=3, random_state=0, divergence="frobenius").fit_transform(X)
    assert kl.divergence == "kl" and eu.divergence == "frobenius"
    assert not np.allclose(kl.H, eu.H)
    for res in (kl, eu):
        assert np.allclose(res.H.sum(axis=1), 1.0, atol=1e-9)
        assert np.all(res.H >= 0)


def test_nmf_rejects_unknown_divergence():
    raised = False
    try:
        GraphRegularizedNMF(n_components=3, divergence="bogus")
    except ValueError:
        raised = True
    assert raised


# ---------------------------------------------------------------------------
# Stop-word filtering in the topic representation
# ---------------------------------------------------------------------------

def test_top_words_can_hide_stopwords_without_touching_the_model():
    originals = _install_fake_parser()
    try:
        docs = _corpus()
        model = BowSCPTM(config=_cfg()).fit(docs, verbose=False)
        n_vocab = model.parsed.n_vocab
        H_before = model.topic_word.copy()

        plain = model.top_words(10, drop_stopwords=False)
        filtered = model.top_words(10, extra_stopwords=["rocket", "hockey"])

        assert "rocket" in plain[0] or any("rocket" in w for w in plain), plain
        assert all("rocket" not in w and "hockey" not in w for w in filtered)
        # Same length where enough words remain, and the model is untouched.
        assert len(filtered[0]) == 10
        assert model.parsed.n_vocab == n_vocab
        assert np.array_equal(model.topic_word, H_before)
    finally:
        _restore(originals)


def test_topic_stopwords_from_config_and_common_words():
    originals = _install_fake_parser()
    try:
        docs = _corpus()
        model = BowSCPTM(config=_cfg()).fit(docs, verbose=False)

        common = model.common_words(top_n=5)
        assert len(common) == 5
        assert all({"word", "doc_freq", "share"} <= set(row) for row in common)
        assert common[0]["doc_freq"] >= common[-1]["doc_freq"]

        blocked = common[0]["word"]
        words = model.top_words(10, extra_stopwords=[blocked])
        assert all(blocked not in row for row in words)

        # The same list can be set on the config and is then used everywhere.
        cfg2 = _cfg(topic_stopwords=[blocked])
        model2 = BowSCPTM(config=cfg2).fit(docs, verbose=False)
        assert all(blocked not in row for row in model2.top_words(10))
        assert all(blocked not in row["top_words"] for row in model2.get_topic_info(10))
    finally:
        _restore(originals)


# ---------------------------------------------------------------------------
# 2D semantic-space figure
# ---------------------------------------------------------------------------

def test_plot_semantic_2d_writes_a_png():
    originals = _install_fake_parser()
    try:
        docs = _corpus()
        model = BowSCPTM(config=_cfg()).fit(docs, verbose=False)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sem.png"
            path = model.plot_semantic_2d(
                top_k=5, save_path=out,
                reducer=lambda X: X[:, :2],       # skip UMAP in the test
            )
            assert Path(path).is_file() and Path(path).stat().st_size > 1000

            # visualize_2d is the neural package's name: same thing, same output.
            alias = model.visualize_2d(
                top_k=5, save_path=Path(tmp) / "sem2.png",
                reducer=lambda X: X[:, :2],
            )
            assert Path(alias).is_file()
    finally:
        _restore(originals)


def test_plot_requires_word_vectors():
    originals = _install_fake_parser()
    try:
        model = BowSCPTM(config=_cfg(vector_backend="none")).fit(_corpus(), verbose=False)
        raised = False
        try:
            model.plot_semantic_2d(save_path="/tmp/should_not_exist.png",
                                   reducer=lambda X: X[:, :2])
        except ValueError:
            raised = True
        assert raised, "a model without word vectors cannot be projected"
    finally:
        _restore(originals)


def test_plot_labels_do_not_overlap_and_are_unique():
    """The whole point of the label layout: no piled-up text, no duplicates."""
    import matplotlib.figure as mfigure

    originals = _install_fake_parser()
    try:
        model = BowSCPTM(config=_cfg(num_topics=5)).fit(_corpus(), verbose=False)
        captured = {}
        original_savefig = mfigure.Figure.savefig

        def _capture(self, *args, **kwargs):
            captured["fig"] = self
            return original_savefig(self, *args, **kwargs)

        mfigure.Figure.savefig = _capture
        try:
            with tempfile.TemporaryDirectory() as tmp:
                model.plot_semantic_2d(
                    top_k=6, save_path=Path(tmp) / "sem.png", dpi=100,
                    figsize=(8.0, 6.0), reducer=lambda X: X[:, :2],
                )
        finally:
            mfigure.Figure.savefig = original_savefig

        fig = captured["fig"]
        ax = fig.axes[0]
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        texts = [t for t in ax.texts if t.get_text()]
        assert len(texts) > 0
        labels = [t.get_text() for t in texts]
        assert len(labels) == len(set(labels)), f"duplicate labels drawn: {labels}"

        topic_labels = [lab for lab in labels if lab.startswith("Topic_")]
        assert len(topic_labels) == 5, "every topic must keep its label"

        boxes = [t.get_window_extent(renderer) for t in texts]
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                assert not boxes[i].overlaps(boxes[j]), (
                    f"labels {labels[i]!r} and {labels[j]!r} overlap"
                )
    finally:
        _restore(originals)


# ---------------------------------------------------------------------------
# K selection
# ---------------------------------------------------------------------------

def test_search_k_returns_the_best_finite_model():
    originals = _install_fake_parser()
    try:
        docs = _corpus(n_per_pool=10)
        best_k, model, results = BowSCPTM.search_k(
            docs, k_grid=(3, 4), finetune=False, verbose=False,
            vector_backend="ppmi_svd", vector_dim=20, min_df=2,
            max_features=200, nmf_max_iter=60, random_state=0,
        )
        assert set(results) == {3, 4}
        assert model is results[best_k]["model"]
        assert model.config.num_topics == best_k
        assert model._is_fitted
        for K, record in results.items():
            assert record["model"].config.num_topics == K
            assert np.isfinite(record["score"])
        finite = {K: r["score"] for K, r in results.items() if np.isfinite(r["score"])}
        assert results[best_k]["score"] == max(finite.values())
    finally:
        _restore(originals)


def test_search_k_finetune_evaluates_between_grid_points():
    originals = _install_fake_parser()
    try:
        docs = _corpus(n_per_pool=8)
        _, _, results = BowSCPTM.search_k(
            docs, k_grid=(3, 5), finetune=True, verbose=False,
            vector_backend="ppmi_svd", vector_dim=20, min_df=2,
            max_features=200, nmf_max_iter=40, random_state=0,
        )
        assert 3 in results and 5 in results
        assert 4 in results, f"finetune must fill the gap: {sorted(results)}"
    finally:
        _restore(originals)


def test_best_finite_k_ignores_nan_and_rejects_all_nan():
    from scptm_bow.selection import _best_finite_k

    assert _best_finite_k({
        3: {"score": float("nan")},
        4: {"score": 0.2},
        5: {"score": float("inf")},
    }) == 4

    raised = False
    try:
        _best_finite_k({3: {"score": float("nan")}, 4: {"score": float("-inf")}})
    except RuntimeError:
        raised = True
    assert raised, "an all-NaN sweep must raise RuntimeError"


def test_nmf_is_deterministic():
    from scipy import sparse

    rng = np.random.default_rng(2)
    X = sparse.csr_matrix(rng.random((30, 15)).astype(np.float32))
    a = GraphRegularizedNMF(n_components=3, random_state=7).fit_transform(X)
    b = GraphRegularizedNMF(n_components=3, random_state=7).fit_transform(X)
    assert np.allclose(a.H, b.H)


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------

def test_pipeline_end_to_end_and_transform():
    originals = _install_fake_parser()
    try:
        docs = _corpus()
        model = BowSCPTM(config=_cfg()).fit(docs, verbose=False)

        assert model.theta.shape == (len(docs), 4)
        assert np.allclose(model.theta.sum(axis=1), 1.0, atol=1e-6)

        top = model.top_words(top_k=5)
        assert len(top) == 4
        # Each topic should be dominated by one word pool.
        pools = {frozenset(p) for p in POOLS}
        hits = 0
        for words in top:
            if frozenset(words[:5]) <= max(pools, key=lambda p: len(p & set(words[:5]))):
                hits += 1
        assert hits == 4, f"topics did not separate cleanly: {top}"

        # Out-of-sample fold-in should land on the matching topic.
        new = [" ".join(POOLS[1][:8]) for _ in range(3)]
        theta_new = model.transform(new)
        assert theta_new.shape == (3, 4)
        assert len(set(theta_new.argmax(axis=1).tolist())) == 1

        info = model.get_topic_info(top_k=5)
        assert len(info) == 4 and {"topic_id", "size", "top_words", "mwes"} <= set(info[0])
        assert sum(r["size"] for r in info) == len(docs)

        evaluated = model.evaluate(top_k=5)
        assert evaluated["diversity"] > 0.9
        assert 0.0 <= evaluated["redundancy"] <= 1.0
    finally:
        _restore(originals)


def test_pipeline_is_deterministic():
    originals = _install_fake_parser()
    try:
        docs = _corpus()
        a = BowSCPTM(config=_cfg()).fit(docs, verbose=False)
        b = BowSCPTM(config=_cfg()).fit(docs, verbose=False)
        assert np.abs(a.theta - b.theta).max() < 1e-12
    finally:
        _restore(originals)


def test_pipeline_cache_reuse_and_invalidation():
    originals = _install_fake_parser()
    try:
        docs = _corpus()
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "c.pkl")

            first = BowSCPTM(config=_cfg()).fit(docs, cache_path=path, verbose=False)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                second = BowSCPTM(config=_cfg()).fit(docs, cache_path=path, verbose=False)
            assert not [w for w in caught if "invalidated" in str(w.message)]
            assert second.parsed.n_vocab == first.parsed.n_vocab
            assert np.allclose(second.parsed.bow.toarray(), first.parsed.bow.toarray())

            # A vocabulary setting change must invalidate it, with a reason.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                BowSCPTM(config=_cfg(min_df=5)).fit(docs, cache_path=path, verbose=False)
            assert any("min_df" in str(w.message) for w in caught)
    finally:
        _restore(originals)


def test_fit_rejects_empty_corpus_and_kwargs_conflict():
    raised = False
    try:
        BowSCPTM(config=_cfg(), num_topics=3)
    except TypeError:
        raised = True
    assert raised, "passing config and kwargs together must raise TypeError"

    originals = _install_fake_parser()
    try:
        raised = False
        try:
            BowSCPTM(config=_cfg()).fit([], verbose=False)
        except ValueError:
            raised = True
        assert raised, "an empty corpus must raise ValueError"
    finally:
        _restore(originals)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_save_load_roundtrip():
    originals = _install_fake_parser()
    try:
        model = BowSCPTM(config=_cfg()).fit(_corpus(n_per_pool=8), verbose=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pkl"
            assert model.save(path) == path
            assert path.exists()
            assert not [p for p in Path(tmp).iterdir() if p.name.startswith(".")], \
                "the atomic write must leave no temp file behind"

            loaded = BowSCPTM.load(path)
            assert loaded._is_fitted
            assert loaded.config.num_topics == model.config.num_topics
            assert list(loaded.vocab) == list(model.vocab)
            assert np.allclose(loaded.theta, model.theta)
            assert np.allclose(loaded.topic_word, model.topic_word)
            assert loaded.get_topic_info(top_k=5) == model.get_topic_info(top_k=5)

            a, b = model.evaluate(), loaded.evaluate()
            assert a.keys() == b.keys()
            assert all(abs(a[k] - b[k]) < 1e-12 for k in a)

            new = ["rocket orbit launch mission rocket", "prayer church faith"]
            assert np.allclose(loaded.transform(new), model.transform(new))

            # The corpus text is not stored unless asked for.
            assert loaded.parsed.lemma_docs == []
            assert loaded.parsed.n_docs == model.parsed.n_docs
    finally:
        _restore(originals)


def test_save_can_include_the_corpus():
    originals = _install_fake_parser()
    try:
        model = BowSCPTM(config=_cfg()).fit(_corpus(n_per_pool=6), verbose=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "full.pkl"
            model.save(path, include_corpus=True)
            loaded = BowSCPTM.load(path)
            assert loaded.parsed.lemma_docs == model.parsed.lemma_docs
            assert len(loaded.parsed.lemma_docs) == model.parsed.n_docs
    finally:
        _restore(originals)


def test_save_requires_a_fitted_model():
    raised = False
    try:
        BowSCPTM(config=_cfg()).save("never_written.pkl")
    except RuntimeError:
        raised = True
    assert raised, "saving an unfitted model must raise RuntimeError"
    assert not Path("never_written.pkl").exists()


def test_load_does_not_need_the_parser():
    originals = _install_fake_parser()
    try:
        model = BowSCPTM(config=_cfg()).fit(_corpus(n_per_pool=6), verbose=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pkl"
            model.save(path)
            _restore(originals)

            def _boom(cfg):
                raise AssertionError("load() must not build a parser")

            parse_mod.build_parser = _boom
            model_mod.build_parser = _boom
            try:
                loaded = BowSCPTM.load(path)
                assert loaded.theta.shape[0] == loaded.parsed.n_docs
                assert loaded.top_words(3)
            finally:
                parse_mod.build_parser = originals[0]
                model_mod.build_parser = originals[1]
    finally:
        _restore(originals)


def test_load_rejects_foreign_corrupt_and_future_files():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad.pkl"

        path.write_bytes(b"definitely not a pickle")
        raised = False
        try:
            BowSCPTM.load(path)
        except ValueError:
            raised = True
        assert raised, "a non-pickle file must raise ValueError"

        with open(path, "wb") as fh:
            pickle.dump({"format": SAVE_FORMAT_VERSION + 1,
                         "kind": "scptm_bow.BowSCPTM",
                         "package_version": "99.0.0"}, fh)
        raised = False
        try:
            BowSCPTM.load(path)
        except ValueError as exc:
            raised = "upgrade" in str(exc).lower() or "newer" in str(exc).lower()
        assert raised, "a newer format must raise an actionable ValueError"

        with open(path, "wb") as fh:
            pickle.dump({"format": SAVE_FORMAT_VERSION, "kind": "something.else"}, fh)
        raised = False
        try:
            BowSCPTM.load(path)
        except ValueError:
            raised = True
        assert raised, "a different kind must raise ValueError"

        with open(path, "wb") as fh:
            pickle.dump({"format": SAVE_FORMAT_VERSION, "kind": "scptm_bow.BowSCPTM"}, fh)
        raised = False
        try:
            BowSCPTM.load(path)
        except ValueError as exc:
            raised = "missing" in str(exc).lower()
        assert raised, "an incomplete state must name the missing keys"

        try:
            BowSCPTM.load(Path(tmp) / "missing.pkl")
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("a missing path must raise FileNotFoundError")


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
