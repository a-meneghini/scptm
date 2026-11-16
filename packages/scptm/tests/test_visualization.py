"""
tests/test_visualization.py
===========================
Regression tests for the 2D semantic-space figure:

1. Labels must never overlap (the old fixed-offset layout stacked them on top of
   each other), and each label is drawn at most once.
2. Every topic keeps its label, even in a crowded figure.
3. ``adjust_labels=False`` still renders the legacy layout.

Hermetic: the UMAP projection is replaced by a cheap deterministic stand-in, so
these run in well under a second and need no model, no network and no numba
compilation.

Run with pytest::

    pytest tests/test_visualization.py -v

or directly (no pytest required)::

    python tests/test_visualization.py
"""

import tempfile
import traceback
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)      # must precede the pyplot import below

import matplotlib.figure as mfigure    # noqa: E402
import numpy as np                     # noqa: E402
import torch                           # noqa: E402
from matplotlib.text import Text       # noqa: E402

from scptm import visualization as viz  # noqa: E402


def _fake_setup(n_topics=12, n_words=60, dim=8, n_mwes=8, seed=0):
    """Fake model + keyword tables shaped like ``extract_separated_topics``."""
    rng = np.random.default_rng(seed)
    words = [f"w{i:02d}" for i in range(n_words)]
    phrases = [f"w{i:02d} w{(i + 3) % n_words:02d}" for i in range(n_mwes)]

    word_embs = torch.tensor(rng.normal(size=(n_words, dim)), dtype=torch.float32)
    mwe_embs = torch.tensor(rng.normal(size=(n_mwes, dim)), dtype=torch.float32)
    topic_embs = torch.tensor(rng.normal(size=(n_topics, dim)), dtype=torch.float32)
    model = types.SimpleNamespace(topic_embeddings=topic_embs)

    topics = {}
    for k in range(n_topics):
        topics[f"Topic_{k + 1}"] = {
            "single": [words[(k * 4 + j) % n_words] for j in range(5)],
            "phrases": [phrases[(k + j) % n_mwes] for j in range(2)],
        }
    return model, word_embs, words, mwe_embs, phrases, topics


def _fake_projection(topic_vecs, word_vecs, n_components=2, **_kwargs):
    """Deterministic stand-in for UMAP: first two vector coordinates."""
    return np.vstack([topic_vecs, word_vecs])[:, :n_components]


def _render(tmp_dir, **kwargs):
    """Run the figure, capturing it before it is closed/shown."""
    model, word_embs, words, mwe_embs, phrases, topics = _fake_setup()
    captured = {}
    original_savefig = mfigure.Figure.savefig

    def _capture(self, *args, **kw):
        captured["fig"] = self
        return original_savefig(self, *args, **kw)

    original_projection = viz._build_umap_projection
    viz._build_umap_projection = _fake_projection
    mfigure.Figure.savefig = _capture
    try:
        viz.view_semantic_2d_paper(
            model, word_embs, words, mwe_embs, phrases, topics,
            save_path=str(Path(tmp_dir) / "sem.png"),
            dpi=100, figsize=(8.0, 6.0), **kwargs,
        )
    finally:
        viz._build_umap_projection = original_projection
        mfigure.Figure.savefig = original_savefig
    assert (Path(tmp_dir) / "sem.png").is_file()
    return captured["fig"], topics


def _texts(fig):
    ax = fig.axes[0]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [t for t in ax.texts if t.get_text()]
    boxes = [Text.get_window_extent(t, renderer) for t in texts]
    return [t.get_text() for t in texts], boxes


def test_labels_do_not_overlap_and_are_unique():
    with tempfile.TemporaryDirectory() as tmp:
        fig, topics = _render(tmp)
        labels, boxes = _texts(fig)

    assert len(labels) == len(set(labels)), (
        f"duplicate labels drawn: {[l for l in labels if labels.count(l) > 1]}"
    )
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            assert not boxes[i].overlaps(boxes[j]), (
                f"labels {labels[i]!r} and {labels[j]!r} overlap"
            )
    for name in topics:
        assert name in labels, f"missing topic label {name}"


def test_legacy_layout_still_renders():
    with tempfile.TemporaryDirectory() as tmp:
        fig, _ = _render(tmp, adjust_labels=False)
        labels, boxes = _texts(fig)
    assert labels, "the legacy layout must still draw labels"


def test_unknown_words_are_skipped():
    with tempfile.TemporaryDirectory() as tmp:
        model, word_embs, words, mwe_embs, phrases, topics = _fake_setup()
        topics["Topic_1"]["single"].append("not_in_the_vocabulary")
        captured = {}
        original_savefig = mfigure.Figure.savefig

        def _capture(self, *args, **kw):
            captured["fig"] = self
            return original_savefig(self, *args, **kw)

        original_projection = viz._build_umap_projection
        viz._build_umap_projection = _fake_projection
        mfigure.Figure.savefig = _capture
        try:
            viz.view_semantic_2d_paper(
                model, word_embs, words, mwe_embs, phrases, topics,
                save_path=str(Path(tmp) / "sem.png"), dpi=100, figsize=(8.0, 6.0),
            )
        finally:
            viz._build_umap_projection = original_projection
            mfigure.Figure.savefig = original_savefig

        labels, _ = _texts(captured["fig"])
        assert "not_in_the_vocabulary" not in labels


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
