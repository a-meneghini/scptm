"""
Tests for the declarative benchmark runner.

The point of these tests is that the *apparatus* is correct even when the
numbers are degenerate: the spec validates, the sweep is resumable, the pairing
is by seed, and the table is produced.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

from test_pipeline import _install_fake_parser, _restore  # noqa: E402

from scptm_bow.benchmark import (  # noqa: E402
    BenchmarkSpec,
    ModelSpec,
    compare,
    run,
    save_grid,
    to_latex,
)


def _docs(n=40, seed=0):
    rng = np.random.default_rng(seed)
    pools = [
        "rocket orbit launch satellite mission".split(),
        "hockey puck goal playoff team".split(),
        "faith prayer church god belief".split(),
        "graphics image file format pixel".split(),
    ]
    return [" ".join(rng.choice(pools[i % 4], size=18)) for i in range(n)]


def _spec(seeds=(0, 1)):
    return BenchmarkSpec(
        corpora={"a": _docs(), "b": _docs(seed=1)},
        models=[
            ModelSpec("lexical", lambda_syntax=0.0, lambda_embed=0.0),
            ModelSpec("syntax", lambda_syntax=0.1, lambda_embed=0.0),
        ],
        seeds=list(seeds),
        topics=[4],
        top_k=5,
    )


def test_spec_validates_its_grid():
    spec = _spec()
    assert spec.n_runs() == 2 * 2 * 2 * 1
    for bad in (
        {"models": [ModelSpec("x"), ModelSpec("x")]},
        {"corpora": {}},
        {"seeds": ()},
        {"topics": ()},
    ):
        kwargs = dict(corpora=spec.corpora, models=spec.models,
                      seeds=spec.seeds, topics=spec.topics)
        kwargs.update(bad)
        raised = False
        try:
            BenchmarkSpec(**kwargs)
        except ValueError:
            raised = True
        assert raised, f"spec with {list(bad)} must be rejected"


def test_model_spec_rejects_unknown_control():
    raised = False
    try:
        ModelSpec("x", control="bogus")
    except ValueError:
        raised = True
    assert raised


def test_run_is_resumable_and_emits_one_row_per_fit():
    originals = _install_fake_parser()
    try:
        spec = _spec()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "bench.csv"
            first = run(spec, out=out, verbose=False)
            assert len(first) == spec.n_runs()
            second = run(spec, out=out, verbose=False)
            assert len(second) == len(first), "resume must not duplicate rows"
            assert {r["model"] for r in second} == {"lexical", "syntax"}
            assert {"c_v", "npmi", "diversity"} <= set(second[0])
    finally:
        _restore(originals)


def test_compare_pairs_on_seed_and_applies_holm():
    originals = _install_fake_parser()
    try:
        spec = _spec(seeds=(0, 1, 2))
        with tempfile.TemporaryDirectory() as tmp:
            rows = run(spec, out=Path(tmp) / "b.csv", verbose=False)
        summary = compare(rows, metric="c_v")
        assert {r["model"] for r in summary} == {"lexical", "syntax"}
        paired = compare(rows, metric="c_v", treatment="syntax", baseline="lexical")
        assert len(paired) == 2, "one row per corpus"
        for row in paired:
            assert row["ci_low"] <= row["diff"] <= row["ci_high"]
            assert -1.0 <= row["cliffs_delta"] <= 1.0
            assert 0.0 <= row["p_holm"] <= 1.0
            assert "smallest attainable" in row["note"], "n=3 must warn"
    finally:
        _restore(originals)


def test_to_latex_and_save_grid():
    rows = [{"corpus": "a", "metric": "c_v", "mean": 0.5, "ci_low": 0.4,
             "ci_high": 0.6, "n": 3}]
    tex = to_latex(rows, caption="c", label="l")
    assert "\\toprule" in tex and "\\bottomrule" in tex and "0.5000" in tex
    assert to_latex([]).startswith("% empty")

    with tempfile.TemporaryDirectory() as tmp:
        path = save_grid(_spec(), Path(tmp) / "grid.json")
        text = path.read_text()
        assert "\"seeds\"" in text and "\"models\"" in text


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:                      # noqa: BLE001
            failures += 1
            print(f"  FAIL  {name}\n        {type(exc).__name__}: {exc}")
    print(f"{'OK' if failures == 0 else 'FAILED'} ({failures} failure(s))")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
