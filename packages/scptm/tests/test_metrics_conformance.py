"""
Conformance test: ``scptm.metrics`` and ``scptm.stats`` must reproduce the
shared fixture bit-for-bit, so results from scptm and scptm-bow are comparable.

Run:  .venv/bin/python tests/test_metrics_conformance.py
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


sys.path.insert(0, str(HERE.parent / "src"))

import scptm.metrics as M  # noqa: E402
import scptm.stats as S  # noqa: E402

FIXTURE = _find_conformance()


def _payload():
    if not FIXTURE.is_file():
        return None
    return json.loads(FIXTURE.read_text())


def _bow(payload):
    rows = [r for r, _, _ in payload["bow_nnz"]]
    cols = [c for _, c, _ in payload["bow_nnz"]]
    vals = [v for _, _, v in payload["bow_nnz"]]
    return sparse.csr_matrix(
        (np.asarray(vals), (np.asarray(rows), np.asarray(cols))),
        shape=tuple(payload["bow_shape"]),
    )


def test_metrics_match_the_fixture():
    payload = _payload()
    if payload is None:
        print("    (fixture assente, salto)")
        return
    bow = _bow(payload)
    got = {
        "c_v": M.c_v_coherence(payload["topics"], tokenized=payload["tokens"], window=5),
        "npmi": M.npmi_coherence(payload["topics"], bow, payload["vocab"]),
        "diversity": M.topic_diversity(payload["topics"]),
        "redundancy": M.redundancy(payload["topics"]),
    }
    for key, expected in payload["metrics"].items():
        assert abs(got[key] - expected) < 1e-10, (key, got[key], expected)


def test_stats_match_the_fixture():
    payload = _payload()
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


def test_key_sets_match_the_bow_package():
    assert M.METRIC_KEYS == ("c_v", "npmi", "diversity", "exclusivity", "redundancy")


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
