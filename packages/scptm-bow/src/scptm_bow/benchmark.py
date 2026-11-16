"""
scptm_bow/benchmark.py
----------------------
The experimental apparatus: a declarative spec in, a reproducible table out.

``benchmark_paper.py`` in the repository root is 82 KB of imperative code that
produces the paper's Table 3 by hand.  This module is the same experiment as a
data structure:

    spec = BenchmarkSpec(
        corpora={"eu": docs_eu, "ng20": docs_ng},
        models=[ModelSpec("nmf-lexical", lambda_syntax=0.0, lambda_embed=0.0),
                ModelSpec("nmf-syntax", lambda_syntax=0.1, lambda_embed=0.0),
                ModelSpec("nmf-matched", lambda_syntax=0.1, lambda_embed=0.0,
                          control="swap")],
        seeds=[0, 1, 2],
    )
    records = run(spec, out="bench.csv")      # resumable
    table = compare(records, metric="c_v")
    print(to_latex(table))

Design rules
------------
* **One row per fit.** Nothing is averaged until :func:`compare`, so the raw
  data can be re-analysed without refitting.
* **Resumable.** Existing ``(corpus, model, seed, k)`` rows are skipped, so an
  interrupted sweep does not restart from zero.
* **Parsing is shared.** Within a seed, every model fits the *same* parsed
  corpus (one cache file), so model differences are not confounded with
  parsing differences.
* **No silent averaging.** ``compare`` returns effect sizes and intervals, and
  applies Holm across the corpus × metric grid.
"""

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from . import stats as _stats
from .metrics import METRIC_KEYS

RESERVED_COLUMNS = ("corpus", "model", "seed", "k", "control", "n_iter", "objective")

CONTROL_MODES = ("none", "swap", "permute")


@dataclass
class ModelSpec:
    """
    One point in the model grid.

    ``control`` replaces the syntactic graph with a statistically matched but
    linguistically empty one, which is what makes the syntax claim testable:

    * ``"none"``   — use the real dependency graph.
    * ``"swap"``   — engine the control: same degree sequence, random topology.
    * ``"permute"``— same graph attached to shuffled words.
    """

    name: str
    divergence: str = "kl"
    lambda_syntax: float = 0.1
    lambda_embed: float = 0.1
    control: str = "none"
    nmf_max_iter: int = 200
    parser_backend: str = "spacy"
    udpipe_model_path: Optional[str] = None
    min_df: int = 5
    max_features: int = 15_000
    extra_stopwords: Optional[Sequence[str]] = None

    def __post_init__(self):
        if self.control not in CONTROL_MODES:
            raise ValueError(f"control must be one of {CONTROL_MODES}, got '{self.control}'")


@dataclass
class BenchmarkSpec:
    """A corpus x model x seed x K grid."""

    corpora: Dict[str, List[str]]
    models: List[ModelSpec]
    seeds: Sequence[int] = (0,)
    topics: Sequence[int] = (10,)
    top_k: int = 10
    lang: str = "eng"

    def __post_init__(self):
        if not self.corpora:
            raise ValueError("no corpora in the spec")
        if not self.models:
            raise ValueError("no models in the spec")
        if not self.seeds:
            raise ValueError("no seeds in the spec")
        if not self.topics:
            raise ValueError("no topic counts in the spec")
        names = [m.name for m in self.models]
        if len(set(names)) != len(names):
            raise ValueError(f"model names must be unique, got {names}")

    def n_runs(self) -> int:
        return len(self.corpora) * len(self.models) * len(self.seeds) * len(self.topics)


def _fit_one(
    documents: List[str],
    model: ModelSpec,
    k: int,
    seed: int,
    lang: str,
    top_k: int,
    cache_path: Optional[Path],
):
    from .config import BowConfig
    from .graphs import permuted_graph, random_graph_matched
    from .model import BowSCPTM

    cfg = BowConfig(
        num_topics=k,
        lang=lang,
        parser_backend=model.parser_backend,
        udpipe_model_path=model.udpipe_model_path,
        min_df=model.min_df,
        max_features=model.max_features,
        divergence=model.divergence,
        lambda_syntax=model.lambda_syntax,
        lambda_embed=model.lambda_embed,
        nmf_max_iter=model.nmf_max_iter,
        random_state=int(seed),
    )
    fitted = BowSCPTM(config=cfg).fit(
        documents, cache_path=str(cache_path) if cache_path else None, verbose=False
    )

    # Control condition: swap the syntactic graph for a matched empty one.
    if model.control != "none" and fitted.syntax_graph is not None:
        if model.control == "swap":
            fitted.syntax_graph = random_graph_matched(fitted.syntax_graph, seed=int(seed))
        else:
            fitted.syntax_graph = permuted_graph(fitted.syntax_graph, seed=int(seed))
        from .factorize import GraphRegularizedNMF

        fitted.result = GraphRegularizedNMF(
            n_components=k,
            lambda_syntax=model.lambda_syntax,
            lambda_embed=model.lambda_embed,
            max_iter=model.nmf_max_iter,
            tol=cfg.nmf_tol,
            random_state=int(seed),
            divergence=cfg.divergence,
        ).fit_transform(fitted.parsed.bow, fitted.syntax_graph, fitted.embed_graph)

    words = fitted.top_words(
        top_k, extra_stopwords=model.extra_stopwords
    )
    scores = fitted.evaluate(top_k=top_k)
    for key in METRIC_KEYS:
        scores.setdefault(key, float("nan"))
    record = {
        "n_iter": int(fitted.result.n_iter),
        "objective": float(fitted.result.objective),
        "control": model.control,
        "n_vocab": int(fitted.parsed.n_vocab),
        "n_docs": int(fitted.parsed.n_docs),
    }
    record.update({key: float(scores[key]) for key in METRIC_KEYS})
    return record


def run(
    spec: BenchmarkSpec,
    out: Union[str, Path] = "benchmark.csv",
    resume: bool = True,
    verbose: bool = True,
    cache_dir: Optional[Union[str, Path]] = None,
) -> List[Dict[str, object]]:
    """
    Execute the grid and append one row per fit to ``out``.

    Returns every row present in the file afterwards (including ones skipped by
    the resume logic), so the caller always gets the full table.
    """
    out_path = Path(out)
    done = set()
    existing: List[Dict[str, object]] = []
    if resume and out_path.is_file():
        with out_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                existing.append(row)
                done.add((row["corpus"], row["model"], int(row["seed"]), int(row["k"])))

    fieldnames = list(RESERVED_COLUMNS) + ["n_vocab", "n_docs"] + list(METRIC_KEYS)
    write_header = not out_path.is_file() or out_path.stat().st_size == 0
    total = spec.n_runs()
    index = 0

    with out_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for corpus_name, documents in spec.corpora.items():
            for seed in spec.seeds:
                cache_path = None
                if cache_dir is not None:
                    cache_path = Path(cache_dir) / f"parse_{corpus_name}.pkl"
                for k in spec.topics:
                    for model in spec.models:
                        index += 1
                        key = (corpus_name, model.name, int(seed), int(k))
                        if key in done:
                            if verbose:
                                print(f"[{index}/{total}] skip (done) {key}")
                            continue
                        if verbose:
                            print(f"[{index}/{total}] {key}")
                        row = _fit_one(
                            documents, model, int(k), int(seed), spec.lang,
                            spec.top_k, cache_path,
                        )
                        row.update({"corpus": corpus_name, "model": model.name,
                                    "seed": int(seed), "k": int(k)})
                        writer.writerow(row)
                        handle.flush()

    with out_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_float(rows: Sequence[Dict[str, object]], key: str) -> np.ndarray:
    return np.asarray([float(r[key]) for r in rows], dtype=np.float64)


def compare(
    records: Sequence[Dict[str, object]],
    metric: str = "c_v",
    treatment: Optional[str] = None,
    baseline: Optional[str] = None,
    group: Sequence[str] = ("corpus", "model"),
    n_resamples: int = 10_000,
    seed: int = 0,
) -> List[Dict[str, object]]:
    """
    Summarise a benchmark table, optionally as a paired comparison.

    Without ``treatment``/``baseline`` returns mean, sd and a bootstrap CI per
    group.  With them, pairs the two models **on the same seed and K** (which is
    what makes the comparison valid) and reports the mean difference, its CI,
    Cliff's delta and — from six seeds upward — a Wilcoxon p-value.

    Holm correction is applied across the whole returned family.
    """
    from .stats import bootstrap_ci, cliffs_delta, paired_bootstrap_ci

    if treatment is None or baseline is None:
        buckets: Dict[tuple, List[float]] = {}
        for row in records:
            value = row.get(metric)
            if value in (None, "", "nan"):
                continue
            value = float(value)
            if not np.isfinite(value):
                continue
            buckets.setdefault(tuple(row.get(g) for g in group), []).append(value)
        out = []
        for key, values in sorted(buckets.items(), key=lambda kv: tuple(map(str, kv[0]))):
            est = bootstrap_ci(values, n_resamples=n_resamples, seed=seed)
            row = dict(zip(group, key))
            row.update({
                "metric": metric, "n": len(values),
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "ci_low": est.low, "ci_high": est.high,
                "p_holm": float("nan"),
            })
            out.append(row)
        return out

    index = {(r["corpus"], r["model"], int(r["seed"]), int(r["k"])): r for r in records}
    pairs: Dict[tuple, List[tuple]] = {}
    for key, row in index.items():
        corpus, model, seed_, k = key
        if model != treatment:
            continue
        other = index.get((corpus, baseline, seed_, k))
        if other is None:
            continue
        try:
            a = float(row[metric]); b = float(other[metric])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(a) and np.isfinite(b):
            pairs.setdefault((corpus, k), []).append((a, b))

    out = []
    for (corpus, k), values in sorted(pairs.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
        a = np.asarray([v[0] for v in values])
        b = np.asarray([v[1] for v in values])
        ci = paired_bootstrap_ci(a, b, n_resamples=n_resamples, seed=seed)
        test = _stats.wilcoxon_paired(a, b)
        out.append({
            "corpus": corpus, "k": k, "metric": metric, "n": len(values),
            "baseline": baseline, "treatment": treatment,
            "mean_baseline": float(b.mean()), "mean_treatment": float(a.mean()),
            "diff": ci.estimate, "ci_low": ci.low, "ci_high": ci.high,
            "cliffs_delta": cliffs_delta(a, b),
            "p_raw": test.p_value, "p_holm": float("nan"),
            "note": test.warning or "",
        })
    if out:
        adjusted = _stats.holm([row["p_raw"] for row in out])
        for row, p in zip(out, adjusted):
            row["p_holm"] = float(p)
    return out


def to_latex(
    rows: Sequence[Dict[str, object]],
    columns: Sequence[str] = ("corpus", "metric", "mean", "ci_low", "ci_high", "n"),
    caption: Optional[str] = None,
    label: Optional[str] = None,
) -> str:
    """Booktabs table, ready to paste into a paper."""
    if not rows:
        return "% empty table\n"
    header = " & ".join(str(c).replace("_", r"\_") for c in columns) + r" \\"
    lines = [r"\begin{table}[htbp]", r"\centering"]
    if caption:
        lines.append(rf"\caption{{{caption}}}")
    if label:
        lines.append(rf"\label{{{label}}}")
    lines += [
        r"\begin{tabular}{" + "l" + "r" * (len(columns) - 1) + "}",
        r"\toprule",
        header,
        r"\midrule",
    ]
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                cells.append("--" if not np.isfinite(value) else f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def save_grid(spec: BenchmarkSpec, path: Union[str, Path]) -> Path:
    """Serialise the spec (without the corpora) for the paper's appendix."""
    payload = {
        "seeds": list(spec.seeds),
        "topics": list(spec.topics),
        "top_k": spec.top_k,
        "lang": spec.lang,
        "corpora": {name: len(docs) for name, docs in spec.corpora.items()},
        "models": [
            {k: v for k, v in model.__dict__.items() if not k.startswith("_")}
            for model in spec.models
        ],
    }
    out = Path(path)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return out
