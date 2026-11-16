# -*- coding: utf-8 -*-
"""
exp_c3b_pmi_comparison.py
──────────────────────────
C3B: Compare the VADER valence of SCPTM's syntactic MWEs against
     the top-N bigrams selected by Pointwise Mutual Information (PMI)
     from the same corpus.

If SCPTM's topic-level syntactic MWEs carry higher average valence than
the highest-PMI bigrams in the corpus, that is evidence that syntactic
selection adds evaluative signal beyond statistical co-occurrence.

Design:
  • For each corpus:
      – load raw documents (same sources as benchmark_paper.py)
      – compute top-200 bigrams by PMI (using unigram/bigram counts)
      – score PMI bigrams with VADER
      – load SCPTM MWE phrases from words_*.csv (best-K, all seeds)
      – score SCPTM MWEs with VADER
      – compare mean VADER compound scores

Dependencies:
    pip install vaderSentiment scikit-learn datasets

    For 20NG  : pip install scikit-learn  (fetch_20newsgroups)
    For reddit: reddit_pol.csv must be at REDDIT_CSV path (see below)
    For EU/UN : pip install datasets  (HuggingFace, internet required)
    For hate  : pip install datasets  (HuggingFace, internet required)

Output:
    benchmark_cache/results_0605/c3b_pmi_comparison.csv
    printed comparison table
"""

import csv
import math
import re
import sys
from pathlib import Path
from collections import Counter, defaultdict

BASE       = Path(__file__).parent
WORDS_DIR  = BASE / "benchmark_cache" / "results_0605"
MAIN_TABLE = WORDS_DIR / "main_table.csv"
OUT_CSV    = WORDS_DIR / "c3b_pmi_comparison.csv"

# Corpus CSV paths (local files)
REDDIT_CSV = BASE / "reddit_pol.csv"
UNGDC_CSV  = BASE / "un-general-debates.csv"

TOP_N_PMI  = 200   # how many top-PMI bigrams to consider
MODEL      = "SCPTM"

# ── VADER ────────────────────────────────────────────────────────────────────

def _vader():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    if not hasattr(_vader, "_inst"):
        _vader._inst = SentimentIntensityAnalyzer()
    return _vader._inst

def vader_score(text: str) -> float:
    return _vader().polarity_scores(text)["compound"]

def mean_vader(phrases):
    scores = [vader_score(p) for p in phrases if p and p.strip()]
    return sum(scores) / len(scores) if scores else float("nan")

# ── Corpus loaders ────────────────────────────────────────────────────────────

def load_20ng():
    from sklearn.datasets import fetch_20newsgroups
    data = fetch_20newsgroups(subset="all",
                              remove=("headers", "footers", "quotes"))
    return data.data

def load_reddit():
    import csv as _csv
    docs = []
    with open(REDDIT_CSV, newline="", encoding="utf-8", errors="replace") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            text = ""
            for key in row:
                if key.strip().lower() in ("text", "body", "selftext", "content"):
                    text = row[key]
                    break
            if text.strip():
                docs.append(text.strip())
    return docs

def load_eu_debates():
    from datasets import load_dataset
    ds = load_dataset("coastalcph/eu_debates", split="train")
    return [row["text"] for row in ds if row.get("text")]

def load_un_debates():
    import csv as _csv
    docs = []
    with open(UNGDC_CSV, newline="", encoding="utf-8", errors="replace") as f:
        for row in _csv.DictReader(f):
            text = row.get("text") or row.get("speech") or ""
            if text.strip():
                docs.append(text.strip())
    return docs

def load_hatespeech():
    from datasets import load_dataset
    ds = load_dataset("ucberkeley-dlab/measuring-hate-speech", split="train")
    seen, docs = set(), []
    for row in ds:
        t = row.get("text", "")
        if t and t not in seen:
            seen.add(t)
            docs.append(t)
    return docs

LOADERS = {
    "20NG":       load_20ng,
    "Reddit_Pol": load_reddit,
    "UN_Debates": load_un_debates,
    "HateSpeech": load_hatespeech,
}

# ── PMI bigrams ───────────────────────────────────────────────────────────────

_TOKENIZE = re.compile(r"[a-z]{2,}")

def tokenize(text: str):
    return _TOKENIZE.findall(text.lower())

def top_pmi_bigrams(docs, n=200, min_count=10):
    """Return the top-n bigrams by Pointwise Mutual Information."""
    uni = Counter()
    bi  = Counter()
    for doc in docs:
        toks = tokenize(doc)
        for w in toks:
            uni[w] += 1
        for w1, w2 in zip(toks, toks[1:]):
            bi[(w1, w2)] += 1

    total_uni = sum(uni.values())
    total_bi  = sum(bi.values())

    pmi = {}
    for (w1, w2), c_bi in bi.items():
        if c_bi < min_count:
            continue
        p12 = c_bi / total_bi
        p1  = uni[w1] / total_uni
        p2  = uni[w2] / total_uni
        if p1 > 0 and p2 > 0:
            pmi[(w1, w2)] = math.log(p12 / (p1 * p2))

    top = sorted(pmi.items(), key=lambda x: x[1], reverse=True)[:n]
    return [f"{w1} {w2}" for (w1, w2), _ in top]

# ── words CSV ─────────────────────────────────────────────────────────────────

def load_best_k(corpus: str, model: str) -> int:
    with open(MAIN_TABLE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["corpus"] == corpus and row["model"] == model:
                return int(row["K"])
    raise ValueError(f"No entry for {corpus}/{model}")

def load_scptm_mwes(corpus: str, best_k: int):
    fpath = WORDS_DIR / f"words_{corpus}.csv"
    mwes = []
    with open(fpath, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["model"] != MODEL or int(row["K"]) != best_k:
                continue
            cell = row.get("mwe_phrases", "")
            if cell:
                mwes.extend(p.strip() for p in cell.split("|") if p.strip())
    return mwes

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    corpora = list(LOADERS.keys())
    rows = []

    for corpus in corpora:
        print(f"\n── {corpus} ──────────────────────────────────────────────")
        try:
            print("  Loading corpus text …")
            docs = LOADERS[corpus]()
            print(f"  {len(docs)} documents loaded")
        except Exception as exc:
            print(f"  [ERROR loading corpus] {exc}")
            continue

        # PMI bigrams
        print(f"  Computing top-{TOP_N_PMI} PMI bigrams …")
        pmi_bigrams = top_pmi_bigrams(docs, n=TOP_N_PMI)
        mean_pmi    = mean_vader(pmi_bigrams)
        print(f"  PMI bigrams mean VADER: {mean_pmi:+.4f}  (n={len(pmi_bigrams)})")
        print(f"  Sample: {pmi_bigrams[:5]}")

        # SCPTM MWEs
        best_k = load_best_k(corpus, MODEL)
        mwes   = load_scptm_mwes(corpus, best_k)
        mean_mwe = mean_vader(mwes)
        print(f"  SCPTM MWEs  mean VADER: {mean_mwe:+.4f}  (n={len(mwes)}, K={best_k})")
        if mwes:
            print(f"  Sample: {mwes[:5]}")

        diff = (mean_mwe - mean_pmi) if mwes else float("nan")
        print(f"  Δ (SCPTM MWE − PMI bigram): {diff:+.4f}")

        rows.append({
            "corpus":           corpus,
            "K_scptm":          best_k,
            "n_pmi_bigrams":    len(pmi_bigrams),
            "n_scptm_mwes":     len(mwes),
            "vader_pmi_mean":   round(mean_pmi,   4),
            "vader_mwe_mean":   round(mean_mwe,   4),
            "delta_mwe_minus_pmi": round(diff,    4),
        })

    if not rows:
        print("No results — check paths.")
        sys.exit(1)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n\nSaved → {OUT_CSV}")

    # ── summary table ─────────────────────────────────────────────────────────
    print("\n── Summary: VADER(MWE) vs VADER(PMI bigrams) per corpus ─────────")
    print(f"  {'corpus':<14} {'PMI_bigrams':>12} {'SCPTM_MWEs':>12} {'Δ':>8}")
    print(f"  {'-'*14} {'-'*12} {'-'*12} {'-'*8}")
    for r in rows:
        print(f"  {r['corpus']:<14} {r['vader_pmi_mean']:>+12.4f} "
              f"{r['vader_mwe_mean']:>+12.4f} {r['delta_mwe_minus_pmi']:>+8.4f}")
    print()
    pos = sum(1 for r in rows if r["delta_mwe_minus_pmi"] > 0)
    print(f"  Positive Δ (syntactic > statistical) in {pos}/{len(rows)} corpora.\n")


if __name__ == "__main__":
    main()
