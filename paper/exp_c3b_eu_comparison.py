# -*- coding: utf-8 -*-
"""
exp_c3b_eu_comparison.py
C3B re-run using EU_Debates from results_v3 — consistent with paper corpus.
Loads EU Parliament Debates from HuggingFace (coastalcph/eu_debates).
Replaces the earlier run that used UN General Debates.
"""

import csv
import math
import re
import sys
from pathlib import Path
from collections import Counter

BASE       = Path(__file__).parent
WORDS_DIR  = BASE.parent / "results_v3"
MAIN_TABLE = WORDS_DIR / "main_table.csv"
OUT_CSV    = WORDS_DIR / "c3b_pmi_comparison.csv"

REDDIT_CSV = BASE / "reddit_pol.csv"

TOP_N_PMI = 200
MODEL     = "SCPTM"

# ── VADER ─────────────────────────────────────────────────────────────────────

def _vader():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    if not hasattr(_vader, "_inst"):
        _vader._inst = SentimentIntensityAnalyzer()
    return _vader._inst

def vader_score(text):
    return _vader().polarity_scores(text)["compound"]

def mean_vader(phrases):
    scores = [vader_score(p) for p in phrases if p and p.strip()]
    return sum(scores) / len(scores) if scores else float("nan")

# ── corpus loaders ────────────────────────────────────────────────────────────

def load_20ng():
    from sklearn.datasets import fetch_20newsgroups
    data = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    return data.data

def load_reddit():
    import csv as _csv
    docs = []
    with open(REDDIT_CSV, newline="", encoding="utf-8", errors="replace") as f:
        for row in _csv.DictReader(f):
            text = ""
            for key in row:
                if key.strip().lower() in ("text", "body", "selftext", "content"):
                    text = row[key]
                    break
            if text.strip():
                docs.append(text.strip())
    return docs

def load_eu_debates(max_docs=20_000, seed=42):
    import json, zipfile, random
    from huggingface_hub import hf_hub_download
    zip_path = hf_hub_download(
        repo_id="coastalcph/eu_debates",
        filename="eu_debates.zip",
        repo_type="dataset",
    )
    docs = []
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("train.jsonl") as f:
            for line in f:
                row = json.loads(line)
                tt = row.get("translated_text", None)
                if tt is not None and isinstance(tt, str) and tt.strip():
                    continue  # skip machine-translated
                text = row.get("text", "")
                if text and len(text.split()) >= 30:
                    docs.append(text)
    if max_docs and len(docs) > max_docs:
        rng = random.Random(seed)
        docs = rng.sample(docs, max_docs)
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
    "HateSpeech": load_hatespeech,
    "Reddit_Pol": load_reddit,
    "EU_Debates": load_eu_debates,
    "20NG":       load_20ng,
}

# ── PMI bigrams ───────────────────────────────────────────────────────────────

_TOKENIZE = re.compile(r"[a-z]{2,}")

def tokenize(text):
    return _TOKENIZE.findall(text.lower())

def top_pmi_bigrams(docs, n=200, min_count=10):
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

def load_best_k(corpus, model):
    with open(MAIN_TABLE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["corpus"] == corpus and row["model"] == model:
                return int(row["K"])
    raise ValueError(f"No entry for {corpus}/{model}")

def load_scptm_mwes(corpus, best_k):
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
    rows = []
    for corpus in LOADERS:
        print(f"\n-- {corpus} --")
        try:
            print("  Loading corpus text ...")
            docs = LOADERS[corpus]()
            print(f"  {len(docs)} documents loaded")
        except Exception as exc:
            print(f"  [ERROR loading corpus] {exc}")
            continue

        print(f"  Computing top-{TOP_N_PMI} PMI bigrams ...")
        pmi_bigrams = top_pmi_bigrams(docs, n=TOP_N_PMI)
        mean_pmi    = mean_vader(pmi_bigrams)
        print(f"  PMI bigrams mean VADER: {mean_pmi:+.4f}  (n={len(pmi_bigrams)})")
        print(f"  Sample PMI: {pmi_bigrams[:5]}")

        best_k   = load_best_k(corpus, MODEL)
        mwes     = load_scptm_mwes(corpus, best_k)
        mean_mwe = mean_vader(mwes)
        diff     = (mean_mwe - mean_pmi) if mwes else float("nan")
        print(f"  SCPTM MWEs  mean VADER: {mean_mwe:+.4f}  (n={len(mwes)}, K={best_k})")
        print(f"  Delta (SCPTM - PMI): {diff:+.4f}")

        rows.append({
            "corpus":              corpus,
            "K_scptm":             best_k,
            "n_pmi_bigrams":       len(pmi_bigrams),
            "n_scptm_mwes":        len(mwes),
            "vader_pmi_mean":      round(mean_pmi,  4),
            "vader_mwe_mean":      round(mean_mwe,  4),
            "delta_mwe_minus_pmi": round(diff,       4),
        })

    if not rows:
        print("No results.")
        sys.exit(1)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved -> {OUT_CSV}")

    print("\n-- Summary: VADER(MWE) vs VADER(PMI bigrams) --")
    print(f"  {'corpus':<14} {'PMI':>10} {'MWE':>10} {'Delta':>8}")
    for r in rows:
        print(f"  {r['corpus']:<14} {r['vader_pmi_mean']:>+10.4f} "
              f"{r['vader_mwe_mean']:>+10.4f} {r['delta_mwe_minus_pmi']:>+8.4f}")
    pos = sum(1 for r in rows if r["delta_mwe_minus_pmi"] > 0)
    print(f"\n  Positive Delta in {pos}/{len(rows)} corpora.")

if __name__ == "__main__":
    main()
