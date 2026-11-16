# -*- coding: utf-8 -*-
"""
exp_c3a_vader_validation.py
────────────────────────────
C3A: Cross-validate the VADER valence gap with a second, independent
     sentiment scorer (TextBlob polarity).

For each corpus, for SCPTM (filtered) at best-K:
  • score each MWE phrase with VADER  → mean_mwe_vader
  • score each unigram  with VADER    → mean_uni_vader
  • same pair with TextBlob polarity
  • valence_gap = mean_mwe − mean_uni (for both scorers)

If the cross-corpus ordering (HateSpeech > Reddit_Pol > UN_Debates / 20NG)
holds under both scorers, VADER is validated despite its social-media calibration.

Dependencies (install in your env if missing):
    pip install vaderSentiment textblob

Output:
    benchmark_cache/results_0605/c3a_vader_validation.csv
    printed table + ordering summary
"""

import csv
import sys
from pathlib import Path

BASE       = Path(__file__).parent
WORDS_DIR  = BASE / "benchmark_cache" / "results_0605"
MAIN_TABLE = WORDS_DIR / "main_table.csv"
OUT_CSV    = WORDS_DIR / "c3a_vader_validation.csv"

CORPORA    = ["HateSpeech", "Reddit_Pol", "UN_Debates", "20NG"]
MODEL      = "SCPTM"   # filtered variant

# ── helpers ──────────────────────────────────────────────────────────────────

def load_best_k(corpus: str, model: str) -> int:
    with open(MAIN_TABLE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["corpus"] == corpus and row["model"] == model:
                return int(row["K"])
    raise ValueError(f"No entry for {corpus}/{model} in {MAIN_TABLE}")


def load_phrases(corpus: str, model: str, best_k: int):
    """Return (mwe_list, unigram_list) for corpus/model/best_k across all seeds."""
    fpath = WORDS_DIR / f"words_{corpus}.csv"
    mwes, unis = [], []
    with open(fpath, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["model"] != model or int(row["K"]) != best_k:
                continue
            cell_m = row.get("mwe_phrases", "")
            cell_u = row.get("keywords", "")
            if cell_m:
                mwes.extend(p.strip() for p in cell_m.split("|") if p.strip())
            if cell_u:
                unis.extend(w.strip() for w in cell_u.split("|") if w.strip())
    return mwes, unis


def _vader():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    if not hasattr(_vader, "_inst"):
        _vader._inst = SentimentIntensityAnalyzer()
    return _vader._inst

def vader_score(text: str) -> float:
    return _vader().polarity_scores(text)["compound"]


def textblob_score(text: str) -> float:
    from textblob import TextBlob
    return TextBlob(text).sentiment.polarity


SCORERS = {"VADER": vader_score, "TextBlob": textblob_score}


def mean(lst):
    return sum(lst) / len(lst) if lst else float("nan")

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    rows = []
    for corpus in CORPORA:
        best_k = load_best_k(corpus, MODEL)
        mwes, unis = load_phrases(corpus, MODEL, best_k)
        if not mwes:
            print(f"[WARN] {corpus}: no MWEs found — skipping")
            continue
        print(f"\n{corpus}  (K={best_k})  →  {len(mwes)} MWEs, {len(unis)} unigrams")
        row = {"corpus": corpus, "K": best_k,
               "n_mwes": len(mwes), "n_unigrams": len(unis)}
        for name, fn in SCORERS.items():
            try:
                mwe_s = [fn(p) for p in mwes]
                uni_s = [fn(w) for w in unis]
                m_mwe = mean(mwe_s)
                m_uni = mean(uni_s)
                gap   = m_mwe - m_uni
                row[f"{name}_mwe"]  = round(m_mwe, 4)
                row[f"{name}_uni"]  = round(m_uni, 4)
                row[f"{name}_gap"]  = round(gap,   4)
                print(f"  {name:<10}  MWE={m_mwe:+.4f}  UNI={m_uni:+.4f}  GAP={gap:+.4f}")
            except Exception as exc:
                print(f"  {name}: ERROR — {exc}")
                row[f"{name}_gap"] = None
        rows.append(row)

    if not rows:
        print("No data — check paths and corpus names.")
        sys.exit(1)

    fields = list(rows[0].keys())
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved → {OUT_CSV}")

    # ── ordering check ────────────────────────────────────────────────────────
    print("\n── Cross-corpus ordering by valence gap ─────────────────────────")
    for name in SCORERS:
        col = f"{name}_gap"
        valid = [r for r in rows if r.get(col) is not None]
        ranked = sorted(valid, key=lambda r: r[col], reverse=True)
        print(f"  {name:<10}: " +
              " > ".join(f"{r['corpus']}({r[col]:+.4f})" for r in ranked))

    print("\nIf both scorers agree on the ordering, VADER is validated.\n")


if __name__ == "__main__":
    main()
