# -*- coding: utf-8 -*-
"""
exp_c4_undebates_only.py
Run C4 (TF-IDF concentration) only for UN_Debates and append to
the existing c4_tfidf_concentration.csv produced by exp_c4_tfidf_concentration.py.
"""

import csv, sys, numpy as np
from pathlib import Path

# ── import shared helpers from main C4 script ─────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from exp_c4_tfidf_concentration import (
    load_best_k_table, run_lda, run_ctm, run_bertopic, run_scptm,
    tfidf_concentration, SEED, MODELS
)

BASE      = Path(__file__).parent
UNGDC_CSV = BASE / "un-general-debates.csv"
WORDS_DIR = BASE / "benchmark_cache" / "results_0605"
OUT_CSV   = WORDS_DIR / "c4_tfidf_concentration.csv"
CACHE_DIR = BASE / "benchmark_cache" / "c4_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def load_un_debates():
    import csv as _csv
    docs = []
    with open(UNGDC_CSV, newline="", encoding="utf-8", errors="replace") as f:
        for row in _csv.DictReader(f):
            text = row.get("text") or row.get("speech") or row.get("value") or ""
            if text.strip():
                docs.append(text.strip())
    return docs

def main():
    from sklearn.feature_extraction.text import TfidfVectorizer

    best_k_table = load_best_k_table()
    corpus = "UN_Debates"

    print(f"\n{'='*60}")
    print(f"CORPUS: {corpus}")
    print(f"{'='*60}")

    docs = load_un_debates()
    if not docs:
        print("ERROR: no documents loaded — check column names in un-general-debates.csv")
        print("First row columns:", end=" ")
        with open(UNGDC_CSV, newline="", encoding="utf-8", errors="replace") as f:
            print(list(next(csv.DictReader(f)).keys()))
        sys.exit(1)
    print(f"  {len(docs)} documents loaded")

    print("  Fitting TF-IDF ...")
    tfidf  = TfidfVectorizer(max_features=20000, min_df=5,
                             sublinear_tf=True, stop_words="english")
    X_tfidf = tfidf.fit_transform(docs)
    print(f"  TF-IDF matrix: {X_tfidf.shape}")

    new_rows = []
    for model_name in MODELS:
        K = best_k_table.get((corpus, model_name))
        if K is None:
            print(f"  [{model_name}] No best-K — skipping")
            continue
        print(f"\n  [{model_name}]  K={K} ...")
        try:
            doc_mask = None
            if model_name == "LDA":
                theta = run_lda(docs, K, SEED)
            elif model_name == "CTM":
                theta = run_ctm(docs, K, SEED)
            elif model_name == "BERTopic":
                theta = run_bertopic(docs, K, SEED)
            elif model_name == "SCPTM":
                theta, doc_mask = run_scptm(docs, K, SEED, corpus, CACHE_DIR)

            labels = np.argmax(theta, axis=1)
            conc   = tfidf_concentration(X_tfidf, labels, doc_mask)
            print(f"    TF-IDF concentration: {conc:.4f}")
            new_rows.append({
                "corpus": corpus, "model": model_name, "K": K,
                "n_docs": len(docs),
                "concentration_tfidf": round(conc, 4),
            })
        except Exception as exc:
            print(f"    [ERROR] {exc}")
            import traceback; traceback.print_exc()

    if not new_rows:
        print("No results."); sys.exit(1)

    # Append to existing CSV
    write_header = not OUT_CSV.exists()
    with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(new_rows[0].keys()))
        if write_header:
            w.writeheader()
        w.writerows(new_rows)
    print(f"\nAppended {len(new_rows)} rows to {OUT_CSV}")

    print(f"\n  {'model':<12} {'K':>4} {'TF-IDF conc':>12}")
    for r in new_rows:
        print(f"  {r['model']:<12} {r['K']:>4} {r['concentration_tfidf']:>12.4f}")

if __name__ == "__main__":
    main()
