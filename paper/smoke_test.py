"""
smoke_test.py
=============
Quick end-to-end validation of the SCPTM benchmark stack.

Runs all 7 models with a tiny 20-newsgroups subset (200 docs, K=5, 1 seed,
10 epochs) and validates that every metric family produces at least some
non-NaN values.  Completes in ~5-15 min on CPU; faster on GPU.

Usage
-----
  python smoke_test.py               # full smoke (all 7 models)
  python smoke_test.py --libs-only   # import checks only, no training
  python smoke_test.py --no-stance   # skip the HF stance classifier
"""

import argparse
import sys
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--libs-only", action="store_true",
                    help="Only check imports, no model training")
parser.add_argument("--no-stance", action="store_true",
                    help="Skip the HF stance classifier (saves ~2 min)")
args = parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# 0. Library probe
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED  = ["torch", "torch_geometric", "sentence_transformers",
             "gensim", "bertopic", "spacy", "sklearn", "scipy", "numpy"]
OPTIONAL  = {
    "contextualized_topic_models": "CTM-vanilla baseline",
    "vaderSentiment":              "RQ3 valence anchor (VADER)",
    "transformers":                "RQ3 stance classifier",
}

print("\n" + "=" * 60)
print("STEP 0 — Library probe")
print("=" * 60)

missing_req, missing_opt = [], []

for lib in REQUIRED:
    try:
        m = __import__(lib)
        v = getattr(m, "__version__", "?")
        print(f"  ✓  {lib} {v}")
    except ImportError as e:
        print(f"  ✗  {lib} MISSING — {e}")
        missing_req.append(lib)

for lib, purpose in OPTIONAL.items():
    try:
        m = __import__(lib)
        v = getattr(m, "__version__", "?")
        print(f"  ✓  {lib} {v}  ({purpose})")
    except ImportError as e:
        print(f"  ⚠  {lib} not installed ({purpose}) — metrics will be NaN")
        missing_opt.append(lib)

if missing_req:
    print(f"\n✗ ABORT: required libraries missing: {missing_req}")
    sys.exit(1)

if args.libs_only:
    print("\n--libs-only: stopping after import check.")
    sys.exit(0 if not missing_req else 1)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Imports
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 1 — Importing benchmark modules")
print("=" * 60)

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

import paper_metrics as pm

# Patch benchmark settings before importing to keep the smoke run tiny
import benchmark_paper as bm
bm.EPOCHS    = 10          # was 50
bm.TOP_K_WORDS = 8
bm.RUN_STANCE  = not args.no_stance
bm.STANCE_SEEDS = [42]

from benchmark_paper import (
    build_bigram_corpus, build_reference_bow,
    precompute_sbert_embeddings,
    run_lda, run_scptm_variant, run_ctm_vanilla, run_bertopic,
    evaluate_all, SCPTM_MODES, MODEL_ORDER, METRIC_COLUMNS,
)
from sentence_transformers import SentenceTransformer

print("  All imports OK.")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Tiny corpus
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 2 — Loading 200-doc 20NG subset")
print("=" * 60)

try:
    raw = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"),
                             categories=["sci.space", "rec.sport.hockey",
                                         "talk.religion.misc", "comp.graphics",
                                         "sci.med"])
    rng = np.random.default_rng(42)
    idx = rng.choice(len(raw.data), size=min(200, len(raw.data)), replace=False)
    docs   = [raw.data[i]   for i in idx]
    labels = raw.target[idx]
    print(f"  {len(docs)} docs, {len(set(labels))} categories")
except Exception as e:
    print(f"  sklearn download failed ({e}); generating synthetic corpus...")
    from itertools import cycle
    themes = [
        "The rocket launched into orbit successfully reaching the space station",
        "The hockey game ended with a goal in overtime period playoffs",
        "Religious beliefs vary widely across different cultures and traditions",
        "The graphics card renders high quality images with fast processing",
        "Medical research shows promising results for new cancer treatments",
    ]
    docs   = [t + f" document number {i}" for i, t in zip(range(200), cycle(themes))]
    labels = np.array([i % 5 for i in range(200)])
    print(f"  Synthetic: {len(docs)} docs, 5 topics")

K   = 5
SEED = 42

# Shared evaluation data
bigram_texts, _ = build_bigram_corpus(docs, min_count=2)
bow_ref, vocab_ref = build_reference_bow(bigram_texts, min_df=2)
sbert = SentenceTransformer("all-MiniLM-L6-v2")
sbert_embs = sbert.encode(docs, show_progress_bar=False)

tokenized_texts = [
    [w for w in d.lower().split() if w.isalpha() and len(w) > 2]
    for d in docs
]
_df = np.asarray((bow_ref > 0).sum(axis=0)).ravel()
df_lookup = {vocab_ref[i]: int(_df[i]) for i in range(len(vocab_ref)) if _df[i] > 0}

cache_dir = Path("/tmp/scptm_smoke")
cache_dir.mkdir(exist_ok=True)
cache_base = str(cache_dir / "smoke_cache.pkl")

print("  Corpus prep done.")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Run all 7 models
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 3 — Training all 7 models  (K=5, seed=42, 10 epochs)")
print("=" * 60)

results = {}
errors  = {}

for model_name in MODEL_ORDER:
    print(f"\n  ── {model_name} ──")
    try:
        theta = None; doc_mask = None
        mwe_phrases = None; single_phrases = None

        if model_name == "LDA":
            tw, theta, doc_mask = run_lda(docs, K, SEED, 2, bigram_texts=bigram_texts)

        elif model_name == "CTM":
            tw, theta, doc_mask = run_ctm_vanilla(docs, K, SEED, sbert_embs, 2)

        elif model_name in SCPTM_MODES:
            tw, theta, doc_mask, mwe_phrases, single_phrases = \
                run_scptm_variant(docs, K, SEED, SCPTM_MODES[model_name],
                                  cache_base, 2)

        elif model_name == "BERTopic":
            tw, theta, doc_mask = run_bertopic(docs, K, SEED, sbert_embs, 2)

        aligned_labels = (np.array(labels)[np.asarray(doc_mask, dtype=bool)]
                          if doc_mask is not None else labels)

        metrics = evaluate_all(
            tw, bow_ref, vocab_ref, sbert,
            tokenized_texts=tokenized_texts,
            df_lookup=df_lookup,
            theta=theta,
            doc_mask=doc_mask,
            doc_embeddings_full=sbert_embs,
            true_labels=aligned_labels,
            mwe_per_topic=mwe_phrases,
            single_per_topic=single_phrases,
            raw_docs=docs,
            run_stance=(bm.RUN_STANCE and model_name in SCPTM_MODES),
        )

        results[model_name] = {c: metrics.get(c, np.nan) for c in METRIC_COLUMNS}
        print(f"    C_V={metrics.get('c_v', float('nan')):.3f}  "
              f"C_NPMI={metrics.get('c_npmi', float('nan')):.3f}  "
              f"Div={metrics.get('diversity', float('nan')):.3f}  "
              f"NMI={metrics.get('nmi', float('nan')):.3f}"
              + (f"  MWE-cmp={metrics['mwe_compactness']:.3f}"
                 if "mwe_compactness" in metrics else ""))

    except Exception as e:
        print(f"    ✗ FAILED: {e}")
        traceback.print_exc()
        errors[model_name] = str(e)
        results[model_name] = {c: np.nan for c in METRIC_COLUMNS}

# ─────────────────────────────────────────────────────────────────────────────
# 4. Validation checks
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 4 — Validation")
print("=" * 60)

df = pd.DataFrame(results).T
print(df.to_string())

FAIL = []

# Every model should have at least one non-NaN coherence metric
core_metrics = ["diversity", "we_coherence", "nmi"]
for model in MODEL_ORDER:
    if model in errors:
        FAIL.append(f"{model}: training FAILED — {errors[model]}")
        continue
    row = df.loc[model]
    for m in core_metrics:
        if pd.notna(row.get(m)) and not np.isnan(float(row.get(m, np.nan))):
            break
    else:
        FAIL.append(f"{model}: all core metrics are NaN")

# SCPTM modes should have MWE metrics
for mode in SCPTM_MODES:
    if mode in errors:
        continue
    row = df.loc[mode]
    if pd.isna(row.get("mwe_compactness")):
        FAIL.append(f"{mode}: mwe_compactness is NaN (MWE extraction may have failed)")

# gensim coherence warning (not a hard fail)
for model in MODEL_ORDER:
    if model in errors:
        continue
    row = df.loc[model]
    if pd.isna(row.get("c_v")):
        print(f"  ⚠  {model}: C_V is NaN — gensim CoherenceModel may need more docs")

# VADER valence warning
for mode in SCPTM_MODES:
    if mode in errors:
        continue
    row = df.loc[mode]
    if pd.isna(row.get("valence_gap")):
        print(f"  ⚠  {mode}: valence_gap is NaN — vaderSentiment may be missing")

print("\n" + "=" * 60)
if FAIL:
    print("✗ SMOKE TEST FAILED:")
    for f in FAIL:
        print(f"    • {f}")
    sys.exit(1)
else:
    print("✓ SMOKE TEST PASSED — all models trained, core metrics non-NaN.")
    if missing_opt:
        print(f"  Optional libs missing (metrics will be NaN in full run): {missing_opt}")
    if errors:
        print(f"  Training errors (will appear as NaN rows in benchmark): {list(errors)}")
    print("  Ready to launch: python benchmark_paper.py")
print("=" * 60)
