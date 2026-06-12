"""
downstream_task.py
==================
Extrinsic evaluation: do topic-model representations help classify
documents in downstream tasks?

Corpora
-------
  HateSpeech : binary  — hateful (target_* ≥ 0.5 or hate_score > 0) vs neutral
  Reddit_Pol : binary  — Conservative (0) vs Liberal (1)

Models compared
---------------
  TF-IDF           TfidfVectorizer(1-2 gram) + LogReg   (raw-text baseline)
  LDA              θ_d
  CTM              θ_d
  SCPTM-none       θ_d
  SCPTM-nosyn      θ_d
  SCPTM-fulldep    θ_d
  SCPTM            θ_d                           (full model)
  SCPTM+MWE        binary MWE-presence features  (syntactic features only)
  SCPTM+θ+MWE      θ_d concatenated with MWE     (combined)
  BERTopic         soft topic probabilities

Protocol
--------
  * Fixed K = K_DOWNSTREAM per corpus
  * SEEDS independent training runs → mean ± std of F1-macro and ROC-AUC
  * 5-fold stratified CV per seed (classifier: LogReg, balanced class weights)
  * Topic models are trained on ALL documents (unsupervised); the classifier
    is evaluated on the resulting features via CV — no label leakage into the
    topic model.
  * doc_mask (SCPTM / LDA drop very short docs) is applied to align labels.

Usage
-----
  python downstream_task.py
  python downstream_task.py --corpus HateSpeech --k 10 --seeds 42 123 2024
  python downstream_task.py --corpus Reddit_Pol --k 15
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import issparse

warnings.filterwarnings("ignore")

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from benchmark_paper import (
    load_hate_speech,
    load_reddit_politics,
    run_lda,
    run_ctm_vanilla,
    run_scptm_variant,
    run_bertopic,
    SBERT_MODEL,
    TOP_K_WORDS,
    EPOCHS,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

# Root for cache files and output.
# Override via --drive-root when running on Colab/server:
#   python downstream_task.py --drive-root /content/drive/MyDrive/benchmark_cache
DRIVE_ROOT = str(ROOT / "benchmark_cache")

# Path to the Reddit CSV (same as benchmark_paper.py).
REDDIT_CSV = str(ROOT / "reddit_pol.csv")

# Cache paths mirror benchmark_paper.py so the server's existing parse caches
# (hatespeech_cache.*.pkl, reddit_cache.*.pkl) are reused automatically.
CORPUS_CACHE = {
    "HateSpeech": "hatespeech_cache.pkl",
    "Reddit_Pol":  "reddit_cache.pkl",
}

# Fixed K for downstream evaluation.
# HateSpeech: 8 ground-truth categories → K=10 gives enough resolution without
#             over-fragmenting; Reddit_Pol is binary but ideological sub-topics
#             exist, K=10 works well.
K_DOWNSTREAM = {
    "HateSpeech": 10,
    "Reddit_Pol":  10,
}

SEEDS    = [42, 123, 2024]
CV_FOLDS = 5

# Model display order in the summary table
MODEL_ORDER = [
    "TF-IDF",
    "LDA", "CTM",
    "SCPTM-none", "SCPTM-nosyn", "SCPTM-fulldep",
    "SCPTM", "SCPTM+MWE", "SCPTM+θ+MWE",
    "BERTopic",
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_mwe_features(docs: list, mwe_phrases_per_topic: list):
    """
    Build a binary document × MWE matrix from SCPTM topic MWEs.

    For each document, each entry is 1 if the MWE string appears anywhere in the
    lowercased document text, 0 otherwise.  Vocabulary = union of all topic MWEs.

    Parameters
    ----------
    docs                 : list[str]  — filtered documents (doc_mask applied)
    mwe_phrases_per_topic: list[list[str]]  — top MWEs per topic from SCPTM

    Returns
    -------
    feat     : np.ndarray (n_docs, n_unique_mwes)
    mwe_vocab: list[str]
    """
    # Build vocabulary preserving first-seen order, deduplicating
    seen = {}
    for topic_mwes in mwe_phrases_per_topic:
        if not topic_mwes:
            continue
        for mwe in topic_mwes:
            mwe_l = mwe.lower().strip()
            if mwe_l and mwe_l not in seen:
                seen[mwe_l] = len(seen)
    mwe_vocab = list(seen.keys())

    if not mwe_vocab:
        return np.zeros((len(docs), 0), dtype=np.float32), mwe_vocab

    docs_lower = [str(d).lower() for d in docs]
    feat = np.array(
        [[1.0 if mwe in doc else 0.0 for mwe in mwe_vocab]
         for doc in docs_lower],
        dtype=np.float32,
    )
    print(f"    MWE vocabulary: {len(mwe_vocab)} unique phrases; "
          f"mean present per doc = {feat.mean(axis=1).mean():.2f}")
    return feat, mwe_vocab


def eval_features(X, y: np.ndarray, n_splits: int = CV_FOLDS, seed: int = 42):
    """
    5-fold stratified CV with a StandardScaler + LogisticRegression pipeline.

    Uses balanced class weights to handle class imbalance.
    ROC-AUC: OvR macro for both binary and multi-class tasks.

    Returns dict: f1_macro_mean, f1_macro_std, auc_mean, auc_std.
    """
    if X is None or (hasattr(X, "shape") and X.shape[1] == 0):
        return dict(f1_macro_mean=np.nan, f1_macro_std=np.nan,
                    auc_mean=np.nan, auc_std=np.nan)

    n_classes = len(np.unique(y))

    # StandardScaler: skip with_mean for sparse matrices
    scaler = StandardScaler(with_mean=not issparse(X))
    clf = Pipeline([
        ("scaler", scaler),
        ("lr", LogisticRegression(
            max_iter     = 2000,
            class_weight = "balanced",
            random_state = seed,
            C            = 1.0,
            solver       = "lbfgs",
        )),
    ])

    # AUC: binary uses probability of positive class (column 1);
    #      multi-class uses OvR macro with full probability matrix.
    def _auc_scorer(estimator, X, y):
        proba = estimator.predict_proba(X)
        try:
            if proba.shape[1] == 2:
                return roc_auc_score(y, proba[:, 1])
            return roc_auc_score(y, proba, multi_class="ovr", average="macro")
        except Exception:
            return np.nan

    scoring = {
        "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
        "auc":      _auc_scorer,
    }

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    try:
        results = cross_validate(clf, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    except Exception as e:
        print(f"    [WARN] cross_validate failed: {e}")
        return dict(f1_macro_mean=np.nan, f1_macro_std=np.nan,
                    auc_mean=np.nan, auc_std=np.nan)

    return dict(
        f1_macro_mean = float(np.mean(results["test_f1_macro"])),
        f1_macro_std  = float(np.std(results["test_f1_macro"])),
        auc_mean      = float(np.nanmean(results["test_auc"])),
        auc_std       = float(np.nanstd(results["test_auc"])),
    )


def _merge_seed_results(seed_results: list) -> dict:
    """Average F1 and AUC across seeds."""
    return dict(
        f1_macro     = float(np.nanmean([r["f1_macro_mean"] for r in seed_results])),
        f1_macro_std = float(np.nanmean([r["f1_macro_std"]  for r in seed_results])),
        auc          = float(np.nanmean([r["auc_mean"]       for r in seed_results])),
        auc_std      = float(np.nanmean([r["auc_std"]        for r in seed_results])),
    )


def _print_row(model_name: str, metrics: dict, tag: str = ""):
    label = f"{model_name}{(' ' + tag) if tag else ''}"
    print(f"  {label:<22s}  "
          f"F1={metrics['f1_macro']:.3f}±{metrics['f1_macro_std']:.3f}  "
          f"AUC={metrics['auc']:.3f}±{metrics['auc_std']:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_downstream(
    corpus_name: str,
    docs: list,
    labels: np.ndarray,
    K: int,
    seeds: list,
    out_path: Path = None,
) -> pd.DataFrame:
    """
    Run downstream classification for all models on a single corpus.
    Saves results incrementally after each model so a crash loses at most
    one model's work.
    Returns DataFrame with one row per (corpus, model).
    """
    from sentence_transformers import SentenceTransformer

    rows = []
    # Reuse the same parse cache as benchmark_paper.py (avoids re-parsing)
    cache_path = str(Path(DRIVE_ROOT) / CORPUS_CACHE[corpus_name])

    def _save_incremental():
        if out_path is not None and rows:
            pd.DataFrame(rows).to_csv(out_path, index=False)

    print(f"\n{'='*64}")
    print(f"CORPUS: {corpus_name}  |  K={K}  |  n={len(docs):,}  "
          f"|  seeds={seeds}")
    label_counts = dict(zip(*np.unique(labels, return_counts=True)))
    print(f"Labels: {label_counts}")
    print(f"{'='*64}")

    def append_row(model, metrics, K_val=K):
        rows.append({"corpus": corpus_name, "model": model, "K": K_val, **metrics})

    # ── TF-IDF baseline ───────────────────────────────────────────────────────
    print("\n[TF-IDF baseline]")
    tfidf = TfidfVectorizer(
        max_features = 20_000,
        ngram_range  = (1, 2),
        sublinear_tf = True,
        min_df       = 3,
    )
    X_tfidf = tfidf.fit_transform(docs)
    sr = [eval_features(X_tfidf, labels, seed=s) for s in seeds]
    m = _merge_seed_results(sr)
    append_row("TF-IDF", m, K_val="—")
    _print_row("TF-IDF", m)
    _save_incremental()

    # ── Pre-compute SBERT (shared for BERTopic / CTM) ─────────────────────────
    print("\n[SBERT] Pre-computing embeddings …")
    sbert = SentenceTransformer(SBERT_MODEL)
    sbert_embs = sbert.encode(docs, show_progress_bar=True, batch_size=64)
    del sbert  # free memory

    # ── LDA ──────────────────────────────────────────────────────────────────
    print("\n[LDA]")
    sr = []
    for seed in seeds:
        try:
            _, theta, doc_mask = run_lda(docs, K=K, seed=seed,
                                         top_k=TOP_K_WORDS)
            y = labels[doc_mask]
            sr.append(eval_features(theta, y, seed=seed))
        except Exception as e:
            print(f"  [WARN] seed={seed}: {e}")
            sr.append(dict(f1_macro_mean=np.nan, f1_macro_std=np.nan,
                           auc_mean=np.nan, auc_std=np.nan))
    m = _merge_seed_results(sr)
    append_row("LDA", m)
    _print_row("LDA", m)
    _save_incremental()

    # ── CTM ──────────────────────────────────────────────────────────────────
    print("\n[CTM]")
    sr = []
    for seed in seeds:
        try:
            _, theta, doc_mask = run_ctm_vanilla(
                docs, K=K, seed=seed, sbert_embs=sbert_embs,
                top_k=TOP_K_WORDS
            )
            if theta is None:
                raise ValueError("CTM theta is None")
            y = labels[doc_mask]
            sr.append(eval_features(theta, y, seed=seed))
        except Exception as e:
            print(f"  [WARN] seed={seed}: {e}")
            sr.append(dict(f1_macro_mean=np.nan, f1_macro_std=np.nan,
                           auc_mean=np.nan, auc_std=np.nan))
    m = _merge_seed_results(sr)
    append_row("CTM", m)
    _print_row("CTM", m)
    _save_incremental()

    # ── SCPTM variants ────────────────────────────────────────────────────────
    scptm_modes = [
        ("SCPTM-none",    "none"),
        ("SCPTM-nosyn",   "no_syntax"),
        ("SCPTM-fulldep", "full_dep"),
        ("SCPTM",         "filtered"),
    ]

    for model_name, graph_mode in scptm_modes:
        print(f"\n[{model_name}]")
        sr_theta = []
        sr_mwe   = []   # only filled for full model
        sr_combo = []   # only filled for full model

        for seed in seeds:
            try:
                _, theta, doc_mask, mwe_phrases, _ = run_scptm_variant(
                    docs, K=K, seed=seed,
                    graph_mode  = graph_mode,
                    cache_path  = cache_path,
                    top_k       = TOP_K_WORDS,
                )
                y = labels[doc_mask]
                sr_theta.append(eval_features(theta, y, seed=seed))

                # MWE and combined features only for the full model
                if model_name == "SCPTM" and mwe_phrases is not None:
                    valid_docs = [d for d, m in zip(docs, doc_mask) if m]
                    mwe_feat, mwe_vocab = make_mwe_features(valid_docs, mwe_phrases)

                    if mwe_feat.shape[1] > 0:
                        sr_mwe.append(eval_features(mwe_feat, y, seed=seed))
                        X_combo = np.hstack([theta, mwe_feat])
                        sr_combo.append(eval_features(X_combo, y, seed=seed))

            except Exception as e:
                print(f"  [WARN] seed={seed}: {e}")
                sr_theta.append(dict(f1_macro_mean=np.nan, f1_macro_std=np.nan,
                                     auc_mean=np.nan, auc_std=np.nan))

        m_theta = _merge_seed_results(sr_theta)
        append_row(model_name, m_theta)
        _print_row(model_name, m_theta, tag="(θ_d)")

        if sr_mwe:
            m_mwe = _merge_seed_results(sr_mwe)
            append_row("SCPTM+MWE", m_mwe)
            _print_row("SCPTM+MWE", m_mwe, tag="(MWE only)")

        if sr_combo:
            m_combo = _merge_seed_results(sr_combo)
            append_row("SCPTM+θ+MWE", m_combo)
            _print_row("SCPTM+θ+MWE", m_combo, tag="(θ_d + MWE)")

        _save_incremental()

    # ── BERTopic ──────────────────────────────────────────────────────────────
    print("\n[BERTopic]")
    sr = []
    for seed in seeds:
        try:
            _, theta, doc_mask = run_bertopic(
                docs, K=K, seed=seed,
                precomputed_embs=sbert_embs,
                top_k=TOP_K_WORDS,
            )
            if theta is None:
                raise ValueError("BERTopic returned None probabilities")
            y = labels[doc_mask]
            sr.append(eval_features(theta, y, seed=seed))
        except Exception as e:
            print(f"  [WARN] seed={seed}: {e}")
            sr.append(dict(f1_macro_mean=np.nan, f1_macro_std=np.nan,
                           auc_mean=np.nan, auc_std=np.nan))
    m = _merge_seed_results(sr)
    append_row("BERTopic", m)
    _print_row("BERTopic", m)
    _save_incremental()

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Downstream classification evaluation for SCPTM paper"
    )
    parser.add_argument(
        "--corpus",
        choices=["HateSpeech", "Reddit_Pol", "both"],
        default="both",
        help="Which corpus to evaluate (default: both)",
    )
    parser.add_argument(
        "--k", type=int, default=None,
        help="Override K for all corpora (default: corpus-specific)"
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=SEEDS,
        help="Training seeds (default: 42 123 2024)"
    )
    parser.add_argument(
        "--drive-root", type=str, default=None,
        help="Override DRIVE_ROOT (use on Colab: /content/drive/MyDrive/benchmark_cache)"
    )
    parser.add_argument(
        "--reddit-csv", type=str, default=None,
        help="Path to reddit_pol.csv if not at DRIVE_ROOT/../reddit_pol.csv"
    )
    args = parser.parse_args()

    # Apply overrides
    global DRIVE_ROOT, REDDIT_CSV
    if args.drive_root:
        DRIVE_ROOT = args.drive_root
    if args.reddit_csv:
        REDDIT_CSV = args.reddit_csv

    OUT_DIR = Path(DRIVE_ROOT) / "downstream"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    # ── HateSpeech ────────────────────────────────────────────────────────────
    if args.corpus in ("HateSpeech", "both"):
        docs_hs, labels_hs, label_names = load_hate_speech(min_tokens=30)
        # 6-class multi-class: race / religion / gender / sexuality / origin /
        # age_disability (labels 0-5). Drop hate_no_target (6) and neutral (7)
        # — too few examples (10 and 11 respectively) for reliable CV.
        mask_hs = labels_hs < 6
        docs_hs_6   = [d for d, m in zip(docs_hs, mask_hs) if m]
        labels_hs_6 = labels_hs[mask_hs]   # already in range 0-5
        print(f"\nHateSpeech 6-class distribution:")
        for i, name in enumerate(label_names[:6]):
            n = int((labels_hs_6 == i).sum())
            print(f"  {i} {name:20s}: {n:,}")
        print(f"  TOTAL: {len(docs_hs_6):,}  "
              f"(dropped {mask_hs.size - mask_hs.sum()} hate_no_target/neutral docs)")
        K_hs = args.k or K_DOWNSTREAM["HateSpeech"]
        hs_out = OUT_DIR / "downstream_HateSpeech.csv"
        df_hs = run_downstream("HateSpeech", docs_hs_6, labels_hs_6, K_hs, args.seeds,
                               out_path=hs_out)
        df_hs.to_csv(hs_out, index=False)
        all_results.append(df_hs)

    # ── Reddit_Pol ────────────────────────────────────────────────────────────
    if args.corpus in ("Reddit_Pol", "both"):
        docs_rd, labels_rd = load_reddit_politics(REDDIT_CSV)
        K_rd = args.k or K_DOWNSTREAM["Reddit_Pol"]
        rd_out = OUT_DIR / "downstream_Reddit_Pol.csv"
        df_rd = run_downstream("Reddit_Pol", docs_rd, labels_rd, K_rd, args.seeds,
                               out_path=rd_out)
        df_rd.to_csv(rd_out, index=False)
        all_results.append(df_rd)

    # ── Summary ───────────────────────────────────────────────────────────────
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(OUT_DIR / "downstream_all.csv", index=False)

    print("\n\n" + "=" * 70)
    print("DOWNSTREAM TASK — SUMMARY")
    print("=" * 70)

    for corpus in combined["corpus"].unique():
        sub = combined[combined["corpus"] == corpus].copy()
        sub = sub.set_index("model").reindex(
            [m for m in MODEL_ORDER if m in sub["model"].values]
        )
        print(f"\n{corpus}")
        print(f"  {'Model':<22s}  {'F1-macro':>10s}  {'AUC':>8s}")
        print(f"  {'-'*22}  {'-'*10}  {'-'*8}")
        for model, row in sub.iterrows():
            f1  = f"{row['f1_macro']:.3f}±{row['f1_macro_std']:.3f}"
            auc = f"{row['auc']:.3f}±{row['auc_std']:.3f}" \
                  if not np.isnan(row["auc"]) else "  —  "
            mark = " ◀" if model in ("SCPTM", "SCPTM+θ+MWE") else ""
            print(f"  {model:<22s}  {f1:>10s}  {auc:>8s}{mark}")

    print(f"\nResults written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
