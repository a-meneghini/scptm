"""
benchmark_paper.py
==================
Full benchmark for the paper:
  "Syntax Matters? A Graph-Augmented Variational Topic Model
   for Computational Social Science"

Models
------
  LDA      gensim LdaMulticore               (BoW generative baseline)
  CTM      SCPTM graph_mode="none"            (VAE ablation — no syntax)
  SCPTM    SCPTM graph_mode="filtered"        (VAE + syntactic GNN)
  BERTopic standard BERTopic                  (dense-clustering baseline)

Corpora
-------
  20 Newsgroups   sklearn built-in, K=20 fixed, NMI with ground truth
  UN Debates      HuggingFace "Eugleo/un-general-debates", K ∈ K_RANGE
  Reddit Pol.     CSV from Kaggle (lib/con), K ∈ K_RANGE

Metrics
-------
  NPMI            token co-occurrence coherence (standard, BoW-biased)
  WE-Coherence    avg pairwise cosine sim of top words in SBERT space
  Diversity       fraction of unique words across topic top-word lists
  Quality         NPMI × Diversity  (composite)
  NMI             cluster vs. ground-truth labels  (20NG only)

Design
------
  * SEEDS runs per (model, corpus, K) → report mean ± std
  * SBERT doc/word embeddings cached after first run (edge_cache_path)
  * spaCy parse cached after first run
  * BERTopic reuses pre-computed SBERT embeddings (no double encoding)
  * LDA operates on the same lemmatised vocabulary as CTM/SCPTM
  * Results saved incrementally to OUT_DIR/*.csv (safe for Colab timeouts)

Usage
-----
  python benchmark_paper.py

Google Colab quick-start
------------------------
  from google.colab import drive; drive.mount('/content/drive')
  !pip install torch torch-geometric sentence-transformers spacy \\
               gensim bertopic datasets umap-learn hdbscan -q
  !python -m spacy download en_core_web_sm -q
  !pip install -e "/content/drive/MyDrive/scptm" -q
  # Then set DRIVE_ROOT below and run.
"""

# ── Standard library ─────────────────────────────────────────────────────────
import os
import warnings
from itertools import combinations
from pathlib import Path

# ── Third-party ──────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import normalized_mutual_info_score
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import gensim
import gensim.corpora as corpora
from gensim.models import LdaMulticore

# ── SCPTM ────────────────────────────────────────────────────────────────────
from scptm import SCPTM
from scptm.evaluation import compute_npmi_coherence, compute_topic_diversity
from scptm.graph import prepare_corpus

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — adjust paths before running
# ─────────────────────────────────────────────────────────────────────────────

# Root for cache files and output.
# On Colab: "/content/drive/MyDrive/benchmark_cache"
# Locally : str(Path(__file__).parent / "benchmark_cache")
DRIVE_ROOT = str(Path(__file__).parent / "benchmark_cache")

# Path to the Reddit Lib/Con CSV.
# On Colab: upload reddit_pol.csv to the S-CPTM folder on Drive and set
#   DRIVE_ROOT = "/content/drive/MyDrive/S-CPTM"  (see above).
# The file is then auto-found at DRIVE_ROOT/reddit_pol.csv.
REDDIT_CSV = str(Path(DRIVE_ROOT) / "reddit_pol.csv")

# Shared SBERT model
SBERT_MODEL = "all-MiniLM-L6-v2"

# K values
K_NEWSGROUPS = 20          # fixed — matches 20 ground-truth categories
K_RANGE      = [5, 10, 15, 20, 25, 30]   # swept for UN Debates & Reddit

# Number of independent runs per (model, corpus, K)
SEEDS = [42, 123, 2024]

# Training epochs for CTM / SCPTM
EPOCHS = 50

# Output directory
OUT_DIR = Path(DRIVE_ROOT) / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_we_coherence(topic_words: list, sbert: SentenceTransformer) -> float:
    """
    Word Embedding Coherence: average pairwise cosine similarity between the
    top words of each topic in SBERT embedding space.

    Agnostic to model paradigm — works equally for LDA, CTM, SCPTM, BERTopic.
    Range [0, 1]: higher = more semantically tight topic clusters.
    """
    scores = []
    for words in topic_words:
        words = [w for w in words if w]
        if len(words) < 2:
            continue
        embs = torch.tensor(
            sbert.encode(words, show_progress_bar=False), dtype=torch.float32
        )
        embs = F.normalize(embs, p=2, dim=-1)
        sim  = torch.matmul(embs, embs.T)
        n    = len(words)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        avg   = sum(sim[i, j].item() for i, j in pairs) / max(len(pairs), 1)
        scores.append(avg)
    return float(np.mean(scores)) if scores else 0.0


def evaluate_all(
    topic_words: list,
    bow_ref,
    vocab_ref: list,
    sbert: SentenceTransformer,
    theta=None,
    true_labels=None,
) -> dict:
    """
    Compute all metrics for a fitted model.

    Parameters
    ----------
    topic_words : list[list[str]]   top-K words per topic
    bow_ref     : sparse matrix     reference BoW (same vocab for all models)
    vocab_ref   : list[str]         vocabulary matching bow_ref
    sbert       : SentenceTransformer
    theta       : (n_docs, K) tensor or None  — required for NMI
    true_labels : array or None               — required for NMI
    """
    npmi  = compute_npmi_coherence(topic_words, bow_ref, vocab_ref)
    div   = compute_topic_diversity(topic_words)
    wec   = compute_we_coherence(topic_words, sbert)
    qual  = npmi * div

    out = dict(npmi=npmi, we_coherence=wec, diversity=div, quality=qual)

    if theta is not None and true_labels is not None:
        dominant = np.array(theta).argmax(axis=-1) if not isinstance(theta, np.ndarray) \
                   else theta.argmax(axis=-1)
        out["nmi"] = normalized_mutual_info_score(true_labels, dominant)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Corpus loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_20newsgroups():
    """Load 20 Newsgroups. Returns (docs, labels, label_names).

    Tries sklearn's built-in downloader first (with a socket-level timeout
    override to avoid indefinite hanging).  If that fails with a network error
    (common on Colab / restricted environments) it falls back to the HuggingFace
    Hub mirror via the ``datasets`` library.
    """
    print("\n[Corpus] Loading 20 Newsgroups …")

    # ── Attempt 1: sklearn fetch (raises on 504 / network errors) ─────────────
    try:
        import socket
        _orig_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(60)          # 60-second cap per connection
        try:
            data = fetch_20newsgroups(
                subset="all", remove=("headers", "footers", "quotes")
            )
        finally:
            socket.setdefaulttimeout(_orig_timeout)

        docs   = data.data
        labels = data.target
        print(f"  {len(docs):,} documents, {len(data.target_names)} categories")
        return docs, labels, data.target_names

    except Exception as e:
        print(f"  sklearn download failed ({type(e).__name__}: {e})")
        print("  Falling back to HuggingFace datasets mirror …")

    # ── Attempt 2: HuggingFace mirror ─────────────────────────────────────────
    try:
        from datasets import load_dataset
        ds = load_dataset("SetFit/20_newsgroups", split="train+test")
        label_names = ds.features["label"].names
        docs   = [r["text"]  for r in ds]
        labels = [r["label"] for r in ds]
        import numpy as np
        labels = np.array(labels)
        print(f"  {len(docs):,} documents, {len(label_names)} categories (HF mirror)")
        return docs, labels, label_names
    except Exception as e2:
        raise RuntimeError(
            "Could not load 20 Newsgroups via sklearn or HuggingFace.\n"
            f"  sklearn error  : {e}\n"
            f"  HF error       : {e2}\n\n"
            "On Colab you can pre-download it manually:\n"
            "  from sklearn.datasets import fetch_20newsgroups\n"
            "  fetch_20newsgroups(subset='all', remove=('headers','footers','quotes'))\n"
            "then re-run the benchmark."
        ) from e2


def load_ungdc(max_docs: int = None):
    """Load UN General Debate Corpus from HuggingFace."""
    print("\n[Corpus] Loading UN General Debates …")
    from datasets import load_dataset
    ds   = load_dataset("Eugleo/un-general-debates", split="train")
    docs = [r["text"] for r in ds]
    if max_docs:
        docs = docs[:max_docs]
    print(f"  {len(docs):,} speeches")
    return docs


def load_reddit_politics(csv_path: str):
    """Load Reddit liberals-vs-conservatives CSV.

    Combines the Title and Text fields (Text is NaN for ~81 % of posts which
    are link submissions; Title alone averages only 12.6 words).  The merged
    document is richer and more stable for topic modelling.

    Returns (docs, labels) where labels is a 0/1 array
    (0 = Conservative, 1 = Liberal) aligned with docs.
    """
    print("\n[Corpus] Loading Reddit politics …")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Reddit CSV not found at '{csv_path}'.\n"
            "Upload reddit_pol.csv to your DRIVE_ROOT folder on Google Drive\n"
            "and make sure DRIVE_ROOT points to that folder."
        )
    df = pd.read_csv(csv_path)

    # Build document = Title + body text (where available)
    def _merge(row):
        title = str(row.get("Title", "")).strip()
        body  = str(row.get("Text",  "")).strip()
        if body and body.lower() != "nan":
            return f"{title} {body}".strip()
        return title

    raw_docs   = df.apply(_merge, axis=1).tolist()
    raw_labels = df.get("Political Lean", pd.Series(dtype=str)).fillna("").tolist()

    # Map labels to int (Conservative=0, Liberal=1); unknown → -1
    label_map = {"Conservative": 0, "Liberal": 1}
    docs, labels = [], []
    for doc, lbl in zip(raw_docs, raw_labels):
        if len(doc.split()) >= 5:          # was 10 — relaxed because titles are short
            docs.append(doc)
            labels.append(label_map.get(lbl, -1))

    labels = np.array(labels)
    n_lib  = int((labels == 1).sum())
    n_con  = int((labels == 0).sum())
    print(f"  {len(docs):,} posts (≥5 words) — Liberal: {n_lib:,}  Conservative: {n_con:,}")
    return docs, labels


# ─────────────────────────────────────────────────────────────────────────────
# Reference BoW builder (shared vocabulary for NPMI across all models)
# ─────────────────────────────────────────────────────────────────────────────

def build_reference_bow(docs: list, min_df: int = 5, max_features: int = 20_000):
    """
    Build a reference BoW matrix and vocabulary used consistently
    for NPMI computation across all models.
    Stop words removed; English only.
    """
    vec      = CountVectorizer(
        stop_words  = "english",
        min_df      = min_df,
        max_features= max_features,
        token_pattern= r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    bow      = vec.fit_transform(docs)
    vocab    = vec.get_feature_names_out().tolist()
    return bow, vocab


# ─────────────────────────────────────────────────────────────────────────────
# LDA runner
# ─────────────────────────────────────────────────────────────────────────────

def run_lda(docs: list, K: int, seed: int, min_df: int = 5):
    """Train gensim LDA. Returns (topic_words, theta).

    Speed notes:
    - passes=5 is enough for benchmarking (was 10).
    - chunksize=4000 reduces overhead on large corpora.
    - Theta is extracted via lda.inference() (one vectorised C call)
      instead of a Python loop over every document.
    """
    import os as _os
    n_workers = max(1, min(4, (_os.cpu_count() or 2) - 1))

    # Tokenise with the same stop-word filter as the reference BoW
    vec   = CountVectorizer(stop_words="english", min_df=min_df,
                            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    vec.fit(docs)
    vocab_set  = set(vec.get_feature_names_out().tolist())

    tokenised_raw = [
        [t for t in doc.lower().split() if t in vocab_set]
        for doc in docs
    ]
    # Track which original docs survive the empty-doc filter (needed for NMI)
    doc_mask  = np.array([bool(t) for t in tokenised_raw])
    tokenised = [t for t in tokenised_raw if t]

    dct    = corpora.Dictionary(tokenised)
    corpus = [dct.doc2bow(tok) for tok in tokenised]

    lda = LdaMulticore(
        corpus,
        num_topics   = K,
        id2word      = dct,
        passes       = 5,           # was 10 — 5 is sufficient for benchmarking
        chunksize    = 4000,        # larger chunks → fewer inter-worker syncs
        workers      = n_workers,
        random_state = seed,
    )

    # Top words per topic
    topic_words = [
        [w for w, _ in lda.show_topic(k, topn=10)]
        for k in range(K)
    ]

    # Document-topic distributions (n_docs × K) — vectorised, no Python loop
    # lda.inference() returns (gamma, sstats); gamma shape = (n_docs, K)
    gamma, _ = lda.inference(corpus)
    theta = (gamma / gamma.sum(axis=1, keepdims=True)).astype(np.float32)

    return topic_words, theta, doc_mask


# ─────────────────────────────────────────────────────────────────────────────
# CTM / SCPTM runners
# ─────────────────────────────────────────────────────────────────────────────

def run_scptm_variant(
    docs: list,
    K: int,
    seed: int,
    graph_mode: str,
    cache_path: str,
    min_df: int = 5,
    max_features: int = 20_000,
):
    """
    Train CTM (graph_mode='none') or SCPTM (graph_mode='filtered').
    Returns (topic_words, theta).
    Cache is shared across K/seed — SBERT and spaCy run only once.
    """
    # Mirror the prepare_corpus filter (graph.py line 120) to build doc_mask.
    # SCPTM silently drops docs with ≤10 chars before training; we track which
    # original docs survive so callers can align true_labels for NMI.
    doc_mask = np.array([len(str(d).strip()) > 10 for d in docs])

    model = SCPTM(
        num_topics   = K,
        graph_mode   = graph_mode,
        lang         = "eng",
        epochs       = EPOCHS,
        apply_chunking= False,
        min_df       = min_df,
        max_features = max_features,
        random_state = seed,
        metrics_every_n_epochs = EPOCHS,   # only at final epoch → faster
    )
    model.fit_transform(docs, edge_cache_path=cache_path)

    info        = model.get_topic_info(top_k=10)
    topic_words = [
        [w.strip() for w in row["keywords"].split(",")]
        for _, row in info.sort_values("topic_id").iterrows()
    ]
    theta = model._theta.numpy()   # (n_valid_docs, K)
    return topic_words, theta, doc_mask


# ─────────────────────────────────────────────────────────────────────────────
# BERTopic runner (reuses pre-computed SBERT embeddings)
# ─────────────────────────────────────────────────────────────────────────────

def run_bertopic(
    docs: list,
    K: int,
    seed: int,
    precomputed_embs: np.ndarray,
    min_df: int = 5,
):
    """Train BERTopic. Reuses pre-computed embeddings to avoid double SBERT."""
    sbert = SentenceTransformer(SBERT_MODEL)
    vec   = CountVectorizer(
        stop_words   = "english",
        min_df       = min_df,
        max_features = 20_000,
        token_pattern= r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    bt = BERTopic(
        embedding_model        = sbert,
        vectorizer_model       = vec,
        nr_topics              = K,
        calculate_probabilities= True,
        verbose                = False,
        umap_model             = _make_umap(seed),
    )
    _, probs = bt.fit_transform(docs, embeddings=precomputed_embs)

    topic_words = []
    for tid in sorted(bt.get_topics()):
        if tid == -1:
            continue
        topic_words.append([w for w, _ in bt.get_topic(tid)[:10]])

    # probs shape: (n_docs, n_topics)  or None
    if probs is not None:
        theta = np.array(probs)
        # Handle outlier column (-1 topic) if present
        if theta.shape[1] == len(topic_words) + 1:
            theta = theta[:, 1:]
    else:
        theta = None

    # BERTopic processes every document (no filtering), so all labels are valid
    doc_mask = np.ones(len(docs), dtype=bool)
    return topic_words, theta, doc_mask


def _make_umap(seed):
    """UMAP with fixed seed for reproducibility."""
    try:
        from umap import UMAP
        return UMAP(n_components=5, min_dist=0.0, metric="cosine",
                    random_state=seed)
    except ImportError:
        return None


def precompute_sbert_embeddings(docs: list, cache_path: str) -> np.ndarray:
    """Encode docs with SBERT; cache result as npy to avoid re-encoding."""
    npy = Path(cache_path).with_suffix(".sbert.npy")
    if npy.exists():
        print(f"  [SBERTCache] Loading embeddings from {npy.name}")
        return np.load(str(npy))
    sbert = SentenceTransformer(SBERT_MODEL)
    print("  Encoding documents with SBERT …")
    embs  = sbert.encode(docs, show_progress_bar=True, batch_size=64)
    np.save(str(npy), embs)
    print(f"  [SBERTCache] Saved to {npy.name}")
    return embs


# ─────────────────────────────────────────────────────────────────────────────
# K-sweep for a single corpus
# ─────────────────────────────────────────────────────────────────────────────

def run_corpus_sweep(
    corpus_name: str,
    docs: list,
    k_values: list,
    bow_ref,
    vocab_ref: list,
    sbert: SentenceTransformer,
    cache_path: str,
    sbert_embs: np.ndarray,
    true_labels=None,
    min_df: int = 5,
) -> pd.DataFrame:
    """
    Run all models × all K × all seeds for one corpus.
    Returns a DataFrame with one row per (model, K, seed).
    Results are appended to a CSV after each row for crash recovery.
    """
    out_csv = OUT_DIR / f"raw_{corpus_name}.csv"
    rows    = []

    # Load existing results if resuming
    if out_csv.exists():
        existing = pd.read_csv(out_csv)
        print(f"  Resuming from {len(existing)} existing rows in {out_csv.name}")
    else:
        existing = pd.DataFrame()

    def _already_done(model, K, seed):
        if existing.empty:
            return False
        return not existing[
            (existing.model == model) &
            (existing.K     == K)     &
            (existing.seed  == seed)
        ].empty

    for K in k_values:
        for seed in SEEDS:
            for model_name in ["LDA", "CTM", "SCPTM", "BERTopic"]:

                if _already_done(model_name, K, seed):
                    print(f"  [skip] {model_name} K={K} seed={seed} (already done)")
                    continue

                print(f"\n  ── {model_name}  K={K}  seed={seed} ──")
                theta    = None
                doc_mask = None
                try:
                    if model_name == "LDA":
                        topic_words, theta, doc_mask = run_lda(docs, K, seed, min_df)

                    elif model_name == "CTM":
                        topic_words, theta, doc_mask = run_scptm_variant(
                            docs, K, seed, "none", cache_path, min_df)

                    elif model_name == "SCPTM":
                        topic_words, theta, doc_mask = run_scptm_variant(
                            docs, K, seed, "filtered", cache_path, min_df)

                    elif model_name == "BERTopic":
                        topic_words, theta, doc_mask = run_bertopic(
                            docs, K, seed, sbert_embs, min_df)

                    # Subset true_labels to only the docs the model actually saw,
                    # avoiding "inconsistent number of samples" NMI errors.
                    aligned_labels = None
                    if true_labels is not None and doc_mask is not None:
                        aligned_labels = np.array(true_labels)[doc_mask]
                    elif true_labels is not None:
                        aligned_labels = true_labels

                    metrics = evaluate_all(
                        topic_words, bow_ref, vocab_ref, sbert,
                        theta=theta, true_labels=aligned_labels,
                    )

                except Exception as exc:
                    print(f"  [ERROR] {model_name} K={K} seed={seed}: {exc}")
                    metrics = dict(npmi=np.nan, we_coherence=np.nan,
                                   diversity=np.nan, quality=np.nan)

                # Always include nmi so every CSV row has the same columns.
                # Rows where NMI wasn't computed get NaN rather than a missing field,
                # which prevents pandas from misreading the CSV on the next restart.
                metrics.setdefault("nmi", np.nan)

                row = dict(corpus=corpus_name, model=model_name, K=K, seed=seed,
                           **metrics)
                rows.append(row)

                # Append to CSV immediately (crash-safe)
                pd.DataFrame([row]).to_csv(
                    out_csv, mode="a",
                    header=not out_csv.exists() or os.stat(out_csv).st_size == 0,
                    index=False,
                )
                print(f"    NPMI={metrics['npmi']:.3f}  "
                      f"WE-Coh={metrics['we_coherence']:.3f}  "
                      f"Div={metrics['diversity']:.3f}  "
                      f"Qual={metrics['quality']:.3f}"
                      + (f"  NMI={metrics.get('nmi', np.nan):.3f}"
                         if "nmi" in metrics else ""))

    # Merge with existing results
    all_rows = pd.read_csv(out_csv)
    return all_rows


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation and plotting
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_results(raw: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std over seeds for each (corpus, model, K)."""
    metrics = ["npmi", "we_coherence", "diversity", "quality", "nmi"]
    metrics = [m for m in metrics if m in raw.columns]
    agg = (
        raw.groupby(["corpus", "model", "K"])[metrics]
        .agg(["mean", "std"])
        .round(4)
    )
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    return agg.reset_index()


def build_main_table(agg: pd.DataFrame) -> pd.DataFrame:
    """
    For each corpus select K* = argmax Quality_mean per model, then
    return one row per (corpus, model) with all metrics at K*.
    20NG is always K=20.
    """
    rows = []
    for corpus in agg["corpus"].unique():
        sub = agg[agg["corpus"] == corpus]
        for model in sub["model"].unique():
            msub = sub[sub["model"] == model]
            if corpus == "20NG":
                best = msub[msub["K"] == K_NEWSGROUPS].iloc[0]
            else:
                best = msub.loc[msub["quality_mean"].idxmax()]
            rows.append(best)
    return pd.DataFrame(rows)


def plot_k_curves(agg: pd.DataFrame, metric: str = "quality_mean"):
    """Line plot of metric vs K for each model, one panel per corpus."""
    corpora  = [c for c in agg["corpus"].unique() if c != "20NG"]
    n        = len(corpora)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]
    colours = {"LDA": "#E07B54", "CTM": "#4C72B0",
               "SCPTM": "#C44E52", "BERTopic": "#CCB974"}

    for ax, corpus in zip(axes, corpora):
        sub = agg[agg["corpus"] == corpus]
        for model in ["LDA", "CTM", "SCPTM", "BERTopic"]:
            ms = sub[sub["model"] == model].sort_values("K")
            if ms.empty:
                continue
            ax.plot(ms["K"], ms[metric], marker="o",
                    label=model, color=colours.get(model))
            if f"{metric.replace('_mean','')}_std" in ms.columns:
                std_col = metric.replace("_mean", "_std")
                ax.fill_between(
                    ms["K"],
                    ms[metric] - ms[std_col],
                    ms[metric] + ms[std_col],
                    alpha=0.12, color=colours.get(model),
                )
        ax.set_title(corpus, fontsize=11, fontweight="bold")
        ax.set_xlabel("Number of topics (K)")
        ax.set_ylabel(metric.replace("_mean", "").replace("_", " ").title())
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Topic Quality vs. K — mean ± std across 3 seeds",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = OUT_DIR / "k_curves.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved K-curve plot → {path}")


def plot_qualitative(
    docs: list,
    K: int,
    seed: int,
    cache_path: str,
    sbert_embs: np.ndarray,
    corpus_name: str = "UN Debates",
    min_df: int = 5,
):
    """
    Side-by-side top-10 word grids for CTM vs SCPTM at the best K.
    Saved to results/qualitative_<corpus>.png
    """
    import math, matplotlib.patches as mpatches

    results = {}
    for model_name, mode in [("CTM", "none"), ("SCPTM", "filtered")]:
        tw, _ = run_scptm_variant(docs, K, seed, mode, cache_path, min_df)
        results[model_name] = {i + 1: tw[i] for i in range(len(tw))}

    fig, all_axes = plt.subplots(
        K, 2, figsize=(10, K * 1.6),
        gridspec_kw={"wspace": 0.05}
    )
    colours = {"CTM": "#4C72B0", "SCPTM": "#C44E52"}

    for col, (mname, topics) in enumerate(results.items()):
        for row, (tid, words) in enumerate(sorted(topics.items())):
            ax = all_axes[row, col]
            ax.add_patch(mpatches.FancyBboxPatch(
                (0, 0.85), 1, 0.15, boxstyle="square,pad=0",
                transform=ax.transAxes, color=colours[mname], zorder=2,
            ))
            ax.text(0.5, 0.92, f"T{tid}", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8,
                    fontweight="bold", color="white", zorder=3)
            ax.text(0.04, 0.80,
                    "\n".join(f"{j+1}. {w}" for j, w in enumerate(words[:10])),
                    transform=ax.transAxes, va="top", fontsize=6.8,
                    fontfamily="monospace", linespacing=1.4)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        all_axes[0, col].set_title(
            mname, fontsize=12, fontweight="bold",
            color=colours[mname], pad=10,
        )

    fig.suptitle(
        f"Qualitative comparison — {corpus_name} (K={K}, seed={seed})",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    slug = corpus_name.lower().replace(" ", "_")
    path = OUT_DIR / f"qualitative_{slug}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved qualitative figure → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    sbert = SentenceTransformer(SBERT_MODEL)
    all_raw = []

    # ── 1. 20 Newsgroups ─────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("CORPUS 1/3 — 20 Newsgroups  (K=20, with NMI ground truth)")
    print("=" * 65)

    ng_docs, ng_labels, _ = load_20newsgroups()
    ng_bow, ng_vocab       = build_reference_bow(ng_docs, min_df=5)
    ng_cache               = str(Path(DRIVE_ROOT) / "20ng_cache.pkl")
    ng_embs                = precompute_sbert_embeddings(
                                ng_docs, str(Path(DRIVE_ROOT) / "20ng"))

    raw_ng = run_corpus_sweep(
        "20NG", ng_docs, [K_NEWSGROUPS],
        ng_bow, ng_vocab, sbert,
        ng_cache, ng_embs,
        true_labels=ng_labels, min_df=5,
    )
    all_raw.append(raw_ng)

    # ── 2. UN General Debates ────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("CORPUS 2/3 — UN General Debates  (K-sweep)")
    print("=" * 65)

    un_docs  = load_ungdc()
    un_bow, un_vocab = build_reference_bow(un_docs, min_df=10)
    un_cache         = str(Path(DRIVE_ROOT) / "ungdc_cache.pkl")
    un_embs          = precompute_sbert_embeddings(
                          un_docs, str(Path(DRIVE_ROOT) / "ungdc"))

    raw_un = run_corpus_sweep(
        "UN_Debates", un_docs, K_RANGE,
        un_bow, un_vocab, sbert,
        un_cache, un_embs,
        min_df=10,
    )
    all_raw.append(raw_un)

    # ── 3. Reddit politics ───────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("CORPUS 3/3 — Reddit Liberals vs Conservatives  (K-sweep)")
    print("=" * 65)

    rd_docs, rd_labels = load_reddit_politics(REDDIT_CSV)
    rd_bow, rd_vocab   = build_reference_bow(rd_docs, min_df=5)
    rd_cache           = str(Path(DRIVE_ROOT) / "reddit_cache.pkl")
    rd_embs            = precompute_sbert_embeddings(
                            rd_docs, str(Path(DRIVE_ROOT) / "reddit"))

    raw_rd = run_corpus_sweep(
        "Reddit_Pol", rd_docs, K_RANGE,
        rd_bow, rd_vocab, sbert,
        rd_cache, rd_embs,
        true_labels=rd_labels, min_df=5,
    )
    all_raw.append(raw_rd)

    # ── Aggregation and output ───────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("AGGREGATING RESULTS")
    print("=" * 65)

    full_raw = pd.concat(all_raw, ignore_index=True)
    full_raw.to_csv(OUT_DIR / "all_raw.csv", index=False)

    agg = aggregate_results(full_raw)
    agg.to_csv(OUT_DIR / "aggregated.csv", index=False)

    main_table = build_main_table(agg)
    main_table.to_csv(OUT_DIR / "main_table.csv", index=False)

    print("\n── Main table (K* per model per corpus) ──")
    cols = ["corpus", "model", "K",
            "npmi_mean", "we_coherence_mean", "diversity_mean",
            "quality_mean", "nmi_mean"]
    cols = [c for c in cols if c in main_table.columns]
    print(main_table[cols].to_string(index=False))

    # ── K-curve plot ─────────────────────────────────────────────────────────
    plot_k_curves(agg, metric="quality_mean")
    plot_k_curves(agg, metric="we_coherence_mean")

    # ── Qualitative: CTM vs SCPTM on UN Debates at K* ────────────────────────
    un_agg    = agg[agg["corpus"] == "UN_Debates"]
    scptm_agg = un_agg[un_agg["model"] == "SCPTM"]
    k_star    = int(scptm_agg.loc[scptm_agg["quality_mean"].idxmax(), "K"])
    print(f"\nQualitative plot: UN Debates at K*={k_star}")
    plot_qualitative(un_docs, k_star, SEEDS[0], un_cache, un_embs,
                     corpus_name="UN Debates", min_df=10)

    print(f"\nAll outputs saved to {OUT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
