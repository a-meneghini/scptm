"""
benchmark_paper.py
==================
Full benchmark for the paper:
  "Syntax Matters? A Graph-Augmented Variational Topic Model
   for Computational Social Science"

Models  (7 — full ablation + honest baselines)
------
  LDA            gensim LdaMulticore                  (BoW generative baseline)
  CTM            CombinedTM (contextualized_topic_models, Bianchi 2021)
                                                       (true neural generative baseline)
  SCPTM-none     SCPTM graph_mode="none"              (ablation: VAE + ctx-beta, no graph)
  SCPTM-nosyn    SCPTM graph_mode="no_syntax"         (ablation: + doc-word edges)
  SCPTM-fulldep  SCPTM graph_mode="full_dep"          (ablation: + all dependency edges)
  SCPTM          SCPTM graph_mode="filtered"          (full model: informative syntax)
  BERTopic       standard BERTopic                    (dense-clustering upper bound)

Corpora
-------
  20 Newsgroups   sklearn built-in, K=20 fixed, NMI with ground truth
  EU Debates      HuggingFace "coastalcph/eu_debates" (English only), K ∈ K_RANGE
  Reddit Pol.     CSV from Kaggle (lib/con), K ∈ K_RANGE
  Hate Speech     UC Berkeley D-Lab, K=8 + sweep, NMI with target categories

Metrics  (see paper_metrics.py)
-------
  A. Predictive    perplexity proxy (held-out completion)
  B. Quality       C_V, C_NPMI (gensim/Röder), Topic Diversity, intra-topic
                   embedding concentration, NMI vs. ground-truth labels
                   (where labels exist: 20NG, HateSpeech, Reddit_Pol)
                   Secondary (Supplementary Material): topic exclusivity,
                   CD score, JS-divergence, WE-coherence
  C. MWE           MWE specificity (IDF), unigram complementarity,
                   semantic compactness, content ratio
                   (only for the 4 SCPTM graph modes)
  D. Discourse     MWE-vs-unigram valence gap (VADER), stance concentration

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
from gensim.models import LdaMulticore, Phrases
from gensim.models.phrases import Phraser

# ── SCPTM ────────────────────────────────────────────────────────────────────
from scptm import SCPTM
from scptm.evaluation import compute_npmi_coherence, compute_topic_diversity
from scptm.graph import prepare_corpus

# ── Paper metrics (families A–D) ──────────────────────────────────────────────
import paper_metrics as pm

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

# Optional local CSV fallback for UN General Debates.
# If the HuggingFace mirror is unavailable, download the CSV from Kaggle:
#   https://www.kaggle.com/datasets/unitednations/un-general-debates
# and upload it to DRIVE_ROOT.  Leave as None to rely on HuggingFace.
UNGDC_CSV = str(Path(DRIVE_ROOT) / "un-general-debates.csv")

# Shared SBERT model
SBERT_MODEL = "all-MiniLM-L6-v2"

# K values
K_NEWSGROUPS = 20          # fixed — matches 20 ground-truth categories
K_RANGE      = [5, 10, 15, 20, 25, 30]   # swept for UN Debates & Reddit

# Number of independent runs per (model, corpus, K)
SEEDS = [42, 123, 2024]

# Training epochs for CTM / SCPTM
EPOCHS = 50

# Top-k words saved per topic — identical for every model
TOP_K_WORDS = 10

# ── Model roster (7) ──────────────────────────────────────────────────────────
# Maps benchmark model name → SCPTM graph_mode (None for non-SCPTM models).
SCPTM_MODES = {
    "SCPTM-none":    "none",        # ablation: VAE + ctx-beta, no graph
    "SCPTM-nosyn":   "no_syntax",   # ablation: + doc-word edges
    "SCPTM-fulldep": "full_dep",    # ablation: + all dependency edges
    "SCPTM":         "filtered",    # full model: informative syntactic edges
}
MODEL_ORDER = ["LDA", "CTM", "SCPTM-none", "SCPTM-nosyn",
               "SCPTM-fulldep", "SCPTM", "BERTopic"]

# Canonical metric schema — EVERY raw-CSV row carries exactly these columns
# (missing values → NaN) so models with different metric sets (e.g. only the
# SCPTM modes emit mwe_* / valence) never misalign the appended CSV header.
METRIC_COLUMNS = [
    "perplexity",                                              # A
    "npmi", "c_v", "c_npmi", "we_coherence", "diversity", "quality",
    "exclusivity", "cd_score", "js_divergence", "between_topic_cosine",
    "intra_topic_concentration",                              # B
    "mwe_compactness", "mwe_specificity", "mwe_complementarity",
    "mwe_content_ratio",                                      # C
    "mwe_valence", "unigram_valence", "valence_gap",
    "stance_concentration",                                   # D
    "nmi",                                                    # extrinsic
]

# ── RQ3-D stance classifier ───────────────────────────────────────────────────
# Heavy: a pre-trained classifier is run over each topic's dominant documents.
# To bound cost it runs only on the seeds listed here (others get NaN, which
# nanmean aggregation handles).  VADER valence (RQ3-C) is cheap and always on.
RUN_STANCE   = True
STANCE_SEEDS = [42]
STANCE_MODEL = "cardiffnlp/twitter-roberta-base-hate-latest"

# sklearn English stop words used as a final safety filter on all top-word lists.
# Each runner already filters internally (CountVectorizer or spaCy), but the
# filter sets differ slightly; this shared post-processing guarantees consistency.
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as _SKL_STOPS
_STOP_SET = set(_SKL_STOPS) | {"said", "says", "also", "would", "could", "may"}

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
        # Replace underscores with spaces so SBERT encodes bigrams correctly
        # ("climate_change" → "climate change")
        words_for_sbert = [w.replace("_", " ") for w in words]
        embs = torch.tensor(
            sbert.encode(words_for_sbert, show_progress_bar=False), dtype=torch.float32
        )
        embs = F.normalize(embs, p=2, dim=-1)
        sim  = torch.matmul(embs, embs.T)
        n    = len(words)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        avg   = sum(sim[i, j].item() for i, j in pairs) / max(len(pairs), 1)
        scores.append(avg)
    return float(np.mean(scores)) if scores else 0.0


def _topic_vectors(topic_words: list, sbert: SentenceTransformer) -> np.ndarray:
    """
    Model-agnostic topic embedding = mean SBERT vector of a topic's top words.
    Used for between-topic cosine and MWE compactness so every model — even
    LDA / BERTopic that have no embedding-space topic vector — is comparable.
    """
    vecs = []
    for words in topic_words:
        words = [w.replace("_", " ") for w in words if w]
        if not words:
            vecs.append(None)
            continue
        e = np.asarray(sbert.encode(words, show_progress_bar=False), dtype=np.float64)
        vecs.append(e.mean(axis=0))
    dim = next((v.shape[0] for v in vecs if v is not None), 384)
    return np.vstack([v if v is not None else np.zeros(dim) for v in vecs])


def evaluate_all(
    topic_words: list,
    bow_ref,
    vocab_ref: list,
    sbert: SentenceTransformer,
    *,
    tokenized_texts: list,
    df_lookup: dict,
    theta=None,
    doc_mask=None,
    doc_embeddings_full=None,
    true_labels=None,
    mwe_per_topic=None,
    single_per_topic=None,
    raw_docs=None,
    run_stance: bool = False,
) -> dict:
    """
    Compute all four metric families for a fitted model.

    Topic-quality metrics that need a per-topic word distribution use a shared
    ``pseudo_beta`` built from (theta, reference BoW), so all models are scored
    identically.  When ``doc_mask`` drops documents (SCPTM length filter), both
    the BoW rows and document embeddings are sliced to stay aligned with theta.

    Parameters
    ----------
    topic_words        : list[list[str]]   top-K unigrams per topic
    bow_ref            : sparse (N, V)      reference BoW (shared vocab)
    vocab_ref          : list[str]          vocabulary matching bow_ref
    tokenized_texts    : list[list[str]]    unigram-tokenised corpus (for gensim)
    df_lookup          : {token: doc_freq}  for MWE specificity
    theta              : (n, K) or None      doc-topic proportions
    doc_mask           : bool array or None  which docs the model scored
    doc_embeddings_full: (N, D) or None      SBERT doc embeddings (full corpus)
    mwe_per_topic      : list[list[str]] or None   MWE phrases (SCPTM modes only)
    single_per_topic   : list[list[str]] or None   single-word keywords
    raw_docs           : list[str] or None   raw documents (stance classifier)
    run_stance         : bool                whether to run the RQ3-D classifier
    """
    n_docs_full = bow_ref.shape[0]

    # ── Align BoW rows / doc embeddings / raw docs with theta ────────────────
    # SCPTM (apply_chunking=False) drops docs with ≤10 chars, so its theta has
    # mask.sum() rows; LDA/CTM/BERTopic score all docs (mask all-True).  We pick
    # the alignment by matching theta's actual row count, which is bullet-proof
    # against rare whitespace edge cases in the length filter.
    theta_np = pm._to_numpy(theta)
    mask = np.asarray(doc_mask, dtype=bool) if doc_mask is not None else None
    if theta_np is not None and mask is not None and theta_np.shape[0] == int(mask.sum()) \
            and int(mask.sum()) != n_docs_full:
        bow_aligned = bow_ref[mask]
        emb_aligned = doc_embeddings_full[mask] if doc_embeddings_full is not None else None
        docs_aligned = ([d for d, m in zip(raw_docs, mask) if m]
                        if raw_docs is not None else None)
    else:
        bow_aligned, emb_aligned, docs_aligned = bow_ref, doc_embeddings_full, raw_docs

    # ── B. Topic quality ─────────────────────────────────────────────────────
    coh = pm.gensim_coherence(topic_words, tokenized_texts, ("c_v", "c_npmi"))
    div = pm.topic_diversity(topic_words)
    wec = compute_we_coherence(topic_words, sbert)

    topic_vecs = _topic_vectors(topic_words, sbert)
    topic_sim  = pm.between_topic_cosine(topic_vecs)            # lower = better

    pb = pm.pseudo_beta_from_theta(theta, bow_aligned)
    umass, excl = pm.per_topic_umass_and_exclusivity(pb, bow_aligned)
    cd          = pm.cd_score(umass, excl) if umass is not None else float("nan")
    exclusivity = float(np.nanmean(excl)) if excl is not None else float("nan")
    js          = pm.topic_js_divergence(pb)
    intra       = pm.intra_topic_concentration(theta, emb_aligned)

    # ── A. Predictive ─────────────────────────────────────────────────────────
    ppl = pm.perplexity_proxy(theta, pb, bow_aligned)

    # Composite quality for K* selection: C_NPMI × diversity (Borčin-style).
    qual = (coh["c_npmi"] * div
            if not (np.isnan(coh["c_npmi"]) or np.isnan(div)) else float("nan"))

    out = dict(
        # legacy in-corpus NPMI kept for continuity with earlier runs
        npmi=compute_npmi_coherence(topic_words, bow_ref, vocab_ref),
        c_v=coh["c_v"], c_npmi=coh["c_npmi"],
        we_coherence=wec, diversity=div, quality=qual,
        exclusivity=exclusivity, cd_score=cd,
        js_divergence=js, between_topic_cosine=topic_sim,
        intra_topic_concentration=intra, perplexity=ppl,
    )

    # ── C. MWE quality (SCPTM modes only) ─────────────────────────────────────
    if mwe_per_topic is not None:
        out["mwe_compactness"]      = pm.mwe_semantic_compactness(mwe_per_topic, topic_vecs, sbert)
        out["mwe_specificity"]      = pm.mwe_specificity(mwe_per_topic, df_lookup, n_docs_full)
        out["mwe_complementarity"]  = pm.mwe_unigram_complementarity(mwe_per_topic, single_per_topic, sbert)
        out["mwe_content_ratio"]    = pm.mwe_content_ratio(mwe_per_topic)
        # ── D. RQ3-C valence gap ──────────────────────────────────────────────
        val = pm.mwe_vs_unigram_valence(mwe_per_topic, single_per_topic)
        out.update(val)

    # ── D. RQ3-D stance concentration ─────────────────────────────────────────
    if run_stance and theta is not None and docs_aligned is not None:
        out["stance_concentration"] = pm.stance_concentration(
            theta, docs_aligned, model_name=STANCE_MODEL)

    # ── NMI ────────────────────────────────────────────────────────────────────
    if theta_np is not None and true_labels is not None:
        dominant = theta_np.argmax(axis=-1)
        labels = np.asarray(true_labels)
        if len(dominant) == len(labels):
            out["nmi"] = normalized_mutual_info_score(labels, dominant)

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
    """Load UN General Debate Corpus.

    Resolution order:
    1. Local CSV at UNGDC_CSV (upload from Kaggle to DRIVE_ROOT — fastest).
    2. HuggingFace mirror "Eugleo/un-general-debates".
    3. HuggingFace mirror "joelniklaus/un-general-debates".

    The Kaggle CSV has columns: session, year, country, text.
    Download from: https://www.kaggle.com/datasets/unitednations/un-general-debates
    """
    print("\n[Corpus] Loading UN General Debates …")

    # ── 1. Local CSV (preferred on Colab — no network dependency) ────────────
    if os.path.exists(UNGDC_CSV):
        print(f"  Loading from local CSV: {UNGDC_CSV}")
        df   = pd.read_csv(UNGDC_CSV)
        text_col = next(
            (c for c in df.columns if c.lower() in ("text", "speech", "statement")),
            df.columns[-1],
        )
        docs = df[text_col].dropna().astype(str).tolist()
        docs = [d for d in docs if len(d.split()) >= 20]
        if max_docs:
            docs = docs[:max_docs]
        print(f"  {len(docs):,} speeches (local CSV)")
        return docs

    # ── 2–3. HuggingFace mirrors ──────────────────────────────────────────────
    from datasets import load_dataset
    hf_paths = [
        ("Eugleo/un-general-debates",     "train", "text"),
        ("joelniklaus/un-general-debates", "train", "text"),
    ]
    last_err = None
    for hf_path, split, text_field in hf_paths:
        try:
            print(f"  Trying HuggingFace: {hf_path} …")
            ds   = load_dataset(hf_path, split=split)
            docs = [r[text_field] for r in ds]
            if max_docs:
                docs = docs[:max_docs]
            print(f"  {len(docs):,} speeches ({hf_path})")
            return docs
        except Exception as e:
            print(f"  ✗ {hf_path}: {e}")
            last_err = e

    raise RuntimeError(
        "Could not load UN General Debates from any source.\n\n"
        "Download the CSV from Kaggle and upload it to your DRIVE_ROOT folder:\n"
        "  https://www.kaggle.com/datasets/unitednations/un-general-debates\n"
        f"Expected path: {UNGDC_CSV}\n"
        f"Last error: {last_err}"
    )


def load_eu_debates(max_docs: int = 20_000, seed: int = 42):
    """
    Load the European Parliament debates corpus (coastalcph/eu_debates,
    Chalkidis & Brandl 2024), keeping ENGLISH-NATIVE speeches only.

    Rationale: ~47% of speeches are originally in English; the rest are
    machine-translated (EasyNMT / M2M-100), which would inject translation
    noise into the syntactic-parse and MWE analysis.  We therefore keep only
    rows whose original language is English — heuristically, those where
    ``translated_text`` is empty (no translation was needed).

    A random subsample of ``max_docs`` is taken for compute tractability.

    Returns
    -------
    docs : list[str]
    """
    print("\n[Corpus] Loading EU Debates (English-native only) …")
    from datasets import load_dataset

    ds = load_dataset("coastalcph/eu_debates", split="train")
    df = ds.to_pandas()

    def _is_english_native(row) -> bool:
        tt = row.get("translated_text", None)
        return tt is None or (isinstance(tt, str) and tt.strip() == "")

    df = df[df.apply(_is_english_native, axis=1)].copy()
    docs = df["text"].dropna().astype(str).tolist()
    docs = [d for d in docs if len(d.split()) >= 30]   # drop procedural one-liners

    if max_docs and len(docs) > max_docs:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(docs), size=max_docs, replace=False)
        docs = [docs[i] for i in sorted(idx)]

    print(f"  {len(docs):,} English-native speeches (≥30 words)")
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


def load_hate_speech(min_tokens: int = 30):
    """
    Load UC Berkeley D-Lab Measuring Hate Speech dataset.
    https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech

    The dataset has one row per (comment, annotator) pair.  We:
      1. Deduplicate by comment_id, keeping the text and averaging binary
         target annotations across annotators (majority vote ≥ 0.5).
      2. Assign a discrete label via priority order on target categories,
         used as ground truth for NMI evaluation.
      3. Filter out very short comments (< min_tokens tokens).

    Label priority:
      0 race          target_race ≥ 0.5
      1 religion      target_religion ≥ 0.5
      2 gender        target_gender ≥ 0.5
      3 sexuality     target_sexuality ≥ 0.5
      4 origin        target_origin ≥ 0.5
      5 age/disab.    target_age ≥ 0.5  OR  target_disability ≥ 0.5
      6 hate/no-tgt   hate_speech_score > 0  (hateful but no specific target)
      7 neutral       everything else

    Returns
    -------
    docs        : list[str]
    labels      : np.ndarray[int]
    label_names : list[str]
    """
    print("\n[Corpus] Loading Measuring Hate Speech (UC Berkeley D-Lab)…")

    from datasets import load_dataset

    ds = load_dataset("ucberkeley-dlab/measuring-hate-speech", split="train")
    df = ds.to_pandas()

    # ── Target columns to aggregate ──────────────────────────────────────────
    TARGET_COLS = [
        "target_race", "target_religion", "target_gender",
        "target_sexuality", "target_origin", "target_age", "target_disability",
    ]
    available_targets = [c for c in TARGET_COLS if c in df.columns]

    # Aggregate: text = first occurrence; targets = mean across annotators
    agg = {"text": "first"}
    for col in available_targets:
        agg[col] = "mean"
    if "hate_speech_score" in df.columns:
        agg["hate_speech_score"] = "mean"

    cdf = df.groupby("comment_id").agg(agg).reset_index()

    # ── Label assignment (priority order) ────────────────────────────────────
    label_names = [
        "race", "religion", "gender", "sexuality",
        "origin", "age_disability", "hate_no_target", "neutral",
    ]

    def _label(row):
        thr = 0.5
        for i, col in enumerate(["target_race", "target_religion",
                                  "target_gender", "target_sexuality",
                                  "target_origin"]):
            if col in row and row[col] >= thr:
                return i
        # age or disability combined → index 5
        age_d = max(row.get("target_age", 0), row.get("target_disability", 0))
        if age_d >= thr:
            return 5
        # hateful but no specific target
        if row.get("hate_speech_score", 0) > 0:
            return 6
        return 7   # neutral

    cdf["label"] = cdf.apply(_label, axis=1)

    # ── Length filter ─────────────────────────────────────────────────────────
    cdf["n_tok"] = cdf["text"].apply(lambda x: len(str(x).split()))
    cdf = cdf[cdf["n_tok"] >= min_tokens].copy()

    docs   = cdf["text"].astype(str).tolist()
    labels = cdf["label"].values

    # ── Summary ───────────────────────────────────────────────────────────────
    from collections import Counter
    counts = Counter(labels)
    for i, name in enumerate(label_names):
        n = counts.get(i, 0)
        if n:
            print(f"  {name:20s}: {n:5,} comments")
    print(f"  {'TOTAL':20s}: {len(docs):5,} comments (≥{min_tokens} tokens)")

    return docs, labels, label_names


# ─────────────────────────────────────────────────────────────────────────────
# Reference BoW builder (shared vocabulary for NPMI across all models)
# ─────────────────────────────────────────────────────────────────────────────

def build_bigram_corpus(
    docs: list,
    min_count: int = 5,
    threshold: float = 10.0,
):
    """
    Detect frequent bigrams with gensim Phrases and return:
      - bigram_texts : list[str]  — docs with bigrams joined by underscore
                                    (e.g. "climate change" → "climate_change")
      - phraser      : Phraser    — reusable transformer for new docs

    All models that support multi-word tokens (LDA, reference BoW) use
    this shared bigram vocabulary so comparisons remain fair.
    SCPTM uses its own syntactic MWE extraction and is not affected.
    """
    tokenized = [doc.lower().split() for doc in docs]
    phrases   = Phrases(tokenized, min_count=min_count, threshold=threshold,
                        connector_words=gensim.models.phrases.ENGLISH_CONNECTOR_WORDS)
    phraser   = Phraser(phrases)
    bigram_tokens = [phraser[toks] for toks in tokenized]
    # Re-join as strings so CountVectorizer can process them
    bigram_texts  = [" ".join(toks) for toks in bigram_tokens]
    n_bigrams = sum(1 for toks in bigram_tokens for t in toks if "_" in t)
    print(f"  [Bigrams] {n_bigrams:,} bigram tokens detected across corpus")
    return bigram_texts, phraser


def build_reference_bow(docs: list, min_df: int = 5, max_features: int = 30_000):
    """
    Build a reference BoW matrix and vocabulary used consistently for NPMI
    computation across all models.

    Accepts both plain docs and bigram-enriched docs (with underscore tokens).
    The token_pattern allows underscores so bigrams like climate_change are
    treated as single vocabulary items.
    """
    vec   = CountVectorizer(
        stop_words   = "english",
        min_df       = min_df,
        max_features = max_features,
        token_pattern= r"(?u)\b[a-zA-Z][a-zA-Z_]+\b",
    )
    bow   = vec.fit_transform(docs)
    vocab = vec.get_feature_names_out().tolist()
    return bow, vocab


# ─────────────────────────────────────────────────────────────────────────────
# Shared top-word post-processing
# ─────────────────────────────────────────────────────────────────────────────

def _clean_topic_words(topic_words: list, top_k: int = TOP_K_WORDS) -> list:
    """
    Uniform post-processing applied to every model's top-word lists:
      - strip whitespace
      - drop empty strings and single-character tokens
      - remove stop words that slipped through model-internal filters
        (spaCy list ≠ sklearn list; both are applied here)
      - truncate / pad to exactly `top_k` entries so all models are comparable
    """
    cleaned = []
    for words in topic_words:
        kept = [
            w.strip() for w in words
            if w.strip() and len(w.strip()) > 1 and w.strip().lower() not in _STOP_SET
        ]
        cleaned.append(kept[:top_k])
    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# LDA runner
# ─────────────────────────────────────────────────────────────────────────────

def run_lda(docs: list, K: int, seed: int, min_df: int = 5,
            top_k: int = TOP_K_WORDS, bigram_texts: list = None):
    """Train gensim LDA on a bigram-enriched corpus.

    Parameters
    ----------
    bigram_texts : list[str] | None
        Pre-built bigram corpus (from build_bigram_corpus). When provided,
        LDA trains on unigram+bigram tokens so topic words can include
        multi-word expressions like 'climate_change'. When None, falls back
        to plain unigram tokenisation.
    """
    import os as _os
    n_workers = max(1, min(4, (_os.cpu_count() or 2) - 1))

    # Use bigram-enriched text when available, otherwise plain docs
    source = bigram_texts if bigram_texts is not None else docs

    # Tokenise allowing underscores so bigrams are treated as single tokens
    vec = CountVectorizer(
        stop_words   = "english",
        min_df       = min_df,
        token_pattern= r"(?u)\b[a-zA-Z][a-zA-Z_]+\b",
    )
    vec.fit(source)
    vocab_set = set(vec.get_feature_names_out().tolist())

    tokenised_raw = [
        [t for t in doc.lower().split() if t in vocab_set]
        for doc in source
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
    topic_words = _clean_topic_words([
        [w for w, _ in lda.show_topic(k, topn=top_k + 5)]   # fetch a few extra, clean, then trim
        for k in range(K)
    ], top_k=top_k)

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
    top_k: int = TOP_K_WORDS,
):
    """
    Train an SCPTM variant at the given ``graph_mode`` (one of
    none / no_syntax / full_dep / filtered).

    Returns (topic_words, theta, doc_mask, mwe_phrases, single_phrases).

    MWEs are extracted for EVERY mode (including 'none') using that mode's
    learned representation, so the ablation measures the syntactic graph's
    contribution to MWE quality directly.

    NOTE — the parse cache stores MODE-DEPENDENT edge lists (filtered by the
    active dependency set) without dependency labels, so it CANNOT be shared
    across graph modes: a cache written by one mode would silently impose its
    edges on another and corrupt the ablation.  We therefore give each mode its
    own cache file.  The spaCy parse is re-run once per (corpus, mode), which is
    negligible amortised over the K-sweep × seeds that reuse it.
    """
    doc_mask = np.array([len(str(d).strip()) > 10 for d in docs])

    mode_cache = (
        cache_path.replace(".pkl", f".{graph_mode}.pkl")
        if cache_path else None
    )

    model = SCPTM(
        num_topics   = K,
        graph_mode   = graph_mode,
        lang         = "eng",
        epochs       = EPOCHS,
        apply_chunking= False,
        min_df       = min_df,
        max_features = max_features,
        random_state = seed,
        metrics_every_n_epochs = EPOCHS,
        use_neighbor_sampling  = False,
    )
    model.fit_transform(docs, edge_cache_path=mode_cache)

    info        = model.get_topic_info(top_k=top_k + 5)
    topic_words = _clean_topic_words([
        [w.strip() for w in row["keywords"].split(",")]
        for _, row in info.sort_values("topic_id").iterrows()
    ], top_k=top_k)
    theta = model._theta.numpy()

    # Extract syntactic MWEs + single words for ALL modes (incl. 'none'):
    # candidates come from spaCy parsing (mode-independent); the topic→phrase
    # ranking uses THIS mode's learned representation, so the ablation isolates
    # the graph's contribution to MWE quality.
    mwe_phrases, single_phrases = None, None
    try:
        from scptm.keywords import extract_separated_topics
        valid_docs = [d for d in docs if len(str(d).strip()) > 10]
        topics_dict, _, _ = extract_separated_topics(
            valid_docs, model._nn, model._vocab,
            model._static_word_embs, model._sbert, model._stop,
            top_k=top_k, min_df=min_df,
        )
        mwe_phrases    = [topics_dict[f"Topic_{k+1}"]["phrases"] for k in range(K)]
        single_phrases = [topics_dict[f"Topic_{k+1}"]["single"]  for k in range(K)]
    except Exception as e:
        print(f"  [WARN] MWE extraction failed: {e}")

    return topic_words, theta, doc_mask, mwe_phrases, single_phrases


# ─────────────────────────────────────────────────────────────────────────────
# CTM vanilla runner — CombinedTM (Bianchi et al. 2021)
# ─────────────────────────────────────────────────────────────────────────────

def run_ctm_vanilla(
    docs: list,
    K: int,
    seed: int,
    sbert_embs: np.ndarray = None,
    min_df: int = 5,
    top_k: int = TOP_K_WORDS,
):
    """
    Train the *real* CTM (CombinedTM) from the contextualized_topic_models
    library — the honest neural-generative baseline (NOT an SCPTM ablation).

    Returns (topic_words, theta, doc_mask). No MWEs (CTM has no syntactic layer).
    """
    import torch as _torch
    _torch.manual_seed(seed)
    np.random.seed(seed)

    from contextualized_topic_models.models.ctm import CombinedTM
    from contextualized_topic_models.utils.data_preparation import (
        TopicModelDataPreparation,
    )

    # Light BoW preprocessing; contextual side uses the raw docs.
    qt = TopicModelDataPreparation(SBERT_MODEL)
    dataset = qt.fit(text_for_contextual=docs, text_for_bow=docs)

    contextual_size = (sbert_embs.shape[1] if sbert_embs is not None else 384)
    ctm = CombinedTM(
        bow_size        = len(qt.vocab),
        contextual_size = contextual_size,
        n_components    = K,
        num_epochs      = EPOCHS,
    )
    ctm.fit(dataset)

    topic_lists = ctm.get_topic_lists(top_k + 5)
    topic_words = _clean_topic_words([list(t) for t in topic_lists], top_k=top_k)

    try:
        theta = np.asarray(ctm.get_doc_topic_distribution(dataset, n_samples=20))
    except Exception:
        theta = None

    # CTM scores every document; if the data-prep dropped empty-BoW rows the
    # theta length will differ — in that case disable theta-based metrics.
    if theta is not None and theta.shape[0] != len(docs):
        print(f"  [WARN] CTM theta rows ({theta.shape[0]}) != docs ({len(docs)}); "
              "theta-based metrics disabled for this run.")
        theta = None

    doc_mask = np.ones(len(docs), dtype=bool)
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
    top_k: int = TOP_K_WORDS,
):
    """Train BERTopic. Reuses pre-computed embeddings to avoid double SBERT."""
    sbert = SentenceTransformer(SBERT_MODEL)
    # min_df=1 here: BERTopic re-runs this vectorizer on K pseudo-documents
    # (one per topic) for c-TF-IDF.  If min_df > K the fit crashes with
    # "max_df corresponds to < documents than min_df".  Vocabulary size is
    # controlled by max_features instead.
    vec   = CountVectorizer(
        stop_words   = "english",
        min_df       = 1,
        max_features = 20_000,
        ngram_range  = (1, 2),
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

    raw_words = []
    for tid in sorted(bt.get_topics()):
        if tid == -1:
            continue
        raw_words.append([w for w, _ in bt.get_topic(tid)[:top_k + 5]])
    topic_words = _clean_topic_words(raw_words, top_k=top_k)

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
    out_csv   = OUT_DIR / f"raw_{corpus_name}.csv"
    words_csv = OUT_DIR / f"words_{corpus_name}.csv"
    rows      = []

    # Build shared bigram corpus for LDA (reference BoW is already bigram-aware,
    # built in main() before calling this function).
    print("  Building bigram corpus for LDA (gensim Phrases)...")
    bigram_texts, _phraser = build_bigram_corpus(docs, min_count=min_df)

    # ── Shared evaluation inputs (built once per corpus) ─────────────────────
    # tokenized_texts : unigram reference corpus for gensim C_V / C_NPMI
    # df_lookup       : {token: document_frequency} for MWE specificity (IDF)
    print("  Preparing coherence reference (tokenised texts + DF lookup)...")
    tokenized_texts = [
        [w for w in str(d).lower().split()
         if w.isalpha() and w not in _STOP_SET and len(w) > 2]
        for d in docs
    ]
    _df = np.asarray((bow_ref > 0).sum(axis=0)).ravel()
    df_lookup = {vocab_ref[i]: int(_df[i]) for i in range(len(vocab_ref)) if _df[i] > 0}

    # Load existing results if resuming
    if out_csv.exists():
        existing = pd.read_csv(out_csv)
        print(f"  Resuming from {len(existing)} existing rows in {out_csv.name}")
    else:
        existing = pd.DataFrame()

    def _already_done(model, K, seed):
        if existing.empty:
            return False
        match = existing[
            (existing.model == model) &
            (existing.K     == K)     &
            (existing.seed  == seed)
        ]
        if match.empty:
            return False
        # Re-run ERROR rows (all metrics NaN)
        return not match["npmi"].isna().all()

    for K in k_values:
        for seed in SEEDS:
            for model_name in MODEL_ORDER:

                if _already_done(model_name, K, seed):
                    print(f"  [skip] {model_name} K={K} seed={seed} (already done)")
                    continue

                print(f"\n  ── {model_name}  K={K}  seed={seed} ──")
                theta    = None
                doc_mask = None
                mwe_phrases = None
                single_phrases = None
                try:
                    if model_name == "LDA":
                        topic_words, theta, doc_mask = run_lda(
                            docs, K, seed, min_df, bigram_texts=bigram_texts)

                    elif model_name == "CTM":
                        topic_words, theta, doc_mask = run_ctm_vanilla(
                            docs, K, seed, sbert_embs, min_df)

                    elif model_name in SCPTM_MODES:
                        topic_words, theta, doc_mask, mwe_phrases, single_phrases = \
                            run_scptm_variant(
                                docs, K, seed, SCPTM_MODES[model_name],
                                cache_path, min_df)

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
                        tokenized_texts=tokenized_texts,
                        df_lookup=df_lookup,
                        theta=theta,
                        doc_mask=doc_mask,
                        doc_embeddings_full=sbert_embs,
                        true_labels=aligned_labels,
                        mwe_per_topic=mwe_phrases,
                        single_per_topic=single_phrases,
                        raw_docs=docs,
                        run_stance=(RUN_STANCE and seed in STANCE_SEEDS),
                    )

                except Exception as exc:
                    print(f"  [ERROR] {model_name} K={K} seed={seed}: {exc}")
                    import traceback; traceback.print_exc()
                    metrics     = dict(npmi=np.nan, c_v=np.nan, c_npmi=np.nan,
                                       we_coherence=np.nan, diversity=np.nan,
                                       quality=np.nan)
                    topic_words = []   # nothing to save for words CSV

                finally:
                    # Free GPU memory between runs regardless of outcome
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    import gc; gc.collect()

                # Reindex to the canonical schema so every row has identical
                # columns regardless of which metrics the model produced.
                row = dict(corpus=corpus_name, model=model_name, K=K, seed=seed,
                           **{c: metrics.get(c, np.nan) for c in METRIC_COLUMNS})
                rows.append(row)

                # ── Append metrics (crash-safe) ───────────────────────────────
                pd.DataFrame([row]).to_csv(
                    out_csv, mode="a",
                    header=not out_csv.exists() or os.stat(out_csv).st_size == 0,
                    index=False,
                )

                # ── Append top words (crash-safe) ─────────────────────────────
                # One row per topic: corpus, model, K, seed, topic_id, keywords
                # Keywords stored as pipe-separated string to avoid CSV quoting
                # issues with commas inside word lists.
                if topic_words:
                    word_rows = []
                    for tid, words in enumerate(topic_words):
                        row_w = {
                            "corpus":     corpus_name,
                            "model":      model_name,
                            "K":          K,
                            "seed":       seed,
                            "topic_id":   tid + 1,
                            "keywords":   " | ".join(words),
                            "mwe_phrases": (
                                " | ".join(mwe_phrases[tid])
                                if mwe_phrases and tid < len(mwe_phrases)
                                else ""
                            ),
                        }
                        word_rows.append(row_w)
                    pd.DataFrame(word_rows).to_csv(
                        words_csv, mode="a",
                        header=not words_csv.exists() or os.stat(words_csv).st_size == 0,
                        index=False,
                    )
                def _f(x):
                    return f"{x:.3f}" if isinstance(x, (int, float)) and not np.isnan(x) else " nan"
                print(f"    C_V={_f(metrics.get('c_v', np.nan))}  "
                      f"C_NPMI={_f(metrics.get('c_npmi', np.nan))}  "
                      f"Div={_f(metrics.get('diversity', np.nan))}  "
                      f"Excl={_f(metrics.get('exclusivity', np.nan))}  "
                      f"Qual={_f(metrics.get('quality', np.nan))}"
                      + (f"  MWE-cmp={_f(metrics['mwe_compactness'])}"
                         if "mwe_compactness" in metrics else "")
                      + (f"  NMI={_f(metrics['nmi'])}" if "nmi" in metrics else ""))

    # Merge with existing results
    all_rows = pd.read_csv(out_csv)
    return all_rows


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation and plotting
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_results(raw: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std over seeds for each (corpus, model, K). Uses nanmean so the
    stance/valence columns (computed on a subset of seeds) aggregate cleanly."""
    metrics = [m for m in METRIC_COLUMNS if m in raw.columns]
    agg = (
        raw.groupby(["corpus", "model", "K"])[metrics]
        .agg([("mean", "mean"), ("std", "std")])   # pandas mean/std skip NaN
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
            if msub.empty:
                continue
            if corpus == "20NG":
                fixed = msub[msub["K"] == K_NEWSGROUPS]
                best = fixed.iloc[0] if not fixed.empty else msub.iloc[0]
            elif msub["quality_mean"].notna().any():
                best = msub.loc[msub["quality_mean"].idxmax()]
            else:
                # quality undefined (e.g. coherence failed) → fall back to max K
                best = msub.loc[msub["K"].idxmax()]
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
               "SCPTM-none": "#8C8C8C", "SCPTM-nosyn": "#9CC3E0",
               "SCPTM-fulldep": "#DD8452", "SCPTM": "#C44E52",
               "BERTopic": "#CCB974"}

    for ax, corpus in zip(axes, corpora):
        sub = agg[agg["corpus"] == corpus]
        for model in MODEL_ORDER:
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
    for model_name, mode in [("SCPTM-none", "none"), ("SCPTM", "filtered")]:
        tw, _, _dm, _mwe, _sg = run_scptm_variant(docs, K, seed, mode, cache_path, min_df)
        results[model_name] = {i + 1: tw[i] for i in range(len(tw))}

    fig, all_axes = plt.subplots(
        K, 2, figsize=(10, K * 1.6),
        gridspec_kw={"wspace": 0.05}
    )
    colours = {"SCPTM-none": "#8C8C8C", "SCPTM": "#C44E52"}

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
    print("CORPUS 1/4 — 20 Newsgroups  (K=20, with NMI ground truth)")
    print("=" * 65)

    ng_docs, ng_labels, _ = load_20newsgroups()
    ng_bigram, _           = build_bigram_corpus(ng_docs, min_count=5)
    ng_bow, ng_vocab       = build_reference_bow(ng_bigram, min_df=5)
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

    # ── 2. EU Parliament Debates ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("CORPUS 2/4 — EU Parliament Debates  (English-native, K-sweep)")
    print("=" * 65)

    eu_docs  = load_eu_debates(max_docs=20_000)
    eu_bigram, _     = build_bigram_corpus(eu_docs, min_count=10)
    eu_bow, eu_vocab = build_reference_bow(eu_bigram, min_df=10)
    eu_cache         = str(Path(DRIVE_ROOT) / "eudebates_cache.pkl")
    eu_embs          = precompute_sbert_embeddings(
                          eu_docs, str(Path(DRIVE_ROOT) / "eudebates"))

    raw_eu = run_corpus_sweep(
        "EU_Debates", eu_docs, K_RANGE,
        eu_bow, eu_vocab, sbert,
        eu_cache, eu_embs,
        min_df=10,
    )
    all_raw.append(raw_eu)

    # ── 3. Reddit politics ───────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("CORPUS 3/4 — Reddit Liberals vs Conservatives  (K-sweep)")
    print("=" * 65)

    rd_docs, rd_labels = load_reddit_politics(REDDIT_CSV)
    rd_bigram, _       = build_bigram_corpus(rd_docs, min_count=5)
    rd_bow, rd_vocab   = build_reference_bow(rd_bigram, min_df=5)
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

    # ── 4. Measuring Hate Speech ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("CORPUS 4/4 — Measuring Hate Speech  (K=8 fixed + K-sweep)")
    print("=" * 65)

    hs_docs, hs_labels, hs_label_names = load_hate_speech(min_tokens=30)
    hs_bigram, _       = build_bigram_corpus(hs_docs, min_count=5)
    hs_bow, hs_vocab   = build_reference_bow(hs_bigram, min_df=5)
    hs_cache           = str(Path(DRIVE_ROOT) / "hatespeech_cache.pkl")
    hs_embs            = precompute_sbert_embeddings(
                            hs_docs, str(Path(DRIVE_ROOT) / "hatespeech"))

    # K=8 matches the 8 ground-truth categories; also sweep for completeness
    hs_k_values = [8] + [k for k in K_RANGE if k != 8]

    raw_hs = run_corpus_sweep(
        "HateSpeech", hs_docs, hs_k_values,
        hs_bow, hs_vocab, sbert,
        hs_cache, hs_embs,
        true_labels=hs_labels, min_df=5,
    )
    all_raw.append(raw_hs)

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
            "c_v_mean", "c_npmi_mean", "diversity_mean", "exclusivity_mean",
            "cd_score_mean", "we_coherence_mean", "quality_mean",
            "mwe_compactness_mean", "mwe_complementarity_mean",
            "valence_gap_mean", "nmi_mean"]
    cols = [c for c in cols if c in main_table.columns]
    print(main_table[cols].to_string(index=False))

    # ── K-curve plot ─────────────────────────────────────────────────────────
    plot_k_curves(agg, metric="quality_mean")
    plot_k_curves(agg, metric="we_coherence_mean")

    # ── Qualitative: SCPTM-none vs SCPTM on EU Debates at K* ─────────────────
    eu_agg    = agg[agg["corpus"] == "EU_Debates"]
    scptm_agg = eu_agg[eu_agg["model"] == "SCPTM"]
    k_star    = int(scptm_agg.loc[scptm_agg["quality_mean"].idxmax(), "K"])
    print(f"\nQualitative plot: EU Debates at K*={k_star}")
    plot_qualitative(eu_docs, k_star, SEEDS[0], eu_cache, eu_embs,
                     corpus_name="EU Debates", min_df=10)

    print(f"\nAll outputs saved to {OUT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
