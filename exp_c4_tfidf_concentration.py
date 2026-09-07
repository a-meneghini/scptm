# -*- coding: utf-8 -*-
"""
exp_c4_tfidf_concentration.py
───────────────────────────────
C4: Recompute intra-topic concentration in a model-independent TF-IDF
    space to address the circular-dependency concern raised by the reviewer.

In the paper, intra-topic concentration was measured using SBERT document
embeddings.  Because BERTopic clusters documents in the same SBERT space,
this biases the metric in its favour.  Here we repeat the measurement using
TF-IDF document vectors, which are completely independent of all models.

Protocol:
  • For each corpus, load documents (same sources as benchmark_paper.py).
  • Fit TF-IDF once on all documents (max 20k features).
  • Re-fit each model at its best-K (from main_table.csv), seed=42.
  • Assign each document to its dominant topic (argmax of theta).
  • Compute concentration = mean intra-cluster cosine similarity (TF-IDF).
  • Save results and compare to the original SBERT-based concentration.

Models re-run: LDA, CTM, BERTopic, SCPTM (filtered).
One seed only (42).  Approx runtime: 2–3 h (SCPTM fits dominate).

Dependencies (in your benchmark env):
    scptm (local), gensim, bertopic, contextualized-topic-models,
    scikit-learn, numpy, tqdm

Output:
    benchmark_cache/results_0605/c4_tfidf_concentration.csv
    printed comparison table: SBERT vs TF-IDF concentration per model × corpus
"""

import csv
import sys
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent
# server layout: ~/scptm/benchmark_cache/results_v3/
# local layout:  ../results_v3/   (one level above scptm/)
_server_path = BASE / "benchmark_cache" / "results_v3"
_local_path  = BASE.parent / "results_v3"
WORDS_DIR  = _server_path if _server_path.exists() else _local_path
MAIN_TABLE = WORDS_DIR / "main_table.csv"
OUT_CSV    = WORDS_DIR / "c4_tfidf_concentration.csv"

# Local corpus CSVs
REDDIT_CSV = BASE / "reddit_pol.csv"
UNGDC_CSV  = BASE / "un-general-debates.csv"

SEED   = 42
MODELS = ["LDA", "CTM", "BERTopic", "SCPTM"]

# ── best-K lookup ─────────────────────────────────────────────────────────────

def load_best_k_table():
    table = {}
    with open(MAIN_TABLE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            table[(row["corpus"], row["model"])] = int(row["K"])
    return table

# ── corpus loaders (mirrors benchmark_paper.py) ───────────────────────────────

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
            # Try column names case-insensitively
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
            seen.add(t); docs.append(t)
    return docs

LOADERS = {
    "20NG":       load_20ng,
    "Reddit_Pol": load_reddit,
    "EU_Debates": load_eu_debates,
    "HateSpeech": load_hatespeech,
}

# ── TF-IDF concentration ──────────────────────────────────────────────────────

def tfidf_concentration(X_tfidf, topic_labels: np.ndarray,
                        doc_mask: np.ndarray = None) -> float:
    """
    Mean intra-topic cosine similarity in TF-IDF space.
    X_tfidf    : sparse (n_docs, vocab) TF-IDF matrix
    topic_labels : (n_used,) integer dominant-topic assignment
    doc_mask   : optional boolean mask of length n_docs selecting the rows of
                 X_tfidf that correspond to topic_labels (needed when a model
                 drops short documents, e.g. SCPTM with min_len > 0).
    """
    from sklearn.metrics.pairwise import cosine_similarity
    X = X_tfidf[doc_mask] if doc_mask is not None else X_tfidf
    assert X.shape[0] == len(topic_labels), (
        f"Shape mismatch: X has {X.shape[0]} rows but topic_labels has "
        f"{len(topic_labels)} entries"
    )
    unique = np.unique(topic_labels)
    sims = []
    for t in unique:
        mask = topic_labels == t
        n = int(mask.sum())
        if n < 2:
            continue
        X_t = X[mask]
        S   = cosine_similarity(X_t)
        sim = (S.sum() - n) / (n * (n - 1))
        sims.append(sim)
    return float(np.mean(sims)) if sims else float("nan")

# ── model runners ─────────────────────────────────────────────────────────────

def run_lda(docs, K, seed, min_df=5, max_features=20000):
    from gensim.corpora import Dictionary
    from gensim.models import LdaMulticore
    from sklearn.feature_extraction.text import CountVectorizer
    import re

    tokenize = lambda t: re.findall(r"[a-z]{2,}", t.lower())
    tokenized = [tokenize(d) for d in docs]
    dct    = Dictionary(tokenized)
    dct.filter_extremes(no_below=min_df, keep_n=max_features)
    corpus = [dct.doc2bow(t) for t in tokenized]

    lda = LdaMulticore(corpus, num_topics=K, id2word=dct,
                       passes=10, random_state=seed, workers=2)
    # Per-document topic distribution
    theta = np.zeros((len(docs), K))
    for i, bow in enumerate(corpus):
        for tid, prob in lda.get_document_topics(bow, minimum_probability=0):
            theta[i, tid] = prob
    return theta


def run_ctm(docs, K, seed):
    from contextualized_topic_models.models.ctm import CombinedTM
    from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
    from sentence_transformers import SentenceTransformer
    import re

    sbert = SentenceTransformer("all-MiniLM-L6-v2")

    tokenize = lambda t: " ".join(re.findall(r"[a-z]{2,}", t.lower()))
    bow_docs = [tokenize(d) for d in docs]

    qt = TopicModelDataPreparation("all-MiniLM-L6-v2")
    dataset = qt.fit(text_for_contextual=docs, text_for_bow=bow_docs)

    ctm = CombinedTM(bow_size=len(qt.vocab), contextual_size=384,
                     n_components=K, num_epochs=50)
    ctm.fit(dataset)
    theta = np.asarray(ctm.get_doc_topic_distribution(dataset, n_samples=20))
    return theta


def run_bertopic(docs, K, seed):
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    from hdbscan import HDBSCAN

    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    embs  = sbert.encode(docs, show_progress_bar=True, batch_size=64)

    umap_model    = UMAP(n_components=5, random_state=seed, metric="cosine")
    hdbscan_model = HDBSCAN(min_cluster_size=max(5, len(docs) // (K * 3)),
                             prediction_data=True)
    bt = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model,
                  nr_topics=K)
    topics, _ = bt.fit_transform(docs, embs)
    # Convert to 0-based dense labels (drop -1 noise)
    labels = np.array(topics)
    labels[labels < 0] = 0
    # Build soft theta: one-hot from hard assignment
    n_topics_actual = labels.max() + 1
    theta = np.zeros((len(docs), n_topics_actual))
    for i, t in enumerate(labels):
        theta[i, t] = 1.0
    return theta


def run_scptm(docs, K, seed, corpus_name, cache_dir):
    from scptm import SCPTM
    cache = str(cache_dir / f"c4_{corpus_name}_filtered.pkl")
    model = SCPTM(num_topics=K, graph_mode="filtered", lang="eng",
                  epochs=50, apply_chunking=False, min_df=5,
                  max_features=20000, random_state=seed,
                  metrics_every_n_epochs=50)
    model.fit_transform(docs, edge_cache_path=cache)
    theta = model._theta.numpy()   # (n_valid_docs, K)
    # SCPTM silently drops documents with len(stripped) <= 10.
    # Build the boolean mask so the caller can align X_tfidf rows.
    doc_mask = np.array([len(str(d).strip()) > 10 for d in docs])
    assert doc_mask.sum() == len(theta), (
        f"doc_mask sum {doc_mask.sum()} != theta rows {len(theta)}"
    )
    return theta, doc_mask

# ── main ─────────────────────────────────────────────────────────────────────

RUNNERS = {
    "LDA":     run_lda,
    "CTM":     run_ctm,
    "BERTopic":run_bertopic,
    "SCPTM":   run_scptm,
}

def main():
    from sklearn.feature_extraction.text import TfidfVectorizer

    best_k_table = load_best_k_table()
    cache_dir    = BASE / "benchmark_cache" / "c4_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    results = []
    corpora = list(LOADERS.keys())

    for corpus in corpora:
        print(f"\n{'='*60}")
        print(f"CORPUS: {corpus}")
        print(f"{'='*60}")

        try:
            docs = LOADERS[corpus]()
        except Exception as exc:
            print(f"  [ERROR loading corpus] {exc}")
            continue
        print(f"  {len(docs)} documents loaded")

        # Fit TF-IDF once for this corpus (shared across all models)
        print("  Fitting TF-IDF (max_features=20000) …")
        tfidf = TfidfVectorizer(max_features=20000, min_df=5,
                                sublinear_tf=True, stop_words="english")
        X_tfidf = tfidf.fit_transform(docs)
        print(f"  TF-IDF matrix: {X_tfidf.shape}")

        for model_name in MODELS:
            K = best_k_table.get((corpus, model_name))
            if K is None:
                print(f"  [{model_name}] No best-K found — skipping")
                continue

            print(f"\n  [{model_name}]  K={K} …")
            try:
                doc_mask = None
                if model_name == "LDA":
                    theta = run_lda(docs, K, SEED)
                elif model_name == "CTM":
                    theta = run_ctm(docs, K, SEED)
                elif model_name == "BERTopic":
                    theta = run_bertopic(docs, K, SEED)
                elif model_name == "SCPTM":
                    theta, doc_mask = run_scptm(docs, K, SEED, corpus, cache_dir)
                else:
                    continue

                labels = np.argmax(theta, axis=1)
                conc   = tfidf_concentration(X_tfidf, labels, doc_mask)
                print(f"    TF-IDF concentration: {conc:.4f}")
                results.append({
                    "corpus":      corpus,
                    "model":       model_name,
                    "K":           K,
                    "n_docs":      len(docs),
                    "concentration_tfidf": round(conc, 4),
                })

            except Exception as exc:
                print(f"    [ERROR] {exc}")
                import traceback; traceback.print_exc()

    if not results:
        print("No results collected.")
        sys.exit(1)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\n\nSaved → {OUT_CSV}")

    # ── summary table ─────────────────────────────────────────────────────────
    print("\n── TF-IDF Intra-topic Concentration (model-independent) ────────")
    print(f"  {'corpus':<14} {'model':<12} {'K':>4} {'TF-IDF conc':>12}")
    print(f"  {'-'*14} {'-'*12} {'-'*4} {'-'*12}")
    for r in results:
        print(f"  {r['corpus']:<14} {r['model']:<12} {r['K']:>4} "
              f"{r['concentration_tfidf']:>12.4f}")
    print()
    print("Note: compare these values to the SBERT-based concentration in")
    print("      main_table.csv (column: concentration_mean).")
    print("      If BERTopic's advantage shrinks under TF-IDF, the circular")
    print("      dependency concern is confirmed and the TF-IDF metric should")
    print("      replace SBERT concentration in Table 1.\n")


if __name__ == "__main__":
    main()
