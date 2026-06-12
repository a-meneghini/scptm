# SCPTM — Structural Contextual Probabilistic Topic Model

A VAE-based topic model that combines heterogeneous graph neural networks over
syntactic dependency graphs with contextual SBERT word embeddings.

---

## Architecture overview

```
Documents ──SBERT──► doc embeddings ┐
                                    ├─► HeteroConv/GAT ──► μ, logσ² ──► z ──► θ (topic mix)
Vocabulary ──SBERT──► word embeddings ┘                                          │
                           │                                                     │
                     K-means init                                                 │
                           │                                                     ▼
                     topic_embeddings ──cosine/T──► β (topic×vocab) ──θ·β──► recon loss
```

**Key design choices:**

| Component | What it does |
|---|---|
| **HeteroConv / GAT encoder** | Propagates information through doc→word, word→word (syntax) and word→doc edges to produce per-document latent representations |
| **Contextual beta** | At evaluation: per-word topic affinity computed via attention pooling over SBERT sentence embeddings. At training: differentiable cosine similarity with temperature scaling |
| **Beta temperature (T=0.1)** | Cosine similarities in R³⁸⁴ concentrate near 0 (std≈1/√384≈0.051). Dividing by T maps them to ≈[−10,+10], making the softmax discriminative and gradients non-zero |
| **Word k-means init** | Topic embeddings are initialised from k-means centroids of the *word* embedding space (not documents), guaranteeing high cosine similarity with nearby vocabulary words from epoch 1 |
| **VAE with KL annealing** | Linear/cyclical schedule + free bits (per-dimension KL floor) to prevent posterior collapse |
| **Topic diversity loss** | Cosine repulsion between topic embedding pairs to prevent topic collapse |

---

## Installation

**From PyPI:**

```bash
pip install scptm

# With comparison benchmarks (BERTopic, CTM)
pip install "scptm[benchmark]"

# All optional dependencies
pip install "scptm[full]"
```

**For development (editable install):**

```bash
git clone https://github.com/a-meneghini/scptm.git
cd scptm
pip install -e ".[dev]"
```

**Required spaCy models:**

```bash
python -m spacy download en_core_web_sm   # English
python -m spacy download it_core_news_sm  # Italian
```

> **Note on `torch-geometric`:** SCPTM depends on [PyTorch Geometric](https://pytorch-geometric.readthedocs.io) (`torch-geometric>=2.4`), which is available on standard PyPI. If you need CUDA-accelerated graph operations, install the CUDA-specific wheel first following the [official PyG installation guide](https://pytorch-geometric.readthedocs.io/en/stable/install/installation.html) before installing SCPTM. CPU-only installs work out of the box with `pip install scptm`.

---

## Quick start

```python
from scptm import SCPTM, SCPTMConfig

documents = [
    "Machine learning is transforming healthcare diagnostics.",
    "Deep neural networks achieve state-of-the-art performance in NLP.",
    "Climate change accelerates biodiversity loss in tropical regions.",
    # ... hundreds more
]

# One-liner with defaults (10 topics, filtered syntax graph, English)
model = SCPTM()
theta = model.fit_transform(documents)    # (n_docs, K) topic mixtures

# Topic overview
model.get_topic_info(top_k=10)

# Out-of-sample inference
new_theta = model.transform(["A new document about AI research."])

# Evaluation
metrics = model.evaluate()
print(metrics)
# → {'npmi_coherence': 0.12, 'topic_diversity': 0.87, ...}

# Persist and reload
model.save("my_model.pkl")
model2 = SCPTM.load("my_model.pkl")
```

---

## Configuration

All hyper-parameters live in `SCPTMConfig`. Passing keyword arguments to `SCPTM()` directly is a shorthand for `SCPTM(config=SCPTMConfig(...))`.

```python
from scptm import SCPTM, SCPTMConfig

cfg = SCPTMConfig(
    # ── Model ──────────────────────────────────────────────────────────────
    num_topics          = 10,
    hidden_channels     = 64,       # GNN/MLP hidden size per attention head

    # ── Graph ──────────────────────────────────────────────────────────────
    graph_mode          = "filtered",
    # "none"      — no graph; pure MLP encoder (CTM-like baseline)
    # "no_syntax" — doc-word edges only, no word-word edges
    # "full_dep"  — all content dependency types
    # "filtered"  — informative dependency types only (default, recommended)

    # ── Training ───────────────────────────────────────────────────────────
    epochs              = 50,
    lr                  = 5e-3,
    batch_size          = 256,
    kl_max              = 1.0,
    kl_warmup_epochs    = 20,
    kl_strategy         = "linear",   # "linear" | "cyclical" | "constant"
    free_bits           = 0.1,        # per-dimension KL floor
    n_mc_samples        = 1,          # >1 enables MC uncertainty report

    # ── Beta ───────────────────────────────────────────────────────────────
    beta_temperature    = 0.1,        # softmax sharpening (lower = sharper)
    beta_refresh_epochs = 5,          # recompute contextual beta every N epochs
    max_ctx_occurrences = 50,         # max SBERT contexts stored per word

    # ── Regularisation ─────────────────────────────────────────────────────
    topic_diversity_weight = 0.1,     # cosine repulsion between topic embeddings

    # ── Corpus ─────────────────────────────────────────────────────────────
    lang                = "eng",      # "eng" | "ita"
    min_df              = 5,
    max_features        = 15_000,
    apply_chunking      = True,
    max_chunk_chars     = 800,

    # ── Keyword extraction ─────────────────────────────────────────────────
    bow_normalization   = "tf",       # "none" | "tf" | "log1p"
    keyword_method      = "cosine",   # "cosine" | "ctfidf"

    # ── Hardware ───────────────────────────────────────────────────────────
    use_mixed_precision = True,       # AMP on CUDA
    use_neighbor_sampling = False,    # NeighborLoader for large corpora

    # ── Reproducibility ────────────────────────────────────────────────────
    random_state        = 42,
)

model = SCPTM(config=cfg)
```

---

## Parse and embedding cache

spaCy lemmatisation, dependency parsing, and contextual SBERT embeddings are the
dominant cost on large corpora. Passing `edge_cache_path` persists all of them to a
single pickle file and skips re-computation on subsequent runs.

```python
# First run — parses corpus, encodes contextual embeddings, writes cache
theta = model.fit_transform(documents, edge_cache_path="corpus.pkl")

# Subsequent runs — skips spaCy and SBERT contextual pass entirely
model2 = SCPTM(config=cfg)
theta2 = model2.fit_transform(documents, edge_cache_path="corpus.pkl")
```

The cache stores: vocabulary, BoW matrix, dependency edge lists, and the
per-word contextual SBERT embeddings. If the corpus size or vocabulary
changes, the stale cache is detected automatically and rebuilt.

---

## Keyword extraction methods

```python
# Set globally
cfg = SCPTMConfig(keyword_method="ctfidf")

# Or override per call
model.get_topic_info(top_k=10, method="cosine")
model.get_topic_info(top_k=10, method="ctfidf")
model.get_topics_dict(top_k=5)          # returns single words + bigrams/trigrams
```

| Method | Ranks by | Best for |
|--------|----------|----------|
| `"cosine"` (default) | Cosine similarity between topic embedding and context-pooled word embedding | Semantically central terms |
| `"ctfidf"` | Class-based TF-IDF (each topic treated as a document class) | Discriminative / distinctive terms |

---

## Iterative refinement

Alternates between standard training and blending document embeddings toward their
dominant topic centroid. Useful when the initial embedding space lacks clear cluster structure.

```python
theta = model.fit(
    documents,
    iterative_refinement = True,
    n_refinement_steps   = 3,     # train → refine → train → ... (N steps)
    refinement_blend     = 0.2,   # alpha: 0 = no blend, 1 = full centroid
).theta
```

---

## Uncertainty quantification (Monte Carlo)

```python
cfg = SCPTMConfig(n_mc_samples=20)
model = SCPTM(config=cfg)
model.fit(documents)

# Per-document uncertainty regime
df = model.get_uncertainty_report()
# Columns: doc_id, regime, mean_std_mc, entropy_theta, dominant_topic, ...
# Regimes: CERTAIN | MODERATE | AMBIGUOUS | POORLY_ENCODED
```

---

## Comparison with baselines

To compare SCPTM against a CTM-like baseline and a TriTopic-like baseline:

```python
import pandas as pd
from scptm import SCPTM, SCPTMConfig

BASE = dict(num_topics=10, lang='eng', epochs=50, apply_chunking=False)

# CTM-like (no graph, MLP encoder only)
m_ctm = SCPTM(**BASE, graph_mode='none')
m_ctm.fit_transform(docs)
r_ctm = m_ctm.evaluate()

# TriTopic-like (no graph + iterative embedding refinement)
m_tri = SCPTM(**BASE, graph_mode='none')
m_tri.fit(docs, iterative_refinement=True, n_refinement_steps=3, refinement_blend=0.2)
r_tri = m_tri.evaluate()

# SCPTM with filtered syntax graph
m_full = SCPTM(**BASE, graph_mode='filtered')
m_full.fit_transform(docs)
r_full = m_full.evaluate()

# SCPTM + refinement
m_best = SCPTM(**BASE, graph_mode='filtered')
m_best.fit(docs, iterative_refinement=True, n_refinement_steps=3, refinement_blend=0.2)
r_best = m_best.evaluate()

rows = [
    ("CTM (no graph)",                r_ctm),
    ("TriTopic-like (no graph+refine)", r_tri),
    ("SCPTM (GNN filtered)",          r_full),
    ("SCPTM + refine",                r_best),
]
df = pd.DataFrame([
    {"model": name,
     "npmi": round(r.get("npmi_coherence", float("nan")), 3),
     "diversity": round(r.get("topic_diversity", float("nan")), 3)}
    for name, r in rows
])
print(df.to_string(index=False))
```

For a full sweep across all four graph modes:

```python
results = SCPTM.run_ablation_study(documents, epochs=50)
```

---

## Visualisations

```python
model.plot_training()     # loss + KL annealing + NPMI + diversity curves
model.visualize_3d()      # interactive Plotly 3D semantic constellation
model.visualize_2d()      # high-res PNG for papers (300 dpi)
```

---

## Architecture comparison

| | LDA | BERTopic | CTM | TriTopic | **SCPTM** |
|--|--|--|--|--|--|
| Model type | Generative (BoW) | Clustering | VAE | Clustering + refinement | VAE-GNN |
| Input signal | Co-occurrence | Embeddings | SBERT | SBERT | SBERT + syntax |
| Syntactic graph | ✗ | ✗ | ✗ | ✗ | ✓ |
| Contextual word embeddings | ✗ | ✗ | ✓ | ✓ | ✓ |
| Out-of-sample inference | ✓ | ✓ | ✓ | ✓ | ✓ |
| MC uncertainty | ✗ | ✗ | ✗ | ✗ | ✓ |
| Iterative refinement | ✗ | ✗ | ✗ | ✓ | ✓ (optional) |
| Multilingual | ✗ | ✓ | ✓ | partial | ✓ (eng/ita) |
| Embedding cache | ✗ | ✗ | ✗ | ✗ | ✓ |

> **When does the syntax graph help?**
> On formal corpora (scientific papers, news, legal documents) syntactic
> dependencies carry strong discriminative signal. On short informal text
> (social media, chat) the gap over a CTM baseline is smaller; use
> `graph_mode="none"` as a fast sanity-check.

---

## Notes on metrics

**NPMI coherence** measures how often a topic's top words co-occur in documents.
Typical target: > 0.10. Scores < 0 are common on short informal text (Reddit,
chat, social media) where words appear in isolation rather than in recurring
co-occurrence patterns — this is a property of the corpus, not a model failure.

**Topic diversity** = fraction of unique words across all topic top-word lists.
Score in [0, 1]; > 0.70 is generally considered good.

---

## Citation

```bibtex
@software{meneghini2026scptm,
  author  = {Meneghini, Alessandro},
  title   = {{SCPTM}: Structural Contextual Probabilistic Topic Model},
  year    = {2026},
  url     = {https://github.com/a-meneghini/scptm}
}
```

## License

MIT
