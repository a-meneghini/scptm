# SCPTM — Structural Contextual Probabilistic Topic Model

A topic model technique that combines heterogeneous graph neural networks over
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
    # ...  more
]

# One-liner with defaults (10 topics, filtered syntax graph, English)
model = SCPTM()
theta = model.fit_transform(documents)    # (n_docs, K) topic mixtures

# Topic overview
model.get_topic_info(top_k=10)

# Out-of-sample inference
new_theta = model.transform(["A new text about politics in development countries"])

# Evaluation
metrics = model.evaluate()
print(metrics)
# → {'npmi_coherence': 0.12, 'topic_diversity': 0.87, ...}

# Save and reload your model
model.save("my_model.pkl")
model2 = SCPTM.load("my_model.pkl")
```

---

## Configuration

All hyper-parameters are defined in `SCPTMConfig`. Passing keyword arguments to `SCPTM()` directly is a shorthand for `SCPTM(config=SCPTMConfig(...))`.

```python
from scptm import SCPTM, SCPTMConfig

cfg = SCPTMConfig(
    # ── Model ──────────────────────────────────────────────────────────────
    num_topics          = 10,
    hidden_channels     = 64,       # GNN/MLP hidden size per attention head

    # ── Graph ──────────────────────────────────────────────────────────────
    graph_mode          = "filtered",
    # "none"      — no graph
    # "no_syntax" — doc-word edges only, no word-word edges
    # "full_dep"  — all content dependency types
    # "filtered"  — dependencies that connect content words (nsubj, obj/dobj, amod, nmod, compound, conj, xcomp) only (default, recommended)

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

spaCy lemmatisation, dependency parsing, and contextual SBERT embeddings are
highly impacting when working on large corpora. Passing `edge_cache_path` persists all of them to a
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
cfg = SCPTMConfig(keyword_method="cosine")

# Or override per call
model.get_topic_info(top_k=10, method="cosine")
model.get_topic_info(top_k=10, method="ctfidf")
model.get_topics_dict(top_k=5)          # returns single words + bigrams/trigrams
```

| Method | Ranks by | Best for |
|--------|----------|----------|
| `"cosine"` (default) | Cosine similarity between topic embedding and context-pooled word embedding | Semantically central terms |
| `"ctfidf"` | Class-based TF-IDF (each topic treated as a document class) similar to BERTopic | Useful to retrieve discriminative terms |

---

## Visualisations

```python
model.plot_training()     # loss + KL annealing + NPMI + diversity curves
model.visualize_3d()      # interactive Plotly 3D semantic constellation
model.visualize_2d()      # high-res PNG for papers (300 dpi)
```

---

## Metrics

**NPMI coherence** measures how often a topic's top words co-occur in documents.

**Topic diversity** = fraction of unique words across all topic top-word lists.
Score in [0, 1]

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
