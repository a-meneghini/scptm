# SCPTM — Structural Contextual Probabilistic Topic Model

A VAE-based topic model that integrates heterogeneous graph neural networks over syntactic dependency graphs with contextual word embeddings.

## Key ideas

- **Syntactic dependency graph** (doc→word, word→word) modelled with a HeteroConv/GAT encoder
- **Contextual beta**: per-word topic affinity computed via attention pooling over SBERT document embeddings
- **VAE framework** with KL annealing (linear / cyclical), free bits, and Monte Carlo uncertainty
- **Scikit-learn API**: `fit()` / `transform()` / `fit_transform()` / `save()` / `load()`
- **Iterative refinement**: optional TriTopic-inspired blending of doc embeddings toward topic centroids
- **Parse cache**: skip expensive spaCy re-parsing on repeated runs with `edge_cache_path`
- **c-TF-IDF ranking**: class-based TF-IDF keyword extraction as an alternative to cosine similarity

## Installation

The package lives in the inner `scptm/` subdirectory. Run all commands from there:

```bash
cd scptm          # enter the project root (where setup.py lives)
pip install -e .
# with benchmark dependencies
pip install -e ".[benchmark]"
```

## Running the demo

```bash
cd scptm          # project root
python3 demo.py   # 20 Newsgroups sample, ~400 docs, 5 topics
```

## Quick start

```python
from scptm import SCPTM, SCPTMConfig

documents = [
    "Machine learning is transforming healthcare diagnostics.",
    "Deep neural networks achieve state-of-the-art performance in NLP.",
    "Climate change accelerates biodiversity loss in tropical regions.",
    # ... hundreds more
]

# Default config (10 topics, filtered syntax graph)
model = SCPTM()
theta = model.fit_transform(documents)

# Topic overview
print(model.get_topic_info())

# Out-of-sample inference
new_theta = model.transform(["A new document about AI research."])

# Evaluation
metrics = model.evaluate()
print(metrics)

# Save and reload
model.save("my_model.pkl")
model2 = SCPTM.load("my_model.pkl")
```

## Configuration

```python
from scptm import SCPTMConfig

cfg = SCPTMConfig(
    num_topics          = 15,
    epochs              = 80,
    graph_mode          = "filtered",   # "none" | "no_syntax" | "full_dep" | "filtered"
    kl_strategy         = "cyclical",
    free_bits           = 0.5,
    n_mc_samples        = 20,           # enables MC uncertainty report
    topic_diversity_weight = 0.1,       # cosine repulsion between topics
    bow_normalization   = "tf",         # "none" | "tf" | "log1p"
    keyword_method      = "cosine",     # "cosine" | "ctfidf"
    use_mixed_precision = True,
    lang                = "eng",        # "eng" | "ita"
)

model = SCPTM(config=cfg)
```

## Parse cache (fast reload)

spaCy lemmatisation and dependency parsing are the dominant cost on large corpora. Pass `edge_cache_path` to persist the results and skip re-parsing on subsequent runs:

```python
# First run: parses the corpus and writes the cache
theta = model.fit_transform(documents, edge_cache_path="corpus_edges.pkl")

# Subsequent runs: skips spaCy entirely (~10× faster graph build)
model2 = SCPTM(config=cfg)
theta2 = model2.fit_transform(documents, edge_cache_path="corpus_edges.pkl")
```

The cache stores vocabulary, BoW matrix, and edge lists. If the corpus size changes, the stale cache is detected automatically and a fresh build is triggered.

Edge indices are also persisted inside `model.save()` so that a loaded model performs full GNN-based inference in `transform()` — previously, reloaded models used empty edge tensors.

## Keyword ranking methods

Two methods are available for extracting top words per topic:

| Method | How it works | Best for |
|--------|-------------|----------|
| `"cosine"` (default) | Ranks by the contextual beta matrix (cosine similarity between topic embeddings and context-pooled word embeddings) | Semantic coherence |
| `"ctfidf"` | Class-based TF-IDF: treats each topic as a document class, scores words by frequency within the class relative to the corpus | Discriminative / distinctive words |

```python
# Set globally via config
cfg = SCPTMConfig(keyword_method="ctfidf")

# Or override per call
model.get_topic_info(top_k=10, method="ctfidf")
model.get_topics_dict(top_k=5, method="cosine")
```

c-TF-IDF tends to surface more discriminative terms that distinguish topics from each other, while cosine similarity tends to surface semantically central terms.

## Ablation study

```python
results = SCPTM.run_ablation_study(documents, epochs=50)
print(results)
```

## Visualisations

```python
model.plot_training()         # loss + metric curves
model.visualize_3d()          # interactive Plotly 3D
model.visualize_2d()          # high-res PNG for papers
```

## Uncertainty report (requires n_mc_samples > 1)

```python
df = model.get_uncertainty_report()
# Regimes: CERTAIN | MODERATE | AMBIGUOUS | POORLY_ENCODED
```

## Bug fixes vs original script

| Fix | Description |
|-----|-------------|
| FIX-1 | `forward_encoder` used consistently in all graph modes (no `cat` workaround) |
| FIX-2 | Beta invalidated after every optimizer step; lazy full recompute every N epochs |
| FIX-3 | Topic diversity loss (cosine repulsion between topic embeddings) |
| FIX-4 | BoW normalisation (raw / TF / log1p) before reconstruction loss |
| FIX-5 | UMAP `random_state` wired from `SCPTMConfig` for reproducible layouts |
| FIX-6 | Edge indices persisted in `save()` / restored in `load()` — loaded models now use the full GNN graph in `transform()` instead of empty edge tensors |

## Comparison

| | LDA | BERTopic | TriTopic | **SCPTM** |
|--|--|--|--|--|
| Model type | Generative (BoW) | Clustering | Clustering (multi-view graph) | Generative VAE-GNN |
| Signals | Co-occurrence | Embeddings | Semantic + Lexical + Metadata | Semantic + Syntactic |
| Out-of-sample | ✓ | ✓ | ✓ | ✓ |
| Uncertainty | ✗ | ✗ | ✗ | ✓ (MC) |
| Syntactic graph | ✗ | ✗ | ✗ | ✓ |
| Topic embeddings | ✗ | ✗ | ✗ | ✓ |
| Iterative refinement | ✗ | ✗ | ✓ | ✓ |

## Citation

```bibtex
@software{scptm2026,
  author  = {Alessandro Meneghini},
  title   = {SCPTM: Structural Contextual Probabilistic Topic Model},
  year    = {2026},
  url     = {https://github.com/a-meneghini/scptm}
}
```

## License

MIT
