<div align="center">

# SCPTM
### Structural Contextual Probabilistic Topic Model

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.4+-orange.svg)](https://pyg.org/)

</div>

---

## What is SCPTM?

Most topic models treat a document as a **bag of words** — word order and grammar are thrown away.  
SCPTM keeps them. It builds a **heterogeneous graph** over your corpus where:

- **doc → word** edges connect documents to the vocabulary terms they contain
- **word → word** edges encode syntactic dependency relations (subject, object, modifier…)

A **Graph Attention Network (GAT)** reads this structure and feeds a **Variational Autoencoder (VAE)** that learns *K* latent topics. Word-topic affinities are not static: each word gets a **contextual beta** computed by attending over the SBERT embeddings of every document the word appears in.

The result is a model that can separate *"bank" (financial)* from *"bank" (river)* because their syntactic neighbourhoods are different — something LDA, BERTopic, and TriTopic cannot do.

---

## Architecture at a glance

```
Documents
    │
    ├─── SBERT encoder ──────────────────────────────────┐
    │                                                     │
    ├─── spaCy dependency parser                         │  Contextual beta
    │         │                                          │  (attention pooling
    │    doc→word edges                                  │   per word)
    │    word→word edges (syntax)                        │
    │         │                                          │
    └─── HeteroData (PyG) ───────────────────────────────┘
                │
         HeteroConv / GAT
         (2 heads, mean aggr)
                │
           μ, log σ²  ──── reparameterise ──── z
                │                               │
           KL loss                         softmax → θ  (doc-topic mixture)
                                                │
                                      θ · β  →  p̂(w|d)
                                                │
                                        reconstruction loss
                                        + diversity penalty
```

---

## Key features

| Feature | Details |
|---------|---------|
| **Syntactic graph** | 4 modes: `none` (CTM baseline), `no_syntax`, `full_dep`, `filtered` |
| **Contextual beta** | Per-word topic affinity via SBERT attention pooling — no static word2vec |
| **KL annealing** | Linear, cyclical, or constant schedules + free bits against posterior collapse |
| **Topic diversity loss** | Cosine repulsion between topic embeddings — prevents topic collapse |
| **MC uncertainty** | Monte Carlo sampling at inference → per-document uncertainty regimes |
| **Iterative refinement** | Blend doc embeddings toward topic centroids (TriTopic-inspired) |
| **Out-of-sample** | `transform()` for new documents without retraining |
| **Multilingual** | English (`en_core_web_sm`) and Italian (`it_core_news_sm`) out of the box |
| **scikit-learn API** | `fit()` / `transform()` / `fit_transform()` / `save()` / `load()` |
| **Full persistence** | Save and reload model, vocabulary, graph state, cached beta |

---

## How SCPTM compares

| | LDA | BERTopic | TriTopic | **SCPTM** |
|---|:---:|:---:|:---:|:---:|
| Model family | Generative | Clustering | Clustering | **Generative VAE** |
| Input signals | Word counts | Embeddings | Semantic + Lexical + Metadata | **Semantic + Syntactic** |
| Syntactic relations | ✗ | ✗ | ✗ | ✅ |
| Contextual word representations | ✗ | Partial | ✗ | ✅ |
| Learnable topic embeddings | ✗ | ✗ | ✗ | ✅ |
| MC uncertainty per document | ✗ | ✗ | ✗ | ✅ |
| Out-of-sample inference | ✅ | ✅ | ✅ | ✅ |
| Iterative refinement | ✗ | ✗ | ✅ | ✅ |
| 100% corpus coverage | ✅ | ✗ (~80%) | ✅ | ✅ |
| Soft topic assignments | ✅ | Partial | ✅ | ✅ |

> **Where SCPTM is unique:** it is the only model in this family that uses syntactic dependency structure and contextual per-word embeddings inside a generative probabilistic framework. This makes it more expressive on corpora with rich morphology, domain-specific terminology, or polysemous vocabulary.

---

## Installation

**Prerequisites:** Python ≥ 3.9, PyTorch ≥ 2.0

```bash
# 1. Clone the repo
git clone https://github.com/your-username/scptm.git
cd scptm

# 2. Install (editable)
pip install -e .

# 3. Download spaCy models
python -m spacy download en_core_web_sm   # English
python -m spacy download it_core_news_sm  # Italian (optional)

# 4. Install with benchmark comparison tools
pip install -e ".[benchmark]"
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `torch` + `torch-geometric` | GNN backbone |
| `sentence-transformers` | SBERT document and word embeddings |
| `spacy` | Dependency parsing |
| `scikit-learn` | Vocabulary, metrics |
| `umap-learn` | Dimensionality reduction for visualisations |
| `plotly` + `matplotlib` | Interactive and publication-quality plots |

---

## Quick start

```python
from scptm import SCPTM

documents = [
    "Machine learning algorithms are transforming medical diagnosis.",
    "Deep neural networks achieve superhuman performance on vision tasks.",
    "Climate change is accelerating the melting of Arctic ice sheets.",
    "Renewable energy adoption is growing faster than fossil fuel use.",
    # ... your corpus here
]

# Fit with defaults (10 topics, filtered syntax graph)
model = SCPTM()
theta = model.fit_transform(documents)

# Inspect topics
print(model.get_topic_info(top_k=10))

# Infer topics for new documents
new_theta = model.transform(["A new paper about transformer architectures."])

# Evaluate
metrics = model.evaluate()

# Save / reload
model.save("my_model.pkl")
model_loaded = SCPTM.load("my_model.pkl")
```

---

## Configuration

All parameters live in `SCPTMConfig`. Pass it to `SCPTM(config=cfg)` or use keyword shortcuts directly.

```python
from scptm import SCPTM, SCPTMConfig

cfg = SCPTMConfig(
    # --- Topics ---
    num_topics              = 15,

    # --- Training ---
    epochs                  = 80,
    lr                      = 5e-3,
    batch_size              = 256,

    # --- Graph structure ---
    graph_mode              = "filtered",   # "none" | "no_syntax" | "full_dep" | "filtered"

    # --- KL annealing ---
    kl_strategy             = "cyclical",   # "linear" | "cyclical" | "constant"
    kl_max                  = 1.0,
    kl_warmup_epochs        = 20,
    free_bits               = 0.5,          # prevents posterior collapse

    # --- Regularisation ---
    topic_diversity_weight  = 0.1,          # cosine repulsion between topic embeddings
    bow_normalization       = "tf",         # "none" | "tf" | "log1p"

    # --- Uncertainty ---
    n_mc_samples            = 20,           # >1 enables MC uncertainty report

    # --- Language ---
    lang                    = "eng",        # "eng" | "ita"

    # --- Hardware ---
    use_mixed_precision     = True,         # AMP on CUDA
    use_neighbor_sampling   = False,        # NeighborLoader for large graphs

    # --- Reproducibility ---
    random_state            = 42,
)

model = SCPTM(config=cfg)
```

### Graph modes

| Mode | Description | Best for |
|------|-------------|----------|
| `"none"` | No graph — pure MLP encoder, equivalent to CTM + KL annealing | Baseline comparison |
| `"no_syntax"` | Doc-word edges only, no word-word edges | Ablation: graph topology without syntax |
| `"full_dep"` | All content dependency types (15 relation types) | Dense corpora with varied syntax |
| `"filtered"` | Informative deps only: nsubj, dobj, amod, nmod, compound, conj, xcomp | **Default — best quality/speed trade-off** |

---

## API reference

### `SCPTM`

```python
model = SCPTM(config=None, **kwargs)
```

| Method | Returns | Description |
|--------|---------|-------------|
| `fit(source, source_type, ...)` | `self` | Fit on corpus. `source_type`: `"list"`, `"folder"`, `"dataframe"` |
| `transform(documents)` | `Tensor (N, K)` | Infer topic mixtures for new documents |
| `fit_transform(source, ...)` | `Tensor (N, K)` | Fit and return training document mixtures |
| `get_topic_info(top_k)` | `DataFrame` | Topic ID, size, top keywords |
| `get_topics_dict(top_k)` | `dict` | Per-topic single words + multi-word phrases |
| `get_document_topics()` | `DataFrame` | Per-document dominant topic and probability |
| `get_uncertainty_report()` | `DataFrame` | MC uncertainty regime per document |
| `evaluate(true_labels, theta_runs)` | `dict` | NPMI, diversity, NMI, F1, stability ARI |
| `plot_training()` | — | Loss + metric curves |
| `visualize_3d()` | — | Interactive Plotly 3D semantic constellation |
| `visualize_2d(save_path)` | — | High-res 2D UMAP figure (300 dpi) |
| `save(path)` | — | Full model pickle |
| `SCPTM.load(path)` | `SCPTM` | Reload saved model |
| `SCPTM.run_ablation_study(docs, ...)` | `DataFrame` | Train all 4 graph modes and compare metrics |

### Key attributes after `fit()`

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `model.theta` | `(N, K)` | Document-topic mixtures |
| `model.topic_embeddings` | `(K, D)` | Learnable topic vectors in SBERT space |
| `model.vocab` | `list[str]` | Vocabulary |

---

## Uncertainty report

When `n_mc_samples > 1`, SCPTM draws multiple samples from the posterior at inference time and classifies each document into one of four regimes:

| Regime | Condition | Interpretation |
|--------|-----------|----------------|
| `CERTAIN` | std < 0.02 | Stable, confident assignment |
| `MODERATE` | 0.02 ≤ std ≤ 0.08 | Minor ambiguity — usually fine |
| `AMBIGUOUS` | std > 0.08 **and** high entropy | Genuine multi-topic document |
| `POORLY_ENCODED` | std > 0.08 **and** low entropy | Out-of-vocabulary or anomalous text |

```python
cfg = SCPTMConfig(n_mc_samples=20)
model = SCPTM(config=cfg)
model.fit(documents)

df = model.get_uncertainty_report()
# columns: doc_id | regime | dominant_topic | dominant_prob | mean_std_mc | entropy_theta
```

---

## Iterative refinement

Inspired by TriTopic, this optional step alternates between training and blending document embeddings toward their dominant topic centroid:

```
emb_refined = (1 - blend) * emb_orig + blend * topic_centroid
```

This sharpens topic boundaries over multiple cycles without changing the model architecture.

```python
model.fit(
    documents,
    iterative_refinement = True,
    n_refinement_steps   = 3,
    refinement_blend     = 0.2,
)
```

---

## Ablation study

```python
results = SCPTM.run_ablation_study(documents, epochs=50, num_topics=10)
print(results)
```

```
graph_mode    description                              final_loss  npmi_coherence  topic_diversity
none          No syntactic graph (CTM baseline)         142.3       0.0821          0.712
no_syntax     Doc-word edges only                       138.7       0.1034          0.748
full_dep      All content dependency types              131.2       0.1189          0.779
filtered      Informative dependency types (default)    128.9       0.1247          0.803
```

---

## Evaluation

```python
# Basic metrics (no labels needed)
metrics = model.evaluate()
# → npmi_coherence, topic_diversity, n_topics

# With ground-truth class labels
metrics = model.evaluate(true_labels=np.array([0, 1, 2, 0, 1, ...]))
# → + nmi, downstream_f1

# With multiple runs (stability)
runs = []
for seed in [42, 123, 999]:
    m = SCPTM(SCPTMConfig(random_state=seed, epochs=50))
    theta = m.fit_transform(documents)
    runs.append(theta)
metrics = model.evaluate(theta_runs=runs)
# → + stability_ari
```

---

## Visualisations

```python
model.plot_training()          # training loss, KL, NPMI, diversity curves

model.visualize_3d()           # interactive 3D Plotly — topics as diamonds,
                                # words as spheres, edges = topic affinity

model.visualize_2d(            # high-res 2D UMAP (300 dpi PNG)
    save_path="figure1.png"
)
```

---

## Design decisions and known limitations

**Why a generative model and not clustering (like BERTopic / TriTopic)?**  
Generative models give you a proper likelihood, principled uncertainty via the VAE posterior, and the ability to treat topic discovery as approximate Bayesian inference. The BoW reconstruction loss also anchors topics to the actual lexical content of documents.

**Limitations to be aware of**

- **Parsing is the bottleneck.** spaCy dependency parsing is O(N) in documents but slow in absolute terms (~1–5 docs/sec on CPU). For corpora > 10k documents consider pre-computing and caching the edge lists, or using `graph_mode="no_syntax"`.
- **`transform()` is an approximation.** New documents are appended to the graph feature matrix but share no edges with the training graph. This works well in practice but is not exact inference.
- **No benchmark numbers yet.** The ablation study measures relative improvement across graph modes. A full comparison against BERTopic and TriTopic on 20 Newsgroups / BBC News / AG News is in progress.
- **Single language per model.** Multilingual corpora require the multilingual SBERT model and a compatible spaCy pipeline.

---

## Roadmap

- [ ] Benchmark evaluation on 20 Newsgroups, BBC News, AG News, arXiv
- [ ] Pre-cached edge list support (avoid re-parsing on reload)
- [ ] c-TF-IDF keyword ranking as alternative to cosine similarity
- [ ] LLM-powered topic labelling (Claude / GPT-4 via optional dependency)
- [ ] HuggingFace Hub integration for model sharing
- [ ] PyPI release

---

## Citation

If you use SCPTM in your research, please cite:

```bibtex
@software{scptm2025,
  author    = {Alessandro Meneghini},
  title     = {{SCPTM}: Structural Contextual Probabilistic Topic Model},
  year      = {2026},
  url       = {https://github.com/a-meneghini/scptm}
}
```

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">
<sub>Built with PyTorch · PyG · sentence-transformers · spaCy</sub>
</div>
