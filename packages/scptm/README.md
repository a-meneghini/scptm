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

# With the optional UDPipe lemmatisation backend
pip install "scptm[udpipe]"

# All optional dependencies
pip install "scptm[full]"
```

**For development (editable install):**

```bash
git clone https://github.com/a-meneghini/scptm.git
cd scptm
pip install -e ".[dev]"
```

**Required spaCy models (default lemmatiser):**

```bash
python -m spacy download en_core_web_sm   # English
python -m spacy download it_core_news_sm  # Italian
```

**UDPipe models (optional lemmatiser)** — installed automatically on first use
and cached under `~/.cache/scptm/udpipe/` (override with
`SCPTM_UDPIPE_MODEL_DIR` or `SCPTMConfig.udpipe_cache_dir`). No manual download
is required; see [Lemmatisation backends](#lemmatisation-backends).

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

All hyper-parameters are defined in `SCPTMConfig`. Passing keyword arguments to `SCPTM()` directly is a shorthand for `SCPTM(config=SCPTMConfig(...))` — but not both at once: `SCPTM(config=cfg, num_topics=25)` raises `TypeError` rather than silently ignoring `num_topics`.

Invalid values (`lang`, `graph_mode`, `lemmatizer`, `device`, …) raise `ValueError` at construction time, and the check also runs under `python -O`.

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

    # ── Lemmatisation backend ──────────────────────────────────────────────
    lemmatizer          = "spacy",    # "spacy" | "udpipe"
    udpipe_model_path   = None,       # explicit .udpipe model (optional)
    udpipe_cache_dir    = None,       # download cache (optional)

    # ── Keyword extraction ─────────────────────────────────────────────────
    bow_normalization   = "tf",       # "none" | "tf" | "log1p"
    keyword_method      = "cosine",   # "cosine" | "ctfidf"

    # ── Hardware ───────────────────────────────────────────────────────────
    device              = "auto",     # "auto" | "cpu" | "cuda" | "mps"
    use_mixed_precision = True,       # AMP on CUDA (ignored on CPU/MPS)
    use_neighbor_sampling = False,    # NeighborLoader (CPU/CUDA only)

    # ── Reproducibility ────────────────────────────────────────────────────
    random_state        = 42,
)

model = SCPTM(config=cfg)
```

---

## Lemmatisation backends

Vocabulary building and syntactic-dependency parsing run through a pluggable
backend selected with `SCPTMConfig.lemmatizer`. Both backends expose the same
token interface, so switching backends changes only the lemmas and dependency
labels — not the model code.

| `lemmatizer` | Pipeline | Models | Install |
|--------------|----------|--------|---------|
| `"spacy"` (default) | spaCy statistical pipeline | `en_core_web_sm`, `it_core_news_sm` | `pip install scptm` |
| `"udpipe"` | UDPipe (tokenise + tag + parse + lemmatise) | UD 2.5 EWT (eng), ISDT (ita) | `pip install "scptm[udpipe]"` |

UDPipe is often preferable for Italian (and other morphologically rich
languages) because its lemmatiser is trained on Universal Dependencies
treebanks and recovers verb/adjective base forms more reliably than the small
spaCy models.

```python
from scptm import SCPTM, SCPTMConfig

# English with UDPipe lemmatisation
cfg = SCPTMConfig(lang="eng", lemmatizer="udpipe")
theta = SCPTM(config=cfg).fit_transform(documents)

# Italian with UDPipe lemmatisation
cfg = SCPTMConfig(lang="ita", lemmatizer="udpipe")
theta = SCPTM(config=cfg).fit_transform(documents)

# Point at a model you downloaded yourself (offline / shared cluster)
cfg = SCPTMConfig(
    lang="ita",
    lemmatizer="udpipe",
    udpipe_model_path="/data/udpipe/italian-isdt-ud-2.5-191206.udpipe",
)
```

Model resolution order (first hit wins):

1. `SCPTMConfig.udpipe_model_path`
2. `SCPTM_UDPIPE_MODEL_ENG` / `SCPTM_UDPIPE_MODEL_ITA` environment variable
3. `SCPTMConfig.udpipe_cache_dir` (or `SCPTM_UDPIPE_MODEL_DIR`, or
   `~/.cache/scptm/udpipe`) — the official model is downloaded there on first
   use and reused afterwards.

If `lemmatizer="udpipe"` is requested without `ufal.udpipe` installed, SCPTM
raises an `ImportError` telling you to `pip install ufal.udpipe`.

---

## Device selection

The GNN, the decoder and the SBERT encoder run on the device set by
`SCPTMConfig.device`:

| `device` | Behaviour |
|----------|-----------|
| `"auto"` (default) | CUDA if available, else Apple Metal (MPS), else CPU |
| `"cpu"` | Force CPU |
| `"cuda"` | Force CUDA — raises if unavailable |
| `"mps"` | Force Apple Metal — raises if unavailable |

```python
SCPTM(device="cuda")   # NVIDIA GPU
SCPTM(device="mps")    # Apple Silicon (Metal)
SCPTM(device="cpu")    # portable / reproducible
```

Notes:

- SBERT, the heterogeneous GAT and the VAE all follow the same device; the
  returned `theta` is always a CPU tensor.
- **Mixed precision stays CUDA-only**: `use_mixed_precision` is ignored on
  CPU/MPS (Metal float16 autocast is numerically fragile), so MPS runs in
  full float32.
- **Neighbour sampling needs `pyg-lib` or `torch-sparse`.** SCPTM checks that
  they are importable *before* building the loader; if they are missing (or the
  device is MPS, where they are unsupported), `use_neighbor_sampling=True` is
  declined with a warning and the graph is trained full-batch instead of
  crashing mid-epoch. Note this also applies to the automatic enable above
  `neighbor_sampling_edge_threshold`: on a machine without those packages you
  get a warning, not an `ImportError`.
- `fit()` never mutates the config you pass: when neighbour sampling has to be
  turned on or off, only a private copy is changed.
- `random_state` seeds CPU, CUDA and the Metal generator, so runs are
  reproducible on MPS too.

---

## Parse and embedding cache

Lemmatisation, dependency parsing, and contextual SBERT embeddings are
highly impacting when working on large corpora. Passing `edge_cache_path` persists all of them to a
single pickle file and skips re-computation on subsequent runs.

```python
# First run — parses corpus, encodes contextual embeddings, writes cache
theta = model.fit_transform(documents, edge_cache_path="corpus.pkl")

# Subsequent runs — skips lemmatisation/parsing and SBERT contextual pass entirely
model2 = SCPTM(config=cfg)
theta2 = model2.fit_transform(documents, edge_cache_path="corpus.pkl")
```

The cache stores: vocabulary, BoW matrix, dependency edge lists, and the
per-word contextual SBERT embeddings.

A cache is reused **only when its signature matches**. The signature covers:

| Field | Why it matters |
|-------|----------------|
| corpus content hash | editing the text invalidates it even at constant document count |
| `lang` | different SBERT/lemmatiser model |
| `lemmatizer` | spaCy vs UDPipe |
| `graph_mode` | the word-word edges depend on the active dependency set |
| `min_df`, `max_features` | they define the vocabulary |
| token pattern | the vocabulary tokenizer rule (e.g. the accented-word fix) |

Any mismatch triggers an automatic rebuild with a warning that names the
changed field, e.g.:

```
[EdgeCache] Cache invalidated — min_df: 1 -> 5. Rebuilding.
```

Changing only `num_topics` (or other training hyper-parameters) does **not**
invalidate the cache, so a `K` sweep re-parses once and reuses the cache for
every subsequent `K`. Caches written before the signature existed are rebuilt
once and then carry a signature.

---

## Performance

Measured on a 300-document / 27k-token corpus (CPU), per fit:

| | before | after |
|---|---:|---:|
| `setup_nlp_pipeline` (2nd fit onwards) | 3.9 s | **0.0 s** |
| SBERT load with the model already cached | 5.4 s | **0.06 s** |
| contextual-embedding pass | 7.7 s | **1.9 s** |
| `compute_contextual_beta` (V=15k) | 1.07 s | **0.4 s** |

Three consecutive fits (K = 2, 3, 4) went from ~23 s each to **23 s / 10 s / 10 s**
— about **13 s saved per extra fit**.

What changed, in order of impact:

1. **The SBERT + lemmatiser pipeline is cached across fits.** Loading it costs
   several seconds and a `search_k` sweep used to pay that for every K. The
   cached objects are shared, so this also saves memory.
   ```python
   from scptm import clear_pipeline_cache
   clear_pipeline_cache()      # release cached models (e.g. to free accelerator memory)
   ```
2. **SBERT is loaded from the local HuggingFace cache, not the Hub.**
   `SentenceTransformer(name)` does a Hub round-trip on *every* construction
   even when the model is cached — 5.4 s and a network dependency. SCPTM now
   reads the cache directly and only downloads when the model is genuinely
   missing. **Cached models work fully offline** (`HF_HUB_OFFLINE=1` is not
   needed); set `SCPTM_HF_OFFLINE=1` to forbid the download fallback entirely
   and get an error instead.
3. **The contextual-embedding pass reuses the graph's document embeddings**
   instead of encoding the corpus a second time, and falls back to a **single
   batched** SBERT call when no embeddings are supplied (per-document encoding
   was ~8× slower).
4. **`nlp.pipe()` batching** for the vocabulary and dependency passes.
5. **Vectorised `compute_contextual_beta`** (chunked, padded batched matmuls
   instead of a Python loop over the vocabulary) — numerically identical to the
   previous per-word computation.

Practical knobs, in order of impact:

- **Pass `edge_cache_path=`** — skip lemmatisation, parsing and SBERT entirely on
  reruns. By far the biggest win for repeated experiments.
- Keep `max_features` / `min_df` reasonable; `beta_refresh_epochs` controls how
  often beta is recomputed; `epochs` is the main training cost.
- `device="cuda"` / `"mps"` for the GNN, and `batch_size` for the GNN batches.

Still on the table: `build_hetero_graph` parses the corpus **twice** (once to
build the vocabulary, once to build the dependency edges), which is the dominant
cost when the cache is cold.

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

`visualize_2d()` places labels with a deterministic greedy layout: each label
tries a ring of offsets around its anchor and keeps the first one that overlaps
nothing (labels, dots or topic stars); words that cannot be placed are dropped
instead of stacked, and topic labels move out on a thin leader line when their
neighbourhood is crowded. Labels are deduplicated (a word shared by two topics is
written once) and MWE labels are italic. `top_k` controls how many words/MWEs are
drawn per topic, and `adjust_labels=False`, `show_mwes=False`, `min_dist=...`,
`cmap=...` are forwarded to `scptm.visualization.view_semantic_2d_paper`.

Note that the label layout relies on the current backend's text metrics, so a
headless run (`MPLBACKEND=Agg`) and a notebook can differ by a pixel or two;
both are guaranteed overlap-free at draw time.

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
