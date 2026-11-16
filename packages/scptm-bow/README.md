# SCPTM-bow

Structure-aware topic modelling **without neural embeddings**.

The idea: keep the part that makes SCPTM distinctive — the syntactic structure and
the multiword expressions it yields — and drop the expensive machinery around it
(SBERT, a heterogeneous GNN, a VAE). What is left is small, deterministic and
fast, and on a real corpus it is also *more* coherent.

```
parse once  ->  lemmas, document-term counts, dependency pairs/triples
fastText    ->  cheap word vectors (PPMI+SVD fallback, nothing to download)
NMF         ->  topics, regularised by the syntactic and semantic graphs
triples     ->  multiword expressions, at no extra parsing cost
```

No `torch`, no `sentence-transformers`, no `torch-geometric`, no GPU.

---

## Install

```bash
pip install -e ".[full]"     # spacy + ufal.udpipe + gensim
pip install -e ".[spacy]"    # parser only
pip install -e ".[vectors]"  # fastText only
pip install -e ".[plots]"    # 2D semantic-space figures (matplotlib + umap-learn)
```

Required: `numpy`, `scipy`, `scikit-learn`, `tqdm`. Everything else is optional —
the pipeline tells you what is missing instead of failing obscurely.

For Italian you also need a spaCy model (`python -m spacy download it_core_news_sm`)
or a UDPipe model file passed as `udpipe_model_path`.

---

## Quick start

```python
from scptm_bow import BowSCPTM

model = BowSCPTM(
    num_topics=10,
    lang="ita",
    parser_backend="udpipe",
    udpipe_model_path="/data/udpipe/italian-isdt-ud.udpipe",
)

model.fit(documents, cache_path="corpus.pkl")   # parse once, reuse for ever
print(model.evaluate())                         # nothing is printed for you
for row in model.get_topic_info(top_k=8):
    print(row["topic_id"], row["top_words"], "|", row["mwes"])
```

`fit()` returns `self`, so a notebook cell shows nothing. `model.theta` is the
document-topic matrix, `model.topic_word` the topic-term matrix, `model.top_words(k)`
the ranked terms. A `fit_transform(documents)` returning `theta` also exists for
symmetry with the neural API.

Out-of-sample inference is a fold-in — no re-training, only lemmatisation:

```python
theta_new = model.transform(["a new document"])
```

---

## Saving and loading

```python
model.save("model.pkl")                     # compact: no corpus text
model = BowSCPTM.load("model.pkl")
theta = model.transform(["a new document"])  # works: the parser is rebuilt lazily
```

`include_corpus=True` also stores the lemmatised training documents, which the
fitted model does not need but which let you inspect the parsed corpus or rebuild
the word vectors:

```python
model.save("model_full.pkl", include_corpus=True)
model.parsed.lemma_docs          # available after loading that file
```

The file is a versioned state dict (format version, kind, package version,
`BowConfig`, vocabulary, BoW, dependency graph and triples, word vectors, the two
graph Laplacians, the NMF factors), written atomically via a temp file and
`os.replace`. Only the fitted model is saved: `save()` raises `RuntimeError` on
an unfitted instance.

Loading never touches the parser — no spaCy model or UDPipe file is needed until
the first `transform()`. Foreign or corrupt files raise `ValueError`, a missing
path raises `FileNotFoundError`, and a file written by a newer format version
raises `ValueError` naming the version that wrote it. The format constant is
exported as `scptm_bow.SAVE_FORMAT_VERSION`; the pickled state contains no parser
and no torch tensors, so it stays readable across processes and hosts with the
same package/NumPy/SciPy versions.

---

## What it costs, and what it buys

20 Newsgroups, 500 documents (99,699 tokens), K=6, same corpus and same metric
implementation for both models:

| model | time | NPMI | diversity | redundancy | exclusivity |
|---|---:|---:|---:|---:|---:|
| scptm-bow — KL (default) | 18.1 s | 0.334 | 0.933 | 0.014 | 0.748 |
| scptm-bow — Frobenius | 15.5 s | **0.356** | **0.967** | **0.007** | **0.833** |
| SCPTM (SBERT + GNN + VAE) | 70.2 s | 0.279 | 0.933 | 0.014 | — |

**3.8× faster than the neural model with either divergence, and more coherent than
it** (0.334/0.356 vs 0.279). On a second, synthetic corpus with six well-separated
word pools it was 4.6× faster (5.1 s vs 23.3 s) with NPMI 0.975 vs 0.938.

> **KL vs Frobenius, measured.** KL is the *statistically* right objective for
> counts (Poisson/multinomial, the LDA-equivalent problem) — but on this corpus
> Frobenius scores better on every topic metric: NPMI 0.356 vs 0.334, diversity
> 0.967 vs 0.933, redundancy halved, exclusivity 0.833 vs 0.748.
>
> The difference is the divergence itself, **not** the graph weight: sweeping
> ``lambda`` from 0.0 to 0.5 moves KL between 0.333 and 0.334 and Frobenius
> between 0.354 and 0.356. KL's multinomial noise spreads mass across the
> vocabulary, giving smoother, less exclusive topics; Frobenius concentrates on
> the highest counts, and NPMI rewards exactly that concentration. Which one is
> "better" therefore depends on whether you want a well-specified likelihood or
> the most interpretable top-word lists. Both are one parameter away.

Reproducibility is exact: two fits with the same `random_state` agree to ~1e-15,
because NMF is deterministic and there is no stochastic encoder.

---

## How it works

**Objective**

```
min_{W,H >= 0}  D(X || W H)
                + lambda_syntax * tr(H L_s H^T)
                + lambda_embed  * tr(H L_e H^T)

D = Kullback-Leibler divergence (default) or squared Frobenius norm
```

* `X` documents x terms (counts), `W` documents x K, `H` K x terms.
* `D` is the generalised KL divergence by default — the count-appropriate
  objective and the same problem LDA's mean-field inference solves.  Set
  `divergence="frobenius"` for the Euclidean variant (see the measured note above).
* `L_s` — Laplacian of the **syntactic** graph: aggregated dependency pairs.
* `L_e` — Laplacian of the **semantic** graph: cosine k-NN over word vectors.
* Connected words are pushed towards similar topic memberships.

Solved with multiplicative updates; `L = D - A` is never materialised (only `A`
and its degree vector), so an iteration is a handful of sparse products.
`lambda_syntax = lambda_embed = 0` gives plain NMF — a one-line ablation, which is
exactly what `model.evaluate_no_syntax()` runs.

**Why the syntax is cheap here.** The parse pass collects, alongside each
document's lemmas, the aggregated `(head, relation, dependent)` triples. The
document-term matrix already encodes the doc-word edges, so no second pass is
needed to build the graph, and MWEs come from a dictionary scan over triples
already in memory. Syntax is extracted **once per corpus**, never per iteration.

---

## Differences from the neural `scptm`

| | `scptm` | `scptm_bow` |
|---|---|---|
| semantics | SBERT contextual embeddings | fastText (or PPMI+SVD), static |
| inference | variational, KL annealing, free bits | multiplicative-update NMF |
| structure | heterogeneous GAT message passing | graph Laplacian penalty on `H` |
| deps | torch, torch-geometric, sentence-transformers | numpy, scipy, sklearn |
| parse passes | 2–3 | 1 |
| determinism | seeds everything, still approximate | exact to ~1e-15 |
| uncertainty | MC samples / ELBO | not implemented (bootstrap-ready) |
| covariates | STM-style prior | not implemented |
| CPU/GPU | GPU recommended | CPU is fine |
| cache | pickle, signature-guarded | pickle, signature-guarded |

The two packages are fully independent: no shared imports, no shared cache format.
You can install, test and ship either one alone.

---

## Configuration

`BowConfig` (all fields have defaults; invalid values raise `ValueError`):

| field | default | meaning |
|---|---|---|
| `lang` | `"eng"` | `"eng"` / `"ita"` |
| `num_topics` | `10` | NMF rank K |
| `min_df`, `max_features` | `5`, `15000` | vocabulary pruning |
| `apply_chunking`, `max_chunk_chars` | `False`, `800` | split long documents into segments |
| `parser_backend` | `"spacy"` | `"spacy"` / `"udpipe"` |
| `udpipe_model_path` | `None` | model file (or `SCPTM_BOW_UDPIPE_{ENG,ITA}`) |
| `dep_types` | content relations | which relations become graph edges |
| `graph_top_k` | `15` | mutual k-NN bound for both graphs |
| `lambda_syntax`, `lambda_embed` | `0.1`, `0.1` | graph penalty weights (0 disables) |
| `vector_backend` | `"fasttext"` | `"fasttext"` / `"ppmi_svd"` / `"none"` |
| `vector_dim`, `fasttext_epochs` | `100`, `10` | word-vector settings |
| `nmf_max_iter`, `nmf_tol` | `200`, `1e-4` | multiplicative-update budget |
| `divergence` | `"kl"` | `"kl"` or `"frobenius"` |
| `mwe_top_k` | `15` | MWEs per topic |
| `topic_stopwords` | `None` | extra words hidden from the reported topic words |
| `random_state` | `42` | NMF initialisation seed |

---

## Stop words in the topic representation

NMF ranks topic words by frequency, so corpus-specific function words surface at
the top of every topic (`know, like, think, time, good` on 20 Newsgroups; `ciò,
esso, proprio` on Italian). They carry no topic information. The parser already
drops the language stop list, but that does not cover these.

```python
model.common_words(top_n=20)          # the most document-frequent lemmas
# [{'word': 'know', 'doc_freq': 123, 'share': 0.25}, ...]

model.top_words(10, extra_stopwords=["ciò", "esso", "proprio"])
model.get_topic_info(10, extra_stopwords=[...])

# or declare them once on the config, so every report uses them
BowConfig(topic_stopwords=["ciò", "esso", "proprio"])
```

Filtering is applied **only to the reported words** — never to the vocabulary —
so the fitted model, the cached artefacts and the metric denominators are
untouched. On 20 Newsgroups removing the 12 most frequent lemmas moves NPMI from
0.379 to 0.400 on the same word lists.

---

## Reporting a result, not a number

The package ships the apparatus a paper needs — the same one `scptm` ships, with
an identical metric API verified against a shared fixture.

```python
from scptm_bow import BowSCPTM, BenchmarkSpec, ModelSpec, compare, run, to_latex

# 1. a declarative grid: corpora x models x seeds x K
spec = BenchmarkSpec(
    corpora={"eu_debates": docs_eu, "ng20": docs_ng},
    models=[ModelSpec("lexical",  lambda_syntax=0.0, lambda_embed=0.0),
            ModelSpec("syntax",   lambda_syntax=0.1, lambda_embed=0.0),
            ModelSpec("control",  lambda_syntax=0.1, lambda_embed=0.0,
                      control="swap")],          # matched, linguistically empty
    seeds=[0, 1, 2, 3, 4, 5], topics=[10, 20],
)
records = run(spec, out="bench.csv")             # resumable
table   = compare(records, metric="c_v", treatment="syntax", baseline="control")
print(to_latex(table))
```

`compare` pairs the two models **on the same seed and K**, reports the mean
difference with a bootstrap CI and Cliff's delta, and Holm-corrects across the
corpus grid. Mean/sd/CI per group: `compare(records)`.

### The four instruments

| | what it answers |
|---|---|
| `metrics.c_v_coherence` | C_V (Röder et al. 2015), sliding-window NPMI context vectors |
| `stats.paired_bootstrap_ci` | is the difference real, given only a few seeds? |
| `graphs.random_graph_matched` | 2-edge swaps: same degrees, random topology |
| `model.heldout_perplexity` | does it generalise, with the vocabulary frozen? |

**A caveat that is easy to hide, and is therefore printed:** with three seeds a
two-sided Wilcoxon cannot reach p < 0.05 (minimum 2/8 = 0.25). `wilcoxon_paired`
warns instead of returning a p-value a reader will over-interpret. From six seeds
the test means what it looks like.

### Also available

```python
model.heldout_perplexity(heldout_docs)   # held-out, vocabulary never refit
model.bootstrap(n_resamples=20)          # topic stability + prevalence spread
model.estimate_effect(covariates, names=["year"])   # STM-style, post-hoc
model.temporal_prevalence(timestamps)    # prevalence per slice, with CIs
model.temporal_drift(timestamps)         # refit per slice, align, measure movement
model.encode_documents(new_docs)         # project onto the fitted vocabulary
model.get_params() / set_params(...)     # sklearn-style introspection
```

```bash
scptm-bow fit corpus.txt --topics 20 --parser udpipe --out model.json --figure fig.png
scptm-bow topics model.json
scptm-bow evaluate model.json heldout.txt
```

---

## Visualisation

```python
model.visualize_2d(top_k=12, save_path="semantic_space_2d.png")
# `plot_semantic_2d` is an exact alias — the neural package calls it visualize_2d
```

The same figure as the neural package's `view_semantic_2d_paper`: a UMAP
projection of the word-vector space where topics are stars, their top words are
dots and MWEs are italic annotations, and each topic is linked to its own words.
Render at 300 dpi by default.

The topic position is the phi-weighted centroid of its terms
(`sum_w phi_kw * vec_w`), so topics and words live in the same space and can be
projected together.

**Labels never overlap.** Word vectors of topically related terms project on top
of each other, so a fixed `ax.text` at every point produces unreadable piles.
Each label now tries a fixed ring of offsets around its anchor and keeps the
first one whose bounding box hits no already-placed label, dot or star; words
that cannot be placed are dropped instead of drawn over something else, and
topic labels that find no room nearby move further out on a thin leader line.
Labels are also deduplicated (a word shared by two topics is written once) and
MWEs are placed at the centroid of their parts rather than at a separately
projected mean vector, which is what produced the duplicated-looking
`comunione`/`riconciliazione` labels.

The layout is used for up to 20+ topics: the palette stops repeating after 20
(`tab10`/`tab20`, then `turbo`), and the UMAP defaults are looser than UMAP's own
(`n_neighbors=30`, `min_dist=0.35`) because the figure has to fit labels, not
preserve micro-structure. Everything is tunable:

```python
model.plot_semantic_2d(
    top_k=8,              # fewer words per topic
    mwe_per_topic=2,
    min_dist=0.5,         # more spacing still
    adjust_labels=True,   # False = legacy behaviour, for comparison
    show_mwes=False,
    cmap=None,            # any matplotlib colormap
)
```

Needs word vectors (`vector_backend != "none"`) and the `[plots]` extra. UMAP and
matplotlib are imported lazily, and `reducer=` accepts any `(n, D) -> (n, 2)`
callable — which is how the tests render the figure without UMAP.

It is also a fast diagnostic: on 20 Newsgroups it immediately exposes a generic
topic (`say, think, know, like, people`) and two overlapping ones.

---

## Choosing K

```python
best_k, model, results = BowSCPTM.search_k(
    docs,
    k_grid=(5, 10, 15, 20),
    cache_path="corpus.pkl",     # parsed once, reused for every K
    lang="ita",
    finetune=True,               # refine between the neighbours of the best K
)
# or the functional form
from scptm_bow import search_k
```

The criterion is the neural package's: **NPMI × diversity** (higher is better).
Parsing dominates the cost and does not depend on K, so the cache is reused for
every fit and the sweep is only K NMF factorisations. ``results`` maps each K to
its fitted model and its `npmi`/`div`/`score`; a K that fails to fit is recorded
with `model=None` and NaN metrics, is never selected, and a non-finite score is
never selected either. If no K yields a finite score, `RuntimeError` says why.

`finetune=True` inserts up to four intermediate K values between the two grid
neighbours of the winner (never duplicating a K already evaluated).

---

## Caching

`fit(documents, cache_path=...)` stores the parse artefacts (vocabulary, BoW,
dependency pairs and triples, lemmatised documents) and reuses them across runs.
The cache is validated by a signature — corpus hash, `lang`, parser backend and
model path, `min_df`, `max_features`, `dep_types` — so a stale vocabulary can
never be reused silently:

```
[cache] invalidated — min_df: 2 -> 5. Rebuilding.
```

Reusing the cache is worth ~11× on a small corpus (parsing dominates), and much
more on a large one.

---

## Tests

```bash
pytest tests/ -v          # or: python tests/test_core.py && python tests/test_pipeline.py
```

The pipeline tests replace the parser with a deterministic fake, so they need no
model, no GPU and no network.
