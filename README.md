# SCPTM

**Topic models that read the grammar, not just the words.**

Most topic models treat a document as a bag of words and throw the sentence
structure away. SCPTM keeps it: lemmas are linked by the dependency edges of a
parser, and those links shape the topics. The result is topics whose vocabulary
follows how words actually combine — plus the multiword expressions
(*"spirito santo"*, *"curia romano"*, *"payload orbit"*) that a bag of words can
only ever see as two unrelated tokens.

[![PyPI: scptm](https://img.shields.io/pypi/v/scptm.svg?label=scptm&color=4C72B0)](https://pypi.org/project/scptm/)
[![PyPI: scptm-bow](https://img.shields.io/pypi/v/scptm-bow.svg?label=scptm-bow&color=DD8452)](https://pypi.org/project/scptm-bow/)
[![CI](https://github.com/a-meneghini/scptm/actions/workflows/ci.yml/badge.svg)](https://github.com/a-meneghini/scptm/actions/workflows/ci.yml)
[![Python](https://img.shields.io/pypi/pyversions/scptm.svg)](https://pypi.org/project/scptm/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

![SCPTM architecture](paper/figures/scptm_architecture.png)

*The neural pipeline (`scptm`). The count-based one (`scptm-bow`) keeps steps 1,
2 and 4 — word vectors, dependency edges, MWE extraction — and replaces step 3
with a regularised matrix factorisation.*

---

## Two implementations, one question

The scientific question is whether syntax earns its parsing cost. SCPTM answers
it twice, from opposite ends of the cost/accuracy spectrum, so a conclusion that
holds in both is a property of the corpus rather than of the architecture.

| | **`scptm`** | **`scptm-bow`** |
|---|---|---|
| how it works | SBERT embeddings + heterogeneous dependency graph + variational graph encoder | dependency-regularised NMF over a lemmatised bag of words + fastText |
| 500 docs, K=6 | 70.2 s | **18.1 s** |
| NPMI | 0.279 | **0.334** |
| reproducible to | local optima (neural) | **1.8 × 10⁻¹⁵** |
| needs | GPU optional, SBERT download | CPU only, nothing to download |

Neither depends on the other. Both expose the *same* metric API — `c_v`, `npmi`,
`diversity`, `exclusivity`, `redundancy` — verified against a single frozen
fixture, so a number from one package is comparable to the same number from the
other.

> On 20 Newsgroups the count-based model is 3.8× faster *and* more coherent than
> the neural one. That is not the expected result, and it is why the ablation
> apparatus below exists.

## What you get

```python
from scptm_bow import BowSCPTM

model = BowSCPTM(num_topics=4, lang="eng")
theta = model.fit_transform(documents)      # documents x topics
```

Real output from [`examples/demo.py`](packages/scptm-bow/examples/demo.py), 200
documents and four hidden word pools:

```
vocabulary: 44 lemmas | dependency edges: 402 | syntax graph: 216 edges
  c_v +0.956   npmi +0.972   diversity +1.000   exclusivity +1.000

Topic 1  (n=50)
  words: rocket, mission, launch, payload, astronaut, trajectory
  MWEs : payload nasa, payload astronaut, payload orbit
Topic 2  (n=50)
  words: resolution, display, graphic, shader, polygon, texture
  MWEs : resolution shader, pixel image, shader image
```

`theta` is the document–topic matrix, `model.topic_word` the topic–term matrix,
`model.get_topic_info()` the table you would put in a paper, and
`model.visualize_2d()` the semantic map below.

![Semantic space](paper/figures/semantic_space_2d.png)

---

## Install

```bash
pip install scptm            # neural: needs torch + sentence-transformers
pip install scptm-bow        # count-based: numpy, scipy, scikit-learn

# extras
pip install "scptm-bow[udpipe]"    # UDPipe lemmatiser instead of spaCy
pip install "scptm-bow[plots]"     # 2D semantic-space figures
pip install "scptm-bow[full]"      # everything
```

Both ship a command line front-end, so a corpus becomes a model in one line:

```bash
scptm-bow fit corpus.txt --topics 20 --lang ita --parser udpipe \
    --out model.json --figure semantic_space_2d.png
scptm-bow topics model.json
scptm-bow evaluate model.json heldout.txt
```

---

## Why this is more than a model zoo

Three things here are unusual for a topic-modelling package, and they exist
because a topic number is meaningless without them.

**A control condition for the syntax claim.** Is it the *content* of the
dependency edges, or just having more regularisation? `random_graph_matched`
answers that literally: 2-edge swaps preserve every word's degree exactly while
destroying the topology, so the only thing removed is the linguistic pairing.

```python
from scptm_bow.graphs import random_graph_matched
control = random_graph_matched(model.syntax_graph, seed=0)   # same degrees, random edges
```

**Intervals instead of point estimates.** `stats.paired_bootstrap_ci` pairs two
models on the same seed and reports the difference with a confidence interval;
Cliff's delta gives the effect size. And when a design *cannot* support a claim,
the package says so instead of silently returning p > 0.05:

```
wilcoxon: n=3: the smallest attainable two-sided p is 0.250, above alpha=0.05.
          Use the paired bootstrap CI instead.
```

**A benchmark as data, not as a script.** `BenchmarkSpec` declares the grid
(corpora × models × seeds × K), the runner is resumable, and the comparison
emits a booktabs table:

```python
from scptm_bow import BenchmarkSpec, ModelSpec, compare, run, to_latex

spec = BenchmarkSpec(
    corpora={"eu_debates": docs_eu, "ng20": docs_ng},
    models=[ModelSpec("lexical", lambda_syntax=0.0),
            ModelSpec("syntax",  lambda_syntax=0.1),
            ModelSpec("control", lambda_syntax=0.1, control="swap")],
    seeds=[0, 1, 2, 3, 4, 5], topics=[10, 20],
)
records = run(spec, out="bench.csv")
print(to_latex(compare(records, metric="c_v",
                       treatment="syntax", baseline="control")))
```

---

## Repository layout

```
packages/
  scptm/        src layout · tests · README · CITATION   → PyPI: scptm
  scptm-bow/    src layout · tests · README · CITATION   → PyPI: scptm-bow
conformance/    the frozen fixture both packages must reproduce
paper/          the paper's analyses, figures, and data provenance
.github/        ci.yml (lint + 121 tests, py3.10–3.12) · publish.yml (Trusted Publishing)
```

Both packages use a `src/` layout, so importing the source directory instead of
the installed package is structurally impossible.

```bash
git clone https://github.com/a-meneghini/scptm && cd scptm
python -m venv .venv && . .venv/bin/activate
pip install -e packages/scptm -e packages/scptm-bow
for f in packages/*/tests/test_*.py; do python "$f" || exit 1; done   # 121 tests
```

## Data

No corpus is redistributed here. [`paper/DATA.md`](paper/DATA.md) records the
source, licence and reproduction contract for each dataset — including how to
rebuild an equivalent corpus where the original cannot be reshared.

## Citation

If you use either package, please cite the paper — *Does Syntax Matter? A
Graph-Augmented Variational Topic Model for Computational Social Sciences* — and
the software (see [`CITATION.cff`](CITATION.cff)).

## Licence

MIT — see [`LICENSE`](LICENSE).
