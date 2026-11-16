# SCPTM

Topic modelling that takes grammatical structure seriously, in two independent
implementations that answer the same question from opposite ends of the
cost/accuracy spectrum.

| package | what it is | speed | deterministic |
|---|---|---|---|
| **[`scptm`](packages/scptm)** | contextual SBERT embeddings + a heterogeneous dependency graph + a variational graph encoder | ~70 s / 500 docs | no (neural) |
| **[`scptm-bow`](packages/scptm-bow)** | dependency-regularised NMF over a lemmatised bag of words + fastText vectors | **~18 s / 500 docs** | yes (1e-15) |

They are **not** dependencies of each other: either installs and works alone.
They share an identical metric API (`c_v`, `npmi`, `diversity`, `exclusivity`,
`redundancy`) verified against a single frozen fixture, so a number produced by
one is comparable to the same number produced by the other.

```bash
pip install scptm          # the neural model
pip install scptm-bow      # the count-based model
```

```python
from scptm_bow import BowSCPTM
model = BowSCPTM(num_topics=20, lang="ita", parser_backend="udpipe")
theta = model.fit_transform(texts)
```

## What the two share, and why it is interesting

The scientific question is whether syntax earns its parsing cost. Both packages
answer it the same way — a within-model ablation of the dependency graph against a
**degree-matched random control** (`random_graph_matched`, 2-edge swaps) — so a
conclusion that holds in both is a property of the corpus, not of the
architecture.

Both also refuse to report a point estimate without an interval:
`stats.paired_bootstrap_ci` for the difference, `wilcoxon_paired` with an explicit
warning when three seeds cannot reach significance.

## Repository layout

```
packages/
  scptm/           pyproject.toml, src/scptm/, tests/, README.md, CITATION.cff
  scptm-bow/       pyproject.toml, src/scptm_bow/, tests/, README.md, CITATION.cff
conformance/       the frozen fixture both packages must reproduce
paper/             the paper's analyses and figures; paper/DATA.md = data provenance
.github/workflows/ ci.yml (lint + both suites, py3.10-3.12), publish.yml (Trusted Publishing)
RELEASING.md       how to cut a release
```

Each package uses a `src/` layout: importing the source directory instead of the
installed package is structurally impossible, which is what prevents the
namespace-shadowing bug this repository hit once already.

## Local development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e packages/scptm
pip install -e packages/scptm-bow
for f in packages/*/tests/test_*.py; do python "$f" || exit 1; done
python conformance/make_fixture.py   # must not change the fixture
```

## Data

No corpus is redistributed here. `paper/DATA.md` records the source, licence and
reproduction contract for each dataset used in the paper.

## Citation

See [`CITATION.cff`](CITATION.cff). The paper is *Does Syntax Matter? A
Graph-Augmented Variational Topic Model for Computational Social Sciences*.

## Licence

MIT — see [`LICENSE`](LICENSE).
