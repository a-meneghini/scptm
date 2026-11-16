# Data provenance

**No corpus is redistributed in this repository.** All scripts expect the data
locally; `*.csv` and `*.pkl` are git-ignored. This file records where each corpus
comes from so the benchmark can be rebuilt from the sources.

Why not ship the data: the corpora are openly available from their publishers, and
re-hosting them would duplicate licences we do not own, bloat the repository, and
risk redistribution terms (the Reddit corpus in particular is not ours to
redistribute).

## Corpora used in the paper

| corpus | used for | source | licence / terms | how to obtain |
|---|---|---|---|---|
| **20 Newsgroups** | technical register baseline | scikit-learn's bundled loader | public, permissive | `sklearn.datasets.fetch_20newsgroups(subset="train", categories=[...], remove=("headers","footers","quotes"))` — no download needed |
| **EU Parliament Debates** | deliberative register | Chalkidis & Brandl (2024) | see the release terms | *TODO: exact URL + version used* |
| **UN General Debates** | deliberative register, 15-year span | Baturo, Dasandi & Mikhaylov, Harvard Dataverse | CC0 / open | *TODO: exact Dataverse DOI + the subset used* |
| **Reddit politics** | argumentative / informal | own collection | **not redistributable** | *TODO: document the collection window, subreddits and filtering rules, so a third party can rebuild an equivalent corpus* |
| **Hate speech** | evaluative register | *TODO: cite the release used in the paper* | see the release terms | *TODO: exact URL* |

## Expected local layout

Scripts read from the repository root:

```
un-general-debates.csv      # git-ignored, ~129 MB
reddit_pol.csv              # git-ignored, ~5 MB
```

## Reproduction contract

For each corpus the paper reports: number of documents, vocabulary size after
minimum document frequency, and the number of dependency edges. Those three
numbers are the cheapest check that a rebuilt corpus matches the one used. They
are written to every benchmark row by `scptm_bow.benchmark`
(`n_docs`, `n_vocab`, plus the graph edge counts) precisely so a mismatch is
visible instead of silent.

## Reddit: what to record instead of the data

Because that corpus cannot be redistributed, a replication needs the *recipe*,
not the bytes. Record at minimum:

1. the API/archive source and the collection date window;
2. the subreddit list and the inclusion rule (e.g. score threshold, minimum length);
3. the de-duplication and language-filtering steps;
4. the resulting document count, so a rebuild can be compared.
