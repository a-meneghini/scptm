"""
SCPTM-bow
==========
Structure-aware topic modelling **without neural embeddings**.

The whole idea: keep the part that makes the model distinctive — the syntactic
structure and the multiword expressions it yields — and drop the expensive,
hard-to-reproduce neural machinery around it.

    parse once  ->  lemmas, document-term counts, dependency pairs/triples
    fastText    ->  cheap word vectors (PPMI+SVD fallback, no download)
    NMF         ->  topics, regularised by the syntactic and semantic graphs
    triples     ->  multiword expressions, at no extra parsing cost

There is no torch, no sentence-transformers, no graph neural network and no
GPU requirement.

Minimal usage::

    from scptm_bow import BowSCPTM

    model = BowSCPTM(num_topics=10, lang="ita")
    model.fit(documents, cache_path="corpus.pkl")
    print(model.evaluate())
    for row in model.get_topic_info(top_k=8):
        print(row["topic_id"], row["top_words"], "|", row["mwes"])
"""

from .benchmark import BenchmarkSpec, ModelSpec, compare, run, to_latex
from .config import DEFAULT_DEP_TYPES, BowConfig
from .explain import extract_mwes, mwe_compactness, topic_labels
from .factorize import GraphRegularizedNMF, NMFResult
from .graphs import Graph, similarity_graph, syntax_graph
from .metrics import (
    npmi_coherence,
    redundancy,
    summary,
    topic_diversity,
    topic_exclusivity,
)
from .model import BowSCPTM
from .parse import ParsedCorpus, chunk_documents, parse_corpus
from .persistence import SAVE_FORMAT_VERSION, load_model, save_model
from .selection import search_k
from .store import corpus_hash, load_cache, save_cache
from .vectors import build_word_vectors, document_vectors
from .visualize import build_projection, plot_semantic_2d, topic_vectors

__version__ = "0.2.0"

__all__ = [
    "BowSCPTM",
    "BowConfig",
    "BenchmarkSpec",
    "ModelSpec",
    "compare",
    "run",
    "to_latex",
    "DEFAULT_DEP_TYPES",
    "ParsedCorpus",
    "parse_corpus",
    "chunk_documents",
    "Graph",
    "syntax_graph",
    "similarity_graph",
    "GraphRegularizedNMF",
    "NMFResult",
    "build_word_vectors",
    "document_vectors",
    "extract_mwes",
    "topic_labels",
    "mwe_compactness",
    "npmi_coherence",
    "topic_diversity",
    "topic_exclusivity",
    "redundancy",
    "summary",
    "corpus_hash",
    "load_cache",
    "save_cache",
    "save_model",
    "load_model",
    "SAVE_FORMAT_VERSION",
    "search_k",
    "plot_semantic_2d",
    "topic_vectors",
    "build_projection",
    "__version__",
]
