"""
scptm_bow/config.py
--------------------
Configuration for SCPTM-bow.

Everything has a sensible default; invalid values raise ``ValueError`` at
construction time (not ``assert``, which ``python -O`` would strip).
"""

from dataclasses import dataclass, field
from typing import FrozenSet, List, Literal, Optional

# Universal Dependencies relations that tend to connect *content* words.
DEFAULT_DEP_TYPES: FrozenSet[str] = frozenset({
    "nsubj", "nsubjpass", "obj", "dobj", "iobj", "amod", "nmod", "compound",
    "conj", "xcomp", "ccomp", "advcl", "relcl", "appos", "attr", "pobj",
})

LANGS = ("eng", "ita")
PARSER_BACKENDS = ("spacy", "udpipe")
VECTOR_BACKENDS = ("fasttext", "ppmi_svd", "none")
DIVERGENCES = ("kl", "frobenius")


@dataclass
class BowConfig:
    """
    Parameters
    ----------
    lang : str
        "eng" or "ita" (selects the parsing model and the stop-word list).
    num_topics : int
        Number of topics (NMF rank K).
    min_df, max_features : int
        Vocabulary pruning, passed to ``CountVectorizer``.
    apply_chunking : bool
        Split long documents into ~``max_chunk_chars`` segments on sentence
        boundaries.  Without it, a long document is a single NMF row.
    max_chunk_chars : int
        Target segment size when ``apply_chunking`` is on.
    parser_backend : str
        "spacy" or "udpipe".
    udpipe_model_path : str | None
        Path to a ``.udpipe`` model (only for ``parser_backend="udpipe"``).
    dep_types : frozenset[str]
        Dependency relations that become graph edges.
    nmf_max_iter, nmf_tol : int, float
        Multiplicative-update budget.
    divergence : str
        "kl" (default) or "frobenius".  KL is the count-appropriate objective and
        matches LDA's mean-field problem; Frobenius is kept for comparison.
    graph_top_k : int
        Keep at most this many neighbours per word in each regularisation graph.
    lambda_syntax, lambda_embed : float
        Weights of the syntactic and semantic graph regularisers.  0.0 disables
        the corresponding term (so ``lambda_syntax=0, lambda_embed=0`` is plain
        NMF, a clean baseline).
    vector_backend : str
        "fasttext" (gensim, trains on the corpus), "ppmi_svd" (pure numpy/scipy
        fallback) or "none" (no semantic graph).
    vector_dim : int
        Dimensionality of the word vectors.
    fasttext_epochs, fasttext_window, fasttext_min_count : int
        fastText training settings.
    random_state : int
        Seed for the NMF initialisation.
    mwe_top_docs : int
        Documents per topic parsed for multiword-expression extraction.
    mwe_top_k : int
        MWEs returned per topic.
    topic_stopwords : list[str] | None
        Extra words hidden from the reported topic words.  They are *not*
        removed from the vocabulary, so the model, the cache and the metrics'
        denominator are unchanged — only the lists you read are cleaner.
    """

    # ---- Corpus ----
    lang: Literal["eng", "ita"] = "eng"
    min_df: int = 5
    max_features: int = 15_000
    apply_chunking: bool = False
    max_chunk_chars: int = 800

    # ---- Parsing ----
    parser_backend: Literal["spacy", "udpipe"] = "spacy"
    udpipe_model_path: Optional[str] = None
    dep_types: FrozenSet[str] = field(default_factory=lambda: DEFAULT_DEP_TYPES)

    # ---- NMF ----
    num_topics: int = 10
    nmf_max_iter: int = 200
    nmf_tol: float = 1e-4
    # "kl" (default) is the Poisson/multinomial objective appropriate for counts
    # and equivalent to LDA's mean-field problem; "frobenius" is the Euclidean
    # variant, kept for comparison.
    divergence: Literal["kl", "frobenius"] = "kl"
    random_state: int = 42

    # ---- Graph regularisation ----
    graph_top_k: int = 15
    lambda_syntax: float = 0.1
    lambda_embed: float = 0.1

    # ---- Word vectors ----
    vector_backend: Literal["fasttext", "ppmi_svd", "none"] = "fasttext"
    vector_dim: int = 100
    fasttext_epochs: int = 10
    fasttext_window: int = 5
    fasttext_min_count: int = 3

    # ---- Explanation ----
    mwe_top_docs: int = 200
    mwe_top_k: int = 15

    # ---- Topic reporting ----
    # Extra words hidden from the reported topic words (never from the
    # vocabulary: the model and the cached artefacts are unaffected).
    # Use BowSCPTM.common_words() to find corpus-specific candidates.
    topic_stopwords: Optional[List[str]] = None

    def __post_init__(self):
        if self.lang not in LANGS:
            raise ValueError(f"lang must be one of {list(LANGS)}, got '{self.lang}'")
        if self.parser_backend not in PARSER_BACKENDS:
            raise ValueError(
                f"parser_backend must be one of {list(PARSER_BACKENDS)}, "
                f"got '{self.parser_backend}'"
            )
        if self.vector_backend not in VECTOR_BACKENDS:
            raise ValueError(
                f"vector_backend must be one of {list(VECTOR_BACKENDS)}, "
                f"got '{self.vector_backend}'"
            )
        if self.divergence not in DIVERGENCES:
            raise ValueError(
                f"divergence must be one of {list(DIVERGENCES)}, "
                f"got '{self.divergence}'"
            )
        if self.num_topics < 2:
            raise ValueError(f"num_topics must be >= 2, got {self.num_topics}")
        if self.max_chunk_chars < 50:
            raise ValueError(
                f"max_chunk_chars must be >= 50, got {self.max_chunk_chars}"
            )
        if self.min_df < 1:
            raise ValueError(f"min_df must be >= 1, got {self.min_df}")
        if self.graph_top_k < 1:
            raise ValueError(f"graph_top_k must be >= 1, got {self.graph_top_k}")
        if self.lambda_syntax < 0 or self.lambda_embed < 0:
            raise ValueError("lambda_syntax / lambda_embed must be >= 0")
        if self.nmf_max_iter < 1:
            raise ValueError(f"nmf_max_iter must be >= 1, got {self.nmf_max_iter}")
