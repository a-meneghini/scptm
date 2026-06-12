"""
scptm/config.py
---------------
Central configuration dataclass for SCPTM.
All hyper-parameters in one place.

Quick reference
---------------
SCPTMConfig is a frozen-style dataclass; all fields have sensible defaults.
Pass it to SCPTM(config=cfg) or use keyword shortcuts SCPTM(num_topics=15).

Most important parameters for tuning results:

  num_topics          — try 5–20; more topics → finer granularity but
                        requires larger corpus for statistical support
  graph_mode          — "filtered" (default) uses informative syntactic
                        dependencies; "none" is a fast CTM-like baseline
  beta_temperature    — lower T → sharper topic-word distributions;
                        reduce to 0.05 for very large vocabularies
  epochs              — 50 is usually enough; increase if Recon still
                        declining at final epoch
  free_bits           — per-dimension KL floor; 0.1 works well for K≤20
  topic_diversity_weight — increase to 0.2+ if topics collapse
  lang                — "eng" (all-MiniLM-L6-v2) or "ita"
                        (paraphrase-multilingual-MiniLM-L12-v2)
"""

from dataclasses import dataclass, field
from typing import Literal, Optional


# Supported graph modes and their human-readable descriptions.
GRAPH_MODES = {
    "none":      "No syntactic graph — equivalent to CTM + KL annealing",
    "no_syntax": "Doc-word edges only, no word-word edges",
    "full_dep":  "All content dependency types (word-word)",
    "filtered":  "Informative dependency types only (default)",
}

# Dependency relation sets used in 'filtered' and 'full_dep' modes.
# "obj" is the Universal Dependencies v2 label used by Italian (and newer
# English) spaCy models; "dobj" is the legacy English label — both included
# for cross-lingual compatibility.
INFORMATIVE_DEP_TYPES: frozenset = frozenset({
    "nsubj", "obj", "dobj", "amod", "nmod", "compound", "conj", "xcomp",
})
ALL_CONTENT_DEP_TYPES: frozenset = frozenset({
    "nsubj", "nsubjpass", "obj", "dobj", "iobj", "amod", "nmod", "compound",
    "conj", "xcomp", "ccomp", "advcl", "relcl", "appos", "attr",
    "pobj", "advmod", "npadvmod",
})


@dataclass
class SCPTMConfig:
    """
    All configuration parameters for SCPTM.

    Parameters
    ----------
    num_topics : int
        Number of latent topics to discover.
    epochs : int
        Training epochs.
    lr : float
        AdamW learning rate.
    batch_size : int
        Documents per gradient step.
    hidden_channels : int
        Hidden dimension of the GNN / MLP encoder.
        NOTE: mu/logvar layers receive hidden_channels * 2 (from 2-head GAT).
    min_df : int
        Minimum document frequency for vocabulary inclusion.
    max_features : int
        Maximum vocabulary size.
    graph_mode : str
        One of {"none", "no_syntax", "full_dep", "filtered"}.
    lang : str
        Language code: "eng" or "ita".
    kl_max : float
        Maximum KL weight after annealing.
    kl_warmup_epochs : int
        Epochs to ramp KL weight from 0 to kl_max.
    kl_strategy : str
        Annealing schedule: "linear", "cyclical", or "constant".
    free_bits : float
        Minimum KL per dimension (prevents posterior collapse).
    n_mc_samples : int
        Monte Carlo samples for uncertainty estimation at inference.
    max_ctx_occurrences : int
        Max document embeddings stored per word for contextual beta.
    beta_refresh_epochs : int
        Recompute contextual beta every N epochs.
    metrics_every_n_epochs : int
        Compute NPMI and diversity every N epochs.
    topic_diversity_weight : float
        Strength of cosine-repulsion penalty between topic embeddings.
        Set to 0.0 to disable.
    bow_normalization : str
        BoW normalisation before reconstruction loss:
        "none"  — raw counts,
        "tf"    — divide by doc length,
        "log1p" — log(1 + count).
    keyword_method : str
        Keyword ranking method: "cosine" (default, beta/cosine similarity) or
        "ctfidf" (class-based TF-IDF, treats each topic as a document class).
    use_mixed_precision : bool
        Enable torch.cuda.amp (GPU only).
    use_neighbor_sampling : bool
        Use PyG NeighborLoader for mini-batch GNN training.
    apply_chunking : bool
        Split long documents into overlapping chunks.
    max_chunk_chars : int
        Maximum characters per chunk.
    random_state : int
        Seed for UMAP and numpy random operations.
    """

    # ---- Model architecture ----
    num_topics: int = 10
    hidden_channels: int = 64

    # ---- Corpus & vocabulary ----
    min_df: int = 5
    max_features: int = 15_000
    lang: Literal["eng", "ita"] = "eng"
    apply_chunking: bool = True
    max_chunk_chars: int = 800

    # ---- Training ----
    epochs: int = 50
    lr: float = 5e-3
    batch_size: int = 256
    kl_max: float = 1.0
    kl_warmup_epochs: int = 20
    kl_strategy: Literal["linear", "cyclical", "constant"] = "linear"
    free_bits: float = 0.1          # floor on KL per dim (0.5 was too high: with
                                    # K=10 the floor=5.0 pinned KL and zeroed
                                    # encoder gradients from the KL term)
    n_mc_samples: int = 1

    # ---- Graph ----
    graph_mode: Literal["none", "no_syntax", "full_dep", "filtered"] = "filtered"

    # ---- Contextual beta ----
    max_ctx_occurrences: int = 50
    beta_refresh_epochs: int = 5

    # ---- Beta temperature ----
    # Scales cosine similarities before softmax.  cos-sim in D=384 space
    # concentrates near 0 (std ≈ 1/√D ≈ 0.051); dividing by T=0.1 maps the
    # range to ≈ [-10, +10] which produces a discriminative softmax.
    beta_temperature: float = 0.1

    # ---- Regularisation ----
    topic_diversity_weight: float = 0.1   # cosine repulsion between topic embeddings

    # ---- BoW normalisation ----
    bow_normalization: Literal["none", "tf", "log1p"] = "tf"   # normalised before reconstruction loss; corrects for document-length bias

    # ---- Keyword extraction ----
    keyword_method: Literal["cosine", "ctfidf"] = "cosine"

    # ---- Logging ----
    metrics_every_n_epochs: int = 10

    # ---- Hardware ----
    use_mixed_precision: bool = True
    use_neighbor_sampling: bool = False

    # ---- Trainable word embeddings ----
    # When True, the decoder word embeddings start from SBERT but are updated
    # end-to-end via the reconstruction loss → β adapts to corpus co-occurrence,
    # increasing NPMI significantly.
    trainable_word_embeddings: bool = True

    # ---- Encoder residual connection ----
    # Add a skip connection from raw doc input to GAT output, mitigating
    # over-smoothing and recovering NMI lost due to graph message passing.
    encoder_residual: bool = True

    # ---- PMI-based graph sparsification ----
    # Filter word-word edges by Positive PMI (PPMI > 0) and keep at most
    # pmi_top_k_neighbors neighbours per word node. Reduces graph density
    # by ~70-90%, making attention selective and preventing over-smoothing.
    pmi_sparse_graph: bool = True
    pmi_top_k_neighbors: int = 15

    # ---- Differentiable coherence losses ----
    # npmi_coherence_weight: weight for NPMI-based coherence loss (pre-computed
    #   co-occurrence matrix). Set > 0 to directly optimise NPMI. Start at 0.05.
    # we_coherence_weight: weight for WE-coherence loss (cosine sim in SBERT
    #   space). Maximises semantic compactness of topics. Start at 0.05.
    # we_coherence_top_k: number of top words used in WE-coherence loss.
    npmi_coherence_weight: float = 0.0
    we_coherence_weight: float = 0.0
    we_coherence_top_k: int = 10

    # ---- Adaptive KL ----
    # When topic entropy drops below min_topic_entropy (topics collapsing),
    # the KL weight is temporarily boosted by adaptive_kl_boost to force
    # the posterior back towards the prior. Helps on homogeneous corpora.
    adaptive_kl: bool = True
    adaptive_kl_boost: float = 0.3
    min_topic_entropy: float = 1.0

    # ---- Reproducibility ----
    random_state: int = 42

    def __post_init__(self):
        assert self.graph_mode in GRAPH_MODES, (
            f"graph_mode must be one of {list(GRAPH_MODES)}, got '{self.graph_mode}'"
        )
        assert self.kl_strategy in ("linear", "cyclical", "constant"), (
            f"kl_strategy must be linear/cyclical/constant, got '{self.kl_strategy}'"
        )
        assert self.bow_normalization in ("none", "tf", "log1p"), (
            f"bow_normalization must be none/tf/log1p, got '{self.bow_normalization}'"
        )
        assert self.keyword_method in ("cosine", "ctfidf"), (
            f"keyword_method must be cosine/ctfidf, got '{self.keyword_method}'"
        )
