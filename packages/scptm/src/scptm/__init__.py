"""
SCPTM — Structural Contextual Probabilistic Topic Model
========================================================
A variational graph-based topic model that integrates:

  - Heterogeneous GNN over syntactic dependency graphs (doc→word, word→word)
  - Contextual word embeddings via SBERT attention pooling
  - Beta temperature scaling to produce discriminative topic-word distributions
  - Word k-means initialisation for stable training from epoch 1
  - VAE framework with KL annealing, free bits, and MC uncertainty
  - Scikit-learn compatible API: fit / transform / fit_transform / save / load
  - Parse + embedding cache to skip re-computation on repeated runs

Minimal usage::

    from scptm import SCPTM
    model = SCPTM(num_topics=10)
    theta = model.fit_transform(documents)
    model.get_topic_info()

Author: Alessandro Meneghini
License: MIT
"""

from . import metrics, stats
from .model import SCPTM
from .config import SCPTMConfig, resolve_device
from .evaluation import SCPTMEvaluator
from .nlp import clear_pipeline_cache
from .selection import search_k

__version__ = "0.2.2"
__all__ = [
    "SCPTM",
    "SCPTMConfig",
    "SCPTMEvaluator",
    "search_k",
    "resolve_device",
    "clear_pipeline_cache",
]
