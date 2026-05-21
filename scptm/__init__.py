"""
SCPTM — Structural Contextual Probabilistic Topic Model
========================================================
A variational graph-based topic model that integrates:
  - Heterogeneous GNN over syntactic dependency graphs (doc-word, word-word)
  - Contextual word embeddings via SBERT attention pooling
  - VAE framework with KL annealing, free bits, and MC uncertainty
  - Scikit-learn compatible API (fit / transform / fit_transform)

Author: (your name)
License: MIT
"""

from .model import SCPTM
from .config import SCPTMConfig
from .evaluation import SCPTMEvaluator

__version__ = "0.1.0"
__all__ = ["SCPTM", "SCPTMConfig", "SCPTMEvaluator"]
