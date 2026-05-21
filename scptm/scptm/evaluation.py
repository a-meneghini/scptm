"""
scptm/evaluation.py
-------------------
Evaluation metrics for SCPTM:
  - NPMI coherence
  - Topic diversity
  - MC uncertainty report
  - Stability (pairwise ARI across multiple runs)
  - NMI against ground-truth labels (if available)
  - Downstream classification F1 (if labels available)
"""

from itertools import combinations
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def compute_topic_diversity(topics_words: List[List[str]]) -> float:
    """
    Proportion of unique words across all topic top-word lists.
    Score in [0, 1]; higher = more diverse topics.
    """
    all_words = [w for topic in topics_words for w in topic]
    if not all_words:
        return 0.0
    return len(set(all_words)) / len(all_words)


def compute_npmi_coherence(
    topics_words: List[List[str]],
    bow_sparse,
    vocab: list,
    top_k: int = 10,
    eps: float = 1e-10,
) -> float:
    """
    Mean pairwise NPMI coherence across all topics.

    Parameters
    ----------
    topics_words : list of word lists (one per topic)
    bow_sparse : scipy sparse matrix, shape (n_docs, vocab_size)
    vocab : list of vocabulary strings
    top_k : number of top words per topic to use

    Returns
    -------
    float in [-1, 1]; higher is better (target > 0.10 is reasonable).
    """
    vocab_idx = {w: i for i, w in enumerate(vocab)}
    n_docs = bow_sparse.shape[0]
    bow_bin = (bow_sparse > 0).astype(np.float32)

    topic_scores = []
    for topic_words in topics_words:
        valid = [w for w in topic_words[:top_k] if w in vocab_idx]
        if len(valid) < 2:
            continue
        pair_scores = []
        for w_i, w_j in combinations(valid, 2):
            i, j = vocab_idx[w_i], vocab_idx[w_j]
            p_i  = float(bow_bin[:, i].sum()) / n_docs + eps
            p_j  = float(bow_bin[:, j].sum()) / n_docs + eps
            p_ij = float(bow_bin[:, i].multiply(bow_bin[:, j]).sum()) / n_docs + eps
            npmi = np.log(p_ij / (p_i * p_j)) / (-np.log(p_ij))
            pair_scores.append(npmi)
        topic_scores.append(float(np.mean(pair_scores)))

    return float(np.mean(topic_scores)) if topic_scores else 0.0


def compute_stability(
    theta_runs: List[torch.Tensor],
    n_sample: int = 500,
) -> float:
    """
    Measure topic assignment stability across multiple training runs.

    For each run, assign each document to its dominant topic.
    Compute pairwise Adjusted Rand Index between all run pairs.

    Parameters
    ----------
    theta_runs : list of (n_docs, K) tensors — one per run
    n_sample : subsample this many documents for speed

    Returns
    -------
    float — mean pairwise ARI in [-1, 1]; higher is better.
    """
    from sklearn.metrics import adjusted_rand_score

    labels = [t.argmax(dim=-1).numpy() for t in theta_runs]
    n_docs = len(labels[0])
    if n_docs > n_sample:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_docs, n_sample, replace=False)
        labels = [lab[idx] for lab in labels]

    aris = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            aris.append(adjusted_rand_score(labels[i], labels[j]))

    return float(np.mean(aris)) if aris else 0.0


def compute_nmi(
    theta: torch.Tensor,
    true_labels: np.ndarray,
) -> float:
    """
    Normalised Mutual Information between dominant topic assignments
    and ground-truth class labels.

    Parameters
    ----------
    theta : (n_docs, K) topic mixture tensor
    true_labels : array of int class labels, shape (n_docs,)

    Returns
    -------
    NMI in [0, 1]; higher is better.
    """
    from sklearn.metrics import normalized_mutual_info_score

    pred_labels = theta.argmax(dim=-1).numpy()
    return float(normalized_mutual_info_score(true_labels, pred_labels))


def compute_downstream_f1(
    theta: torch.Tensor,
    true_labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> float:
    """
    Train a logistic regression on document-topic mixtures (theta) and
    evaluate macro-F1 on a held-out split.

    Parameters
    ----------
    theta : (n_docs, K) topic mixture tensor
    true_labels : array of class labels
    test_size : fraction of data for evaluation
    random_state : int

    Returns
    -------
    float — macro-F1
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split

    X = theta.numpy()
    y = true_labels
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    clf = LogisticRegression(max_iter=1000, random_state=random_state)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    return float(f1_score(y_te, y_pred, average="macro"))


# ---------------------------------------------------------------------------
# MC Uncertainty report
# ---------------------------------------------------------------------------

def interpret_mc_uncertainty(
    theta_final: torch.Tensor,
    theta_uncertainty: torch.Tensor,
    corpus: List[str],
    topic_names: List[str],
    high_threshold: float = 0.08,
    low_threshold: float = 0.02,
) -> pd.DataFrame:
    """
    Classify each document into an uncertainty regime based on MC samples.

    Regimes:
      CERTAIN         — std < low_threshold
      MODERATE        — low_threshold <= std <= high_threshold
      AMBIGUOUS       — std > high_threshold AND high entropy (multi-topic)
      POORLY_ENCODED  — std > high_threshold AND low entropy (out-of-vocabulary)

    Parameters
    ----------
    theta_final : (n_docs, K) mean topic mixture
    theta_uncertainty : (n_docs, K) std of MC samples
    corpus : list of document strings
    topic_names : list of topic label strings
    high_threshold, low_threshold : float thresholds on mean std

    Returns
    -------
    pd.DataFrame with one row per document.
    """
    K = theta_final.shape[1]
    mean_std = theta_uncertainty.mean(dim=1).numpy()
    entropy = -(theta_final * (theta_final + 1e-10).log()).sum(dim=1).numpy()
    max_prob = theta_final.max(dim=1).values.numpy()
    dominant = theta_final.argmax(dim=1).numpy()

    records = []
    for i in range(len(corpus)):
        std = mean_std[i]
        ent = entropy[i]
        max_ent = np.log(K)

        if std < low_threshold:
            regime = "CERTAIN"
            desc = "Stable topic assignment."
        elif std > high_threshold and ent > max_ent * 0.6:
            regime = "AMBIGUOUS"
            desc = "High std + high entropy: genuine multi-topic document."
        elif std > high_threshold and ent <= max_ent * 0.6:
            regime = "POORLY_ENCODED"
            desc = "High std + low entropy: anomalous vocabulary."
        else:
            regime = "MODERATE"
            desc = "Moderate uncertainty — qualitative review recommended."

        dom_idx = dominant[i]
        dom_name = (
            topic_names[dom_idx] if dom_idx < len(topic_names)
            else f"Topic_{dom_idx + 1}"
        )
        records.append({
            "doc_id":          i,
            "regime":          regime,
            "description":     desc,
            "mean_std_mc":     round(float(std), 4),
            "entropy_theta":   round(float(ent), 4),
            "dominant_topic":  dom_name,
            "dominant_prob":   round(float(max_prob[i]), 3),
            "text_preview":    corpus[i][:120] + "..." if len(corpus[i]) > 120 else corpus[i],
        })

    df = pd.DataFrame(records)
    print("\n[MC Uncertainty Report]")
    for regime in ["CERTAIN", "MODERATE", "AMBIGUOUS", "POORLY_ENCODED"]:
        n = (df["regime"] == regime).sum()
        print(f"  {regime:16s}: {n:5d} ({100*n/len(corpus):.1f}%)")
    return df


# ---------------------------------------------------------------------------
# Full evaluator class
# ---------------------------------------------------------------------------

class SCPTMEvaluator:
    """
    Convenience wrapper that gathers all evaluation metrics.

    Usage
    -----
    evaluator = SCPTMEvaluator(model, vocab, bow_sparse, corpus)
    report = evaluator.evaluate(theta, true_labels=labels)
    """

    def __init__(self, model, vocab: List[str], bow_sparse, corpus: List[str]):
        self.model      = model
        self.vocab      = vocab
        self.bow_sparse = bow_sparse
        self.corpus     = corpus

    def evaluate(
        self,
        theta: torch.Tensor,
        true_labels: Optional[np.ndarray] = None,
        theta_uncertainty: Optional[torch.Tensor] = None,
        theta_runs: Optional[List[torch.Tensor]] = None,
        top_k: int = 10,
    ) -> Dict:
        """
        Compute all available metrics and return a summary dict.

        Parameters
        ----------
        theta : (n_docs, K) topic mixtures
        true_labels : optional ground-truth class labels
        theta_uncertainty : optional MC uncertainty tensor
        theta_runs : optional list of theta from multiple runs (for stability)
        top_k : top words per topic for coherence

        Returns
        -------
        dict with metric names and values.
        """
        from .keywords import extract_top_words

        top_words = extract_top_words(self.model, self.vocab, top_k=top_k)
        npmi  = compute_npmi_coherence(top_words, self.bow_sparse, self.vocab, top_k)
        div   = compute_topic_diversity(top_words)

        report: Dict = {
            "npmi_coherence":  round(npmi, 4),
            "topic_diversity": round(div, 4),
            "n_topics":        self.model.num_topics,
        }

        if true_labels is not None:
            nmi  = compute_nmi(theta, true_labels)
            f1   = compute_downstream_f1(theta, true_labels)
            report["nmi"]              = round(nmi, 4)
            report["downstream_f1"]    = round(f1, 4)

        if theta_runs is not None and len(theta_runs) >= 2:
            stab = compute_stability(theta_runs)
            report["stability_ari"] = round(stab, 4)

        print("\n[Evaluation Summary]")
        for k, v in report.items():
            print(f"  {k:22s}: {v}")

        return report
