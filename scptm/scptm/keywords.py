"""
scptm/keywords.py
-----------------
Keyword and multi-word expression (MWE) extraction from trained SCPTM.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer


# ---------------------------------------------------------------------------
# c-TF-IDF scoring
# ---------------------------------------------------------------------------

def compute_ctfidf_scores(
    bow_sparse,
    theta: torch.Tensor,
    num_topics: int,
) -> np.ndarray:
    """
    Compute c-TF-IDF scores (K x V) treating each topic as a class.

    Each document is assigned to its dominant topic. Words are scored by
    how characteristic they are of a topic relative to the full corpus.

    Parameters
    ----------
    bow_sparse : scipy sparse matrix, shape (n_docs, V)
        Raw BoW counts.
    theta : torch.Tensor, shape (n_docs, K)
        Topic mixtures from the trained model.
    num_topics : int

    Returns
    -------
    scores : np.ndarray, shape (K, V)
    """
    V = bow_sparse.shape[1]
    dominant = theta.argmax(dim=-1).numpy()

    # Aggregate BoW counts per topic class (sparse row slicing)
    class_bow = np.zeros((num_topics, V), dtype=np.float64)
    for k in range(num_topics):
        mask = dominant == k
        if mask.any():
            class_bow[k] = np.asarray(bow_sparse[mask].sum(axis=0)).flatten()

    # TF: word frequency within the class
    class_totals = class_bow.sum(axis=1, keepdims=True).clip(min=1.0)
    tf = class_bow / class_totals

    # IDF: log(1 + K / number of classes containing the word)
    word_in_classes = (class_bow > 0).sum(axis=0).clip(min=1)
    idf = np.log(1.0 + num_topics / word_in_classes)

    return tf * idf  # (K, V)


# ---------------------------------------------------------------------------
# Top-word extraction
# ---------------------------------------------------------------------------

def extract_top_words(
    model,
    vocab: Optional[List[str]],
    top_k: int = 10,
    method: str = "cosine",
    bow_sparse=None,
    theta: Optional[torch.Tensor] = None,
) -> List[List[str]]:
    """
    Extract top-k words per topic.

    Parameters
    ----------
    model : VariationalGraphTopicModel
    vocab : list of strings (vocabulary), or None for index-based output
    top_k : int
    method : "cosine" | "ctfidf"
        "cosine"  — ranks by the cached beta matrix (cosine-similarity based).
        "ctfidf"  — ranks by c-TF-IDF; requires `bow_sparse` and `theta`.
    bow_sparse : scipy sparse matrix, shape (n_docs, V)
        Required when method="ctfidf".
    theta : torch.Tensor, shape (n_docs, K)
        Required when method="ctfidf".

    Returns
    -------
    List of lists: one list of word strings (or indices) per topic.
    """
    if method == "ctfidf":
        if bow_sparse is None or theta is None:
            raise ValueError("bow_sparse and theta are required for method='ctfidf'.")
        scores = compute_ctfidf_scores(bow_sparse, theta, model.num_topics)
        topics = []
        for k in range(model.num_topics):
            top_idx = np.argsort(scores[k])[::-1][:top_k].tolist()
            if vocab is not None:
                topics.append([vocab[i] for i in top_idx])
            else:
                topics.append(top_idx)
        return topics

    # Default: cosine (beta-based)
    model.eval()
    with torch.no_grad():
        if model._cached_beta is None:
            return [[] for _ in range(model.num_topics)]
        beta = model._cached_beta.cpu()
        topics = []
        for k in range(model.num_topics):
            top_idx = beta[k].argsort(descending=True)[:top_k].tolist()
            if vocab is not None:
                topics.append([vocab[i] for i in top_idx])
            else:
                topics.append(top_idx)
    return topics


def extract_separated_topics(
    corpus: List[str],
    model,
    orig_vocab: List[str],
    orig_word_embs: torch.Tensor,
    sbert_model,
    stop_words,
    top_k: int = 5,
    min_df: int = 5,
    method: str = "cosine",
    bow_sparse=None,
    theta: Optional[torch.Tensor] = None,
) -> Tuple[dict, torch.Tensor, np.ndarray]:
    """
    Separate keyword extraction for single words and multi-word expressions (MWE).

    For each topic, returns the top-k single lemmas AND top-k bigrams/trigrams.
    Single words are ranked by cosine similarity (default) or c-TF-IDF.
    Phrases are always ranked by cosine similarity (no BoW signal for MWEs).

    Parameters
    ----------
    corpus : list of document strings
    model : trained VariationalGraphTopicModel
    orig_vocab : vocabulary list
    orig_word_embs : static SBERT word embeddings on the model's device
    sbert_model : SentenceTransformer instance
    stop_words : stop word list / string for CountVectorizer
    top_k : int
    min_df : int
    method : "cosine" | "ctfidf"
        Ranking method for single words.
    bow_sparse : scipy sparse matrix — required when method="ctfidf".
    theta : torch.Tensor — required when method="ctfidf".

    Returns
    -------
    topics_dict : {TopicN: {"single": [...], "phrases": [...]}}
    mwe_embs_tensor : torch.Tensor of MWE embeddings
    mwe_vocab : np.ndarray of MWE strings
    """
    print("\nExtracting single words + MWEs per topic...")
    vectorizer = CountVectorizer(
        ngram_range=(2, 3), min_df=min_df, stop_words=stop_words
    )
    vectorizer.fit(corpus)
    mwe_vocab = vectorizer.get_feature_names_out()

    mwe_embs = sbert_model.encode(mwe_vocab.tolist(), show_progress_bar=False)
    mwe_embs_tensor = torch.tensor(mwe_embs, dtype=torch.float32).to(
        orig_word_embs.device
    )

    # Pre-compute c-TF-IDF scores once if needed
    ctfidf_scores = None
    if method == "ctfidf":
        if bow_sparse is None or theta is None:
            raise ValueError("bow_sparse and theta are required for method='ctfidf'.")
        ctfidf_scores = compute_ctfidf_scores(bow_sparse, theta, model.num_topics)

    topics_dict = {}
    model.eval()
    with torch.no_grad():
        for k in range(model.num_topics):
            topic_emb = model.topic_embeddings[k]

            # Single words
            if ctfidf_scores is not None:
                top_single = [
                    orig_vocab[i]
                    for i in np.argsort(ctfidf_scores[k])[::-1][:top_k].tolist()
                ]
            else:
                sim_single = F.cosine_similarity(
                    topic_emb.unsqueeze(0), orig_word_embs
                )
                top_single = [
                    orig_vocab[i]
                    for i in sim_single.argsort(descending=True)[:top_k].tolist()
                ]

            # Phrases — always cosine (no unigram BoW signal available)
            sim_mwe = F.cosine_similarity(
                topic_emb.unsqueeze(0), mwe_embs_tensor
            )
            top_mwe = [
                mwe_vocab[i]
                for i in sim_mwe.argsort(descending=True)[:top_k].tolist()
            ]

            topics_dict[f"Topic_{k + 1}"] = {
                "single":  top_single,
                "phrases": top_mwe,
            }

    return topics_dict, mwe_embs_tensor, mwe_vocab
