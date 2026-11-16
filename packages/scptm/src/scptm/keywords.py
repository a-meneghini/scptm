"""
scptm/keywords.py
-----------------
Keyword and multi-word expression (MWE) extraction from trained SCPTM.
Includes RAKE keyphrase extraction from top-ranked segments per topic.
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union

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


def _build_mwe_vocab_from_dep_triples(
    dep_triples: Dict[Tuple[str, str, str], int],
    orig_vocab: List[str],
    orig_word_embs: torch.Tensor,
    min_df: int,
) -> Tuple[Optional[np.ndarray], Optional[torch.Tensor], Optional[List[str]]]:
    """
    Build the MWE candidate set from syntactic dependency triples instead of
    surface n-grams.

    Each retained triple (head_lemma, dep_label, dependent_lemma) becomes one
    phrase "head dependent", scored later by the mean of the two constituent
    word embeddings — no extra SBERT encoding pass needed, since both words
    are already embedded in `orig_word_embs`.

    `min_df` here is a minimum raw occurrence count across the corpus (how
    many times that exact syntactic pair was parsed), not a document-count
    as in the n-gram fallback — a triple repeated many times within one long
    document already indicates a stable syntactic pattern.

    Returns (None, None, None) if no triple survives filtering, signalling
    the caller to fall back to the n-gram extractor.
    """
    vocab_idx = {w: i for i, w in enumerate(orig_vocab)}
    kept = [
        (h, r, d) for (h, r, d), cnt in dep_triples.items()
        if cnt >= min_df and h in vocab_idx and d in vocab_idx
    ]
    if not kept:
        return None, None, None

    mwe_vocab = np.array([f"{h} {d}" for h, _, d in kept])
    head_idx = [vocab_idx[h] for h, _, _ in kept]
    dep_idx  = [vocab_idx[d] for _, _, d in kept]
    mwe_embs_tensor = (orig_word_embs[head_idx] + orig_word_embs[dep_idx]) / 2
    mwe_relations = [r for _, r, _ in kept]
    return mwe_vocab, mwe_embs_tensor, mwe_relations


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
    dep_triples: Optional[Dict[Tuple[str, str, str], int]] = None,
) -> Tuple[dict, torch.Tensor, np.ndarray]:
    """
    Separate keyword extraction for single words and multi-word expressions (MWE).

    For each topic, returns the top-k single lemmas AND top-k MWEs.
    Single words are ranked by cosine similarity (default) or c-TF-IDF.
    Phrases are always ranked by cosine similarity.

    MWEs are extracted from the corpus's syntactic dependency graph when
    `dep_triples` is provided (see graph.build_hetero_graph): each candidate
    is a (head, dependent) pair connected by a retained dependency relation
    (nsubj, obj, amod, ...), scored by the mean cosine similarity of the two
    words' embeddings to the topic embedding — this is what Section 4.5
    describes. When `dep_triples` is empty or None (graph_mode in {"none",
    "no_syntax"}, or an edge cache written before this field existed), MWEs
    fall back to surface n-grams (CountVectorizer) over the raw corpus text,
    which capture textual co-occurrence but not syntactic structure.

    Parameters
    ----------
    corpus : list of document strings
    model : trained VariationalGraphTopicModel
    orig_vocab : vocabulary list
    orig_word_embs : static SBERT word embeddings on the model's device
    sbert_model : SentenceTransformer instance
    stop_words : stop word list / string for CountVectorizer (n-gram fallback)
    top_k : int
    min_df : int
        For n-gram MWEs: minimum document frequency (CountVectorizer semantics).
        For syntactic MWEs: minimum raw occurrence count of the triple.
    method : "cosine" | "ctfidf"
        Ranking method for single words.
    bow_sparse : scipy sparse matrix — required when method="ctfidf".
    theta : torch.Tensor — required when method="ctfidf".
    dep_triples : {(head_lemma, dep_label, dependent_lemma): count}, optional
        From graph.build_hetero_graph(). Enables syntactic MWE extraction.

    Returns
    -------
    topics_dict : {TopicN: {"single": [...], "phrases": [...],
                             "phrase_relations": [...] | None}}
        "phrase_relations" holds the UD dependency label behind each phrase
        in the same order as "phrases", or None when the n-gram fallback
        was used (no relation label available).
    mwe_embs_tensor : torch.Tensor of MWE embeddings
    mwe_vocab : np.ndarray of MWE strings
    """
    print("\nExtracting single words + MWEs per topic...")

    mwe_relations_full: Optional[List[str]] = None
    if dep_triples:
        mwe_vocab, mwe_embs_tensor, mwe_relations_full = _build_mwe_vocab_from_dep_triples(
            dep_triples, orig_vocab, orig_word_embs, min_df
        )
        if mwe_vocab is None:
            warnings.warn(
                "No syntactic dependency triples survived min_df filtering; "
                "falling back to surface n-grams for MWE extraction."
            )

    if not dep_triples or mwe_vocab is None:
        print("  MWE source: surface n-grams (no syntactic graph available).")
        vectorizer = CountVectorizer(
            ngram_range=(2, 3), min_df=min_df, stop_words=stop_words
        )
        vectorizer.fit(corpus)
        mwe_vocab = vectorizer.get_feature_names_out()

        mwe_embs = sbert_model.encode(mwe_vocab.tolist(), show_progress_bar=False)
        mwe_embs_tensor = torch.tensor(mwe_embs, dtype=torch.float32).to(
            orig_word_embs.device
        )
        mwe_relations_full = None
    else:
        print(f"  MWE source: syntactic dependency graph ({len(mwe_vocab)} candidate phrases).")
        mwe_embs_tensor = mwe_embs_tensor.to(orig_word_embs.device)

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
            top_idx = sim_mwe.argsort(descending=True)[:top_k].tolist()
            top_mwe = [mwe_vocab[i] for i in top_idx]
            top_relations = (
                [mwe_relations_full[i] for i in top_idx]
                if mwe_relations_full is not None else None
            )

            topics_dict[f"Topic_{k + 1}"] = {
                "single":           top_single,
                "phrases":          top_mwe,
                "phrase_relations": top_relations,
            }

    return topics_dict, mwe_embs_tensor, mwe_vocab


# ---------------------------------------------------------------------------
# RAKE keyword extraction
# ---------------------------------------------------------------------------

def extract_rake_keywords(
    corpus: List[str],
    theta: torch.Tensor,
    stop_words: Union[List[str], str],
    top_n_docs: int = 30,
    top_k: int = 10,
) -> Dict[str, List[str]]:
    """
    RAKE keyphrase extraction from top-ranked segments per topic.

    For each topic k, selects the top_n_docs segments by theta_k score,
    concatenates their text, and runs RAKE to extract keyphrases.
    This is corpus-driven and complements cosine-similarity-based MWEs.

    Parameters
    ----------
    corpus : list of str
        Segment strings (model._corpus).
    theta : torch.Tensor, shape (N, K)
        Topic mixtures (model._theta).
    stop_words : list of str or "english"
        Stop words from the NLP pipeline.
    top_n_docs : int
        Number of top segments to aggregate per topic.
    top_k : int
        Number of keyphrases to return per topic.

    Returns
    -------
    dict : {"Topic_k": [keyphrase1, ...]}
    """
    try:
        from rake_nltk import Rake
    except ImportError:
        raise ImportError(
            "rake-nltk is required: pip install rake-nltk"
        )

    sw_list = None if (isinstance(stop_words, str) and stop_words == "english") \
        else list(stop_words)

    theta_np = theta.numpy() if hasattr(theta, "numpy") else np.array(theta)
    K = theta_np.shape[1]
    results: Dict[str, List[str]] = {}

    for k in range(K):
        top_idx = np.argsort(theta_np[:, k])[::-1][:top_n_docs]
        text = " ".join(corpus[i] for i in top_idx)

        rake = Rake(stopwords=sw_list, min_length=1, max_length=4)
        rake.extract_keywords_from_text(text)
        phrases = rake.get_ranked_phrases()[:top_k]
        results[f"Topic_{k + 1}"] = phrases

    return results
