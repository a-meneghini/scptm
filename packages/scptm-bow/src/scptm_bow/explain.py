"""
scptm_bow/explain.py
---------------------
Multiword expressions and readable topic labels, straight from the dependency
graph collected during parsing.

This is the part of the "syntactic" story that keeps its value at almost no
cost: the triples (head, relation, dependent) are already aggregated, so
extracting MWEs is a dictionary scan — no re-parsing, no per-topic document
reprocessing.
"""

from collections import Counter
from typing import Dict, List, Sequence

# Relations that typically form a lexical unit rather than a clause.
MWE_RELATIONS = frozenset({"amod", "compound", "nmod", "flat", "fixed", "det"})


def extract_mwes(
    topic_words: Sequence[Sequence[str]],
    dep_counts: Dict,
    top_k: int = 15,
    relations: frozenset = MWE_RELATIONS,
) -> Dict[str, List[Dict[str, object]]]:
    """
    Extract multiword expressions per topic.

    Candidates are dependency pairs whose head *and* dependent are among the
    topic's top words (strong candidates), plus half-weighted pairs with only
    one word in the topic.  The relation label is kept so a phrase can be
    inspected or filtered by type.

    Parameters
    ----------
    topic_words : sequence of sequences of str
        Top words per topic, ordered by relevance.
    dep_counts : dict
        ``{(head_lemma, relation, dependent_lemma): count}`` from the parse pass.
    top_k : int
        Maximum MWEs returned per topic.
    relations : frozenset
        Relations considered "lexical" (see :data:`MWE_RELATIONS`).

    Returns
    -------
    dict
        ``{"Topic_1": [{"phrase", "relation", "score", "count", "in_topic"}, …]}``
    """
    results: Dict[str, List[Dict[str, object]]] = {}

    for k, words in enumerate(topic_words):
        topic_set = set(words)
        best: Dict[str, Dict[str, object]] = {}

        for (head, relation, dependent), count in dep_counts.items():
            if relation not in relations:
                continue
            in_head = head in topic_set
            in_dep = dependent in topic_set
            if not (in_head or in_dep):
                continue
            # Both words on-topic is much stronger evidence than one.
            weight = 1.0 if (in_head and in_dep) else 0.5
            phrase = f"{head} {dependent}"
            score = float(count) * weight
            previous = best.get(phrase)
            if previous is None or score > previous["score"]:
                best[phrase] = {
                    "phrase": phrase,
                    "relation": relation,
                    "score": score,
                    "count": int(count),
                    "in_topic": int(in_head) + int(in_dep),   # 1 or 2
                }

        ranked = sorted(best.values(), key=lambda item: -item["score"])[:top_k]
        results[f"Topic_{k + 1}"] = ranked

    return results


def topic_labels(
    topic_words: Sequence[Sequence[str]],
    mwes: Dict[str, List[Dict[str, object]]],
    n: int = 3,
) -> List[str]:
    """
    Short human-readable labels: the best MWE per topic plus top single words.

    Purely for plotting and tables; no extra computation.
    """
    labels: List[str] = []
    for k, words in enumerate(topic_words):
        parts = [m["phrase"] for m in mwes.get(f"Topic_{k + 1}", [])[:1]]
        parts.extend(w for w in words[:n] if w not in parts)
        labels.append(", ".join(parts[:n]))
    return labels


def mwe_compactness(
    mwes: Dict[str, List[Dict[str, object]]],
    vectors,
    vocab: Sequence[str],
    top_k: int = 10,
) -> float:
    """
    Mean cosine similarity between the two words of each MWE.

    A phrase whose parts are semantically close (e.g. "spirito santo") scores
    high; a syntactically-linked but semantically loose pair scores low.  Cheap
    because it only needs the word vectors already computed.
    """
    if vectors is None:
        return float("nan")
    import numpy as np

    index = {w: i for i, w in enumerate(vocab)}
    unit = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10)
    sims: List[float] = []
    for phrases in mwes.values():
        for item in phrases[:top_k]:
            a, _, b = str(item["phrase"]).partition(" ")
            ia, ib = index.get(a), index.get(b)
            if ia is None or ib is None:
                continue
            sims.append(float(np.dot(unit[ia], unit[ib])))
    return float(np.mean(sims)) if sims else float("nan")


def mwe_counts(mwes: Dict[str, List[Dict[str, object]]]) -> Counter:
    """How many MWEs were produced per topic (a quick coverage check)."""
    return Counter({k: len(v) for k, v in mwes.items()})
