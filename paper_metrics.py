"""
paper_metrics.py
================
All quantitative metrics for the SCPTM paper, organised in four families:

  A. PREDICTIVE          perplexity / held-out log-likelihood proxy
  B. TOPIC QUALITY       gensim C_V & C_NPMI (Röder 2015 / Newman 2010),
                         topic diversity (TD), exclusivity, CD score
                         (Sciandra q3), Jensen-Shannon divergence (q6),
                         between-topic cosine (q4), WE-coherence,
                         intra-topic embedding concentration (new)
  C. MWE QUALITY         semantic compactness, specificity (IDF),
                         unigram complementarity, content ratio (POS) — NEW,
                         the paper's contribution.  Comparable across topics.
  D. RQ3 ANCHORS         valence density (VADER) on MWE vs unigrams,
                         stance/hate classifier concentration over topics.

Design principles
-----------------
* Every metric degrades gracefully: if an optional dependency (gensim,
  vaderSentiment, transformers) is missing, the function returns ``np.nan``
  instead of crashing the benchmark.
* Topic-quality metrics that need a per-topic word distribution use a single
  shared, model-agnostic ``pseudo_beta`` built from (theta, reference BoW).
  This puts LDA / CTM / SCPTM / BERTopic on identical footing.
* All per-topic metrics are averaged over topics so they are comparable
  across models and across K.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Optional dependency probing (done once, cached)
# ─────────────────────────────────────────────────────────────────────────────

try:
    from gensim.corpora import Dictionary
    from gensim.models.coherencemodel import CoherenceModel
    _HAS_GENSIM = True
except Exception:                                  # pragma: no cover
    _HAS_GENSIM = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _HAS_VADER = True
    _VADER = SentimentIntensityAnalyzer()
except Exception:                                  # pragma: no cover
    _HAS_VADER = False
    _VADER = None


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_numpy(theta) -> Optional[np.ndarray]:
    """Coerce a theta-like object to a float numpy array, or None."""
    if theta is None:
        return None
    if hasattr(theta, "detach"):
        theta = theta.detach().cpu().numpy()
    return np.asarray(theta, dtype=np.float64)


def pseudo_beta_from_theta(theta, bow_ref) -> Optional[np.ndarray]:
    """
    Build a model-agnostic topic-word weight matrix (K, V) from document-topic
    proportions and the reference BoW.

        tf[k, v] = Σ_d theta[d, k] * count[d, v]

    This is a soft, theta-weighted term-frequency table.  Both a topic-word
    distribution p(w|k) (row-normalised) and a word-topic distribution p(k|w)
    (column-normalised) can be derived from it, which feeds exclusivity, JS
    divergence and between-topic similarity in a way that is identical for
    every model.

    Returns None when theta is unavailable (e.g. BERTopic with probs=None).
    """
    theta = _to_numpy(theta)
    if theta is None:
        return None
    # bow_ref: scipy sparse (n_docs, V)
    n_docs, V = bow_ref.shape
    if theta.shape[0] != n_docs:
        # theta may have been computed on a doc subset (doc_mask) — cannot align.
        return None
    # tf[k, v] = Σ_d theta[d,k] * count[d,v].
    # Computed as (sparse.T @ dense).T — the scipy-safe ordering for
    # sparse·dense products (avoids the unreliable dense @ sparse path).
    tf = (bow_ref.T @ theta).T        # (K, V) dense
    tf = np.asarray(tf, dtype=np.float64)
    return tf


# ─────────────────────────────────────────────────────────────────────────────
# A. PREDICTIVE
# ─────────────────────────────────────────────────────────────────────────────

def perplexity_proxy(theta, pseudo_beta, bow_ref) -> float:
    """
    Document-completion-style perplexity proxy.

        p(w | d) = Σ_k theta[d,k] * p(w|k)
        perplexity = exp( - Σ_d Σ_w count[d,w] log p(w|d) / Σ_d Σ_w count[d,w] )

    Lower is better.  Defined for any model exposing theta + a topic-word
    distribution; returns np.nan otherwise (e.g. BERTopic).
    """
    theta = _to_numpy(theta)
    if theta is None or pseudo_beta is None:
        return float("nan")
    # Row-normalise pseudo_beta to p(w|k)
    row = pseudo_beta.sum(axis=1, keepdims=True)
    row[row == 0] = 1.0
    p_wk = pseudo_beta / row                       # (K, V)
    # p(w|d) = theta @ p_wk
    p_wd = theta @ p_wk                            # (n_docs, V)
    p_wd = np.clip(p_wd, 1e-12, None)
    bow = bow_ref.tocsr()
    total_tokens = bow.sum()
    if total_tokens == 0:
        return float("nan")
    # Σ count * log p, computed sparsely
    coo = bow.tocoo()
    log_lik = np.sum(coo.data * np.log(p_wd[coo.row, coo.col]))
    return float(np.exp(-log_lik / total_tokens))


# ─────────────────────────────────────────────────────────────────────────────
# B. TOPIC QUALITY
# ─────────────────────────────────────────────────────────────────────────────

def gensim_coherence(
    topic_words: List[List[str]],
    tokenized_texts: List[List[str]],
    measures: Sequence[str] = ("c_v", "c_npmi"),
) -> dict:
    """
    Compute gensim coherence measures (Röder et al. 2015).

    C_V  : sliding-window NPMI + cosine confirmation; highest human correlation.
    C_NPMI: document/window co-occurrence NPMI (Newman 2010).

    Topic words are filtered to those present in the dictionary so out-of-vocab
    tokens (e.g. bigrams with underscores) don't silently zero a topic.

    Returns {measure: float}. Missing gensim → all NaN.
    """
    out = {m: float("nan") for m in measures}
    if not _HAS_GENSIM:
        return out
    # Underscored bigrams ("climate_change") → split so they match unigram texts
    topics = []
    for ws in topic_words:
        flat = []
        for w in ws:
            flat.extend(w.split("_"))
        topics.append([t for t in flat if t])
    try:
        dictionary = Dictionary(tokenized_texts)
        # keep only words gensim knows about
        topics = [[w for w in t if w in dictionary.token2id] for t in topics]
        topics = [t for t in topics if len(t) >= 2]
        if len(topics) < 1:
            return out
        for m in measures:
            try:
                cm = CoherenceModel(
                    topics=topics,
                    texts=tokenized_texts,
                    dictionary=dictionary,
                    coherence=m,
                    processes=1,
                )
                out[m] = float(cm.get_coherence())
            except Exception:
                out[m] = float("nan")
    except Exception:
        return out
    return out


def topic_diversity(topic_words: List[List[str]]) -> float:
    """Fraction of unique words across all topic top-word lists. Higher = better."""
    all_words = [w for t in topic_words for w in t]
    if not all_words:
        return float("nan")
    return len(set(all_words)) / len(all_words)


def topic_exclusivity(pseudo_beta, top_n: int = 10) -> float:
    """
    Mean topic exclusivity (Bischof & Airoldi 2012 / STM q2).

    For each topic's top_n words, exclusivity = p(k|w) = how concentrated the
    word's probability mass is on this topic vs all others.  Averaged over
    words and topics. Range [0, 1]; higher = more distinctive topics.
    """
    if pseudo_beta is None:
        return float("nan")
    K, V = pseudo_beta.shape
    col = pseudo_beta.sum(axis=0, keepdims=True)    # (1, V)  Σ_k tf[k,w]
    col[col == 0] = 1.0
    p_kw = pseudo_beta / col                        # (K, V)  p(k|w)
    scores = []
    for k in range(K):
        top = np.argsort(pseudo_beta[k])[::-1][:top_n]
        if len(top) == 0:
            continue
        scores.append(float(np.mean(p_kw[k, top])))
    return float(np.mean(scores)) if scores else float("nan")


def cd_score(coherence_per_topic: np.ndarray, exclusivity_per_topic: np.ndarray) -> float:
    """
    Consistency-and-Differentiation score (Sciandra q3): L2 norm of the
    min-max-normalised (coherence, exclusivity) pair, averaged over topics.
    Pinpoints topics in the top-right (coherent AND exclusive) region.
    """
    c = np.asarray(coherence_per_topic, dtype=np.float64)
    e = np.asarray(exclusivity_per_topic, dtype=np.float64)
    if c.size == 0 or e.size == 0 or c.size != e.size:
        return float("nan")

    def _mm(x):
        lo, hi = np.nanmin(x), np.nanmax(x)
        return np.zeros_like(x) if hi - lo < 1e-12 else (x - lo) / (hi - lo)

    cn, en = _mm(c), _mm(e)
    return float(np.mean(np.sqrt(cn ** 2 + en ** 2)))


def per_topic_umass_and_exclusivity(pseudo_beta, bow_ref, top_n: int = 10):
    """
    Per-topic UMass coherence (Mimno 2011, in-corpus, q1) and per-topic
    exclusivity, returned as two arrays so cd_score can combine them.
    """
    if pseudo_beta is None:
        return None, None
    K, V = pseudo_beta.shape
    bow_bin = (bow_ref > 0).astype(np.float64).tocsc()
    df = np.asarray(bow_bin.sum(axis=0)).ravel() + 1.0   # smoothed doc freq
    col = pseudo_beta.sum(axis=0, keepdims=True)
    col[col == 0] = 1.0
    p_kw = pseudo_beta / col

    umass = np.full(K, np.nan)
    excl = np.full(K, np.nan)
    for k in range(K):
        top = np.argsort(pseudo_beta[k])[::-1][:top_n]
        if len(top) < 2:
            continue
        # UMass: Σ_{i<j} log( (co-df(wi,wj)+1) / df(wj) )
        s, n = 0.0, 0
        for a in range(len(top)):
            for b in range(a):
                wi, wj = top[a], top[b]
                co = bow_bin[:, wi].multiply(bow_bin[:, wj]).sum() + 1.0
                s += math.log(co / df[wj])
                n += 1
        umass[k] = s / n if n else np.nan
        excl[k] = float(np.mean(p_kw[k, top]))
    return umass, excl


def between_topic_cosine(topic_embeddings) -> float:
    """
    Mean pairwise cosine similarity between topic embedding vectors (q4).
    LOWER is better (topics far apart). Returns NaN if embeddings absent.
    """
    if topic_embeddings is None:
        return float("nan")
    M = _to_numpy(topic_embeddings)
    if M is None or M.shape[0] < 2:
        return float("nan")
    norm = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    sim = norm @ norm.T
    K = sim.shape[0]
    iu = np.triu_indices(K, k=1)
    return float(np.mean(sim[iu]))


def topic_js_divergence(pseudo_beta) -> float:
    """
    Mean pairwise Jensen-Shannon divergence between topic-word distributions
    (Deveaud 2014, q6). HIGHER is better (topics semantically separated).
    Range [0, 1] (log base 2). NaN if theta unavailable.
    """
    if pseudo_beta is None:
        return float("nan")
    row = pseudo_beta.sum(axis=1, keepdims=True)
    row[row == 0] = 1.0
    P = pseudo_beta / row                            # (K, V) distributions
    K = P.shape[0]
    if K < 2:
        return float("nan")

    def _kl(a, b):
        a = np.clip(a, 1e-12, None)
        b = np.clip(b, 1e-12, None)
        return np.sum(a * np.log2(a / b))

    vals = []
    for i in range(K):
        for j in range(i + 1, K):
            m = 0.5 * (P[i] + P[j])
            vals.append(0.5 * _kl(P[i], m) + 0.5 * _kl(P[j], m))
    return float(np.mean(vals)) if vals else float("nan")


def intra_topic_concentration(theta, doc_embeddings) -> float:
    """
    NEW. Semantic concentration of documents assigned to each topic.

    For each topic k, take its dominant documents (argmax theta == k), compute
    their embedding centroid, and measure the mean cosine similarity of each
    member to that centroid.  Averaged over topics.

    Range ~[-1, 1]; HIGHER = documents in a topic are semantically tight
    (the assignment carves out a coherent region of embedding space).
    Complementary to WE-coherence (which acts on top WORDS, not documents).
    """
    theta = _to_numpy(theta)
    emb = _to_numpy(doc_embeddings)
    if theta is None or emb is None:
        return float("nan")
    if theta.shape[0] != emb.shape[0]:
        return float("nan")
    dominant = theta.argmax(axis=1)
    K = theta.shape[1]
    norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    scores = []
    for k in range(K):
        members = norm[dominant == k]
        if len(members) < 2:
            continue
        centroid = members.mean(axis=0)
        centroid /= (np.linalg.norm(centroid) + 1e-12)
        scores.append(float(np.mean(members @ centroid)))
    return float(np.mean(scores)) if scores else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# C. MWE QUALITY  (the paper's contribution — comparable across topics)
# ─────────────────────────────────────────────────────────────────────────────

def _encode(sbert, strings: List[str]) -> Optional[np.ndarray]:
    if not strings:
        return None
    embs = sbert.encode([s.replace("_", " ") for s in strings],
                        show_progress_bar=False)
    embs = np.asarray(embs, dtype=np.float64)
    return embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)


def mwe_semantic_compactness(mwe_per_topic, topic_embeddings, sbert) -> float:
    """
    Phrase-level analogue of WE-coherence: mean cosine similarity between each
    topic's MWE phrases and the topic embedding centroid. Averaged over topics.
    HIGHER = phrases are semantically tight around the topic.
    """
    M = _to_numpy(topic_embeddings)
    if M is None or not mwe_per_topic:
        return float("nan")
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    scores = []
    for k, phrases in enumerate(mwe_per_topic):
        if k >= M.shape[0] or not phrases:
            continue
        pe = _encode(sbert, list(phrases))
        if pe is None:
            continue
        scores.append(float(np.mean(pe @ M[k])))
    return float(np.mean(scores)) if scores else float("nan")


def mwe_specificity(mwe_per_topic, df_lookup: dict, n_docs: int) -> float:
    """
    Mean inverse-document-frequency of MWE phrases (using the per-token IDF of
    the rarest content token in each phrase). HIGHER = phrases are more
    topic-specific / less generic. Comparable across topics and corpora.

    df_lookup : {token: document_frequency} over the reference corpus.
    """
    if not mwe_per_topic or not df_lookup:
        return float("nan")
    scores = []
    for phrases in mwe_per_topic:
        for ph in phrases:
            toks = ph.replace("_", " ").split()
            idfs = [math.log(n_docs / df_lookup[t]) for t in toks if t in df_lookup]
            if idfs:
                scores.append(max(idfs))   # rarest token drives specificity
    return float(np.mean(scores)) if scores else float("nan")


def mwe_unigram_complementarity(mwe_per_topic, single_per_topic, sbert) -> float:
    """
    NEW. How much *new* semantic information MWE phrases add over the single-word
    keywords of the same topic.

    For each MWE phrase, 1 - max cosine similarity to any unigram in the topic.
    Averaged over phrases and topics. HIGHER = phrases are not redundant with
    unigrams (they carry structure unigrams cannot). LOW = phrases just echo
    the single words.
    """
    if not mwe_per_topic or not single_per_topic:
        return float("nan")
    scores = []
    for phrases, singles in zip(mwe_per_topic, single_per_topic):
        if not phrases or not singles:
            continue
        pe = _encode(sbert, list(phrases))
        se = _encode(sbert, list(singles))
        if pe is None or se is None:
            continue
        sim = pe @ se.T                       # (n_phrase, n_single)
        max_sim = sim.max(axis=1)
        scores.append(float(np.mean(1.0 - max_sim)))
    return float(np.mean(scores)) if scores else float("nan")


def mwe_content_ratio(mwe_per_topic, nlp=None) -> float:
    """
    NEW. Fraction of MWE phrases that are *content-bearing* rather than
    functional collocations.  Semi-automates the manual content/functional/
    relational categorisation.

    Heuristic (POS via spaCy if available, else stopword-based):
      content-bearing  = phrase contains ≥1 NOUN/PROPN/ADJ and is not composed
                         entirely of function words.
    Returns the mean fraction over topics. HIGHER = richer phrase descriptors.
    """
    if not mwe_per_topic:
        return float("nan")
    _FUNCTION = {
        "just", "like", "know", "dont", "don", "youre", "im", "ive", "thats",
        "really", "way", "going", "got", "want", "make", "think", "thing",
        "things", "lot", "yeah", "okay", "ok", "good", "bad", "people", "said",
    }
    ratios = []
    for phrases in mwe_per_topic:
        if not phrases:
            continue
        n_content = 0
        for ph in phrases:
            text = ph.replace("_", " ")
            is_content = False
            if nlp is not None:
                try:
                    doc = nlp(text)
                    is_content = any(t.pos_ in ("NOUN", "PROPN", "ADJ") for t in doc)
                except Exception:
                    nlp = None     # fall through to heuristic on later phrases
            if nlp is None:
                toks = text.lower().split()
                is_content = any(t not in _FUNCTION and len(t) > 3 for t in toks)
            n_content += int(is_content)
        ratios.append(n_content / len(phrases))
    return float(np.mean(ratios)) if ratios else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# D. RQ3 ANCHORS
# ─────────────────────────────────────────────────────────────────────────────

def valence_density(strings: List[str]) -> float:
    """
    Mean absolute VADER valence over a list of strings (phrases or words).
    HIGHER = the descriptor set carries more evaluative / stance-laden content.
    Returns NaN if VADER is unavailable.
    """
    if not _HAS_VADER or not strings:
        return float("nan")
    vals = [abs(_VADER.polarity_scores(s.replace("_", " "))["compound"])
            for s in strings]
    return float(np.mean(vals)) if vals else float("nan")


def mwe_vs_unigram_valence(mwe_per_topic, single_per_topic) -> dict:
    """
    Compare evaluative loading of MWE phrases vs single-word keywords.

    Returns {mwe_valence, unigram_valence, valence_gap} where valence_gap =
    mwe_valence - unigram_valence. A positive gap is the RQ3 signal: phrases
    carry stance/frame that single words don't.
    """
    out = {"mwe_valence": float("nan"),
           "unigram_valence": float("nan"),
           "valence_gap": float("nan")}
    if not _HAS_VADER:
        return out
    mwe_flat = [p for t in (mwe_per_topic or []) for p in t]
    uni_flat = [w for t in (single_per_topic or []) for w in t]
    mv = valence_density(mwe_flat)
    uv = valence_density(uni_flat)
    out["mwe_valence"] = mv
    out["unigram_valence"] = uv
    if not math.isnan(mv) and not math.isnan(uv):
        out["valence_gap"] = mv - uv
    return out


# Lazy global so the (heavy) classifier is loaded at most once per process.
_STANCE_PIPE = None
_STANCE_FAILED = False


def _get_stance_pipe(model_name: str):
    global _STANCE_PIPE, _STANCE_FAILED
    if _STANCE_FAILED:
        return None
    if _STANCE_PIPE is None:
        try:
            from transformers import pipeline
            _STANCE_PIPE = pipeline(
                "text-classification", model=model_name,
                truncation=True, max_length=256, top_k=None,
            )
        except Exception as e:                      # pragma: no cover
            print(f"  [stance] classifier unavailable ({e}); skipping RQ3-D.")
            _STANCE_FAILED = True
            return None
    return _STANCE_PIPE


def stance_concentration(
    theta,
    docs: List[str],
    model_name: str = "cardiffnlp/twitter-roberta-base-hate-latest",
    max_docs_per_topic: int = 40,
) -> float:
    """
    NEW (RQ3-D). For each topic, run a pre-trained stance/hate classifier over
    its dominant documents and measure how *concentrated* the predicted-label
    distribution is:  concentration = 1 - H(labels) / log(n_labels).

    HIGHER = a topic's documents share a consistent stance (the model carved out
    a stance-coherent region, not just a lexical one). Averaged over topics.
    Returns NaN if transformers / the model is unavailable.
    """
    theta = _to_numpy(theta)
    if theta is None or not docs:
        return float("nan")
    pipe = _get_stance_pipe(model_name)
    if pipe is None:
        return float("nan")
    dominant = theta.argmax(axis=1)
    K = theta.shape[1]
    rng = np.random.default_rng(0)
    concentrations = []
    for k in range(K):
        idx = np.where(dominant == k)[0]
        if len(idx) < 5:
            continue
        if len(idx) > max_docs_per_topic:
            idx = rng.choice(idx, size=max_docs_per_topic, replace=False)
        batch = [docs[i][:1000] for i in idx]
        try:
            preds = pipe(batch)
        except Exception:
            continue
        # top-1 label per doc
        labels = []
        for p in preds:
            if isinstance(p, list):
                p = max(p, key=lambda d: d["score"])
            labels.append(p["label"])
        # entropy of label distribution
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / counts.sum()
        H = -np.sum(probs * np.log(probs + 1e-12))
        n_lab = max(len(counts), 2)
        concentrations.append(1.0 - H / math.log(n_lab))
    return float(np.mean(concentrations)) if concentrations else float("nan")
