"""
scptm/graph.py
--------------
Corpus preprocessing, vocabulary building, and heterogeneous graph construction.

Public functions
----------------
prepare_corpus(source, source_type, ...) → List[str]
    Load and optionally chunk a text corpus from a folder, DataFrame, or list.

build_hetero_graph(documents, sbert_model, nlp_model, stop_words, cfg,
                   edge_cache_path=None)
    → (HeteroData, vocab_list, bow_sparse, n_dw, n_ww)

    Builds a heterogeneous PyG graph with node types "doc" and "word" and
    edge types:
      ("doc",  "contains",     "word")  — doc-word co-occurrence
      ("word", "rev_contains", "doc")   — mirrored
      ("word", "relates",      "word")  — syntactic dependency edges

    When edge_cache_path is provided:
      * First run  — parses corpus, encodes SBERT embeddings, writes all to cache.
      * Later runs — loads from cache, skips lemmatisation/parsing AND SBERT
        encode entirely.
      * A cache is reused only when its signature matches (see
        :func:`_cache_signature`): corpus content hash, lang, lemmatizer,
        graph_mode, min_df and max_features.  Any change triggers an automatic
        rebuild with a warning naming the changed field.

collect_contextual_embeddings(documents, nlp_model, sbert_model, vocab,
                               max_occurrences_per_word) → dict
    Returns word → Tensor(N, D) of SBERT document embeddings for contexts
    in which each vocabulary word appears.

load_ctx_embs_from_cache(path, vocab_size) → list | None
save_ctx_embs_to_cache(path, ctx_embs_list)
    Piggy-back the contextual embeddings onto the parse cache pickle,
    so the expensive SBERT contextual pass is also skipped on reload.

estimate_graph_memory(n_docs, vocab_size, n_edges_dw, n_edges_ww, ...) → dict
    Estimate GPU memory for the heterogeneous graph (prints a warning if
    the estimate exceeds 8 GB).
"""

import hashlib
import math
import os
import pickle
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import CountVectorizer
from torch_geometric.data import HeteroData
from tqdm import tqdm

from .config import (
    ALL_CONTENT_DEP_TYPES,
    GRAPH_MODES,
    INFORMATIVE_DEP_TYPES,
    SCPTMConfig,
)


# Token pattern for the vocabulary CountVectorizer: at least two Unicode
# letters, no digits or underscores.
#
# The previous pattern was r"(?u)\b[a-zA-Z][a-zA-Z]+\b", which silently dropped
# every accented word — in Italian that meant losing "città", "però", "più",
# "società", "università", … because there is no word boundary between "t" and
# "à" (both are word characters under Unicode).  [^\W\d_] is "a word character
# that is neither a digit nor an underscore", i.e. any Unicode letter.
VOCAB_TOKEN_PATTERN = r"(?u)\b[^\W\d_]{2,}\b"

# Minimum length (characters, after whitespace normalisation) for a raw
# document to become a segment.  Documents at or below this are dropped, both
# with and without chunking, so the two paths stay consistent.
MIN_DOC_CHARS = 10


# ---------------------------------------------------------------------------
# Corpus helpers
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> List[str]:
    """Split on sentence-final punctuation followed by whitespace.

    The previous implementation used ``text.split(". ")``, which missed "? ",
    "! " and ".<newline>" and then re-appended a "." to every chunk, producing
    doubled punctuation such as "..".
    """
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s]


def prepare_corpus(
    source,
    source_type: str = "folder",
    text_col: Optional[str] = None,
    apply_chunking: bool = True,
    max_chunk_chars: int = 800,
    return_doc_indices: bool = False,
) -> Union[List[str], Tuple[List[str], List[int]]]:
    """
    Load and optionally chunk a text corpus.

    Parameters
    ----------
    source : str | pd.DataFrame
        Path to a folder of .txt files, or a DataFrame.
    source_type : str
        "folder", "dataframe" or "list".
    text_col : str | None
        Column name when source_type == "dataframe".
    apply_chunking : bool
        Split long documents into shorter segments.
    max_chunk_chars : int
        Maximum characters per chunk.
    return_doc_indices : bool
        If True, also return a list mapping each segment to its source
        document index (useful for aligning per-document covariates).

    Notes
    -----
    Documents shorter than :data:`MIN_DOC_CHARS` characters are dropped, with
    and without chunking.  Every other document contributes at least one
    segment (the trailing chunk of a long document is kept even when short).
    Dropped documents are absent from ``doc_indices`` too, so covariate
    alignment stays consistent.

    Returns
    -------
    List[str]
        Processed document strings.
    (List[str], List[int])
        Segments + doc_indices when return_doc_indices=True.
    """
    raw: List[str] = []

    if source_type == "folder":
        if not os.path.exists(source):
            raise FileNotFoundError(f"Directory '{source}' not found.")
        for fn in sorted(os.listdir(source)):
            if fn.endswith(".txt"):
                with open(os.path.join(source, fn), "r", encoding="utf-8", errors="ignore") as f:
                    raw.append(f.read().strip())

    elif source_type == "dataframe":
        if not isinstance(source, pd.DataFrame) or text_col not in source.columns:
            raise ValueError("Provide a valid DataFrame and the text column name.")
        raw = source[text_col].dropna().astype(str).tolist()

    elif source_type == "list":
        if not isinstance(source, list):
            raise ValueError("source must be a list of strings when source_type='list'.")
        raw = [str(s) for s in source]

    else:
        raise ValueError("source_type must be 'folder', 'dataframe', or 'list'.")

    print(f"Loaded {len(raw)} raw documents.")

    if not apply_chunking:
        docs: List[str] = []
        doc_indices: List[int] = []
        for doc_idx, t in enumerate(raw):
            if len(t.strip()) > MIN_DOC_CHARS:
                docs.append(" ".join(t.split()))
                doc_indices.append(doc_idx)
        return (docs, doc_indices) if return_doc_indices else docs

    docs = []
    doc_indices = []
    for doc_idx, text in enumerate(raw):
        clean = " ".join(text.split())
        # Same minimum length as the no-chunking branch.  The trailing chunk is
        # always kept: the old `if len(chunk) > 50` silently discarded the last
        # piece of every document — and *entire* documents under 50 characters,
        # including their entry in doc_indices.
        if len(clean) <= MIN_DOC_CHARS:
            continue
        sentences = _split_sentences(clean)
        chunk = ""
        for sent in sentences:
            if not sent:
                continue
            if chunk and len(chunk) + 1 + len(sent) > max_chunk_chars:
                docs.append(chunk)
                doc_indices.append(doc_idx)
                chunk = sent
            else:
                chunk = f"{chunk} {sent}" if chunk else sent
        if chunk:
            docs.append(chunk)
            doc_indices.append(doc_idx)

    print(f"Final corpus: {len(docs)} segments.")
    return (docs, doc_indices) if return_doc_indices else docs


# ---------------------------------------------------------------------------
# Contextual embeddings per word
# ---------------------------------------------------------------------------

def _iter_parsed(nlp_model, documents: List[str], batch_size: int = 64):
    """
    Iterate parsed documents, batching through ``nlp_model.pipe`` when available.

    ``spacy.Language.pipe`` batches tokenisation/tagging/parsing and is ~1.3x
    faster than a per-document loop; :class:`scptm.nlp.UDPipePipeline` exposes
    the same method for API parity (sequential internally).
    """
    pipe = getattr(nlp_model, "pipe", None)
    if callable(pipe):
        yield from pipe(documents, batch_size=batch_size)
    else:                                   # pragma: no cover - custom backend
        for text in documents:
            yield nlp_model(text)


def collect_contextual_embeddings(
    documents: List[str],
    nlp_model,
    sbert_model,
    vocab: dict,
    max_occurrences_per_word: int = 50,
    random_state: int = 42,
    doc_embeddings=None,
    batch_size: int = 64,
) -> dict:
    """
    For each vocabulary word, collect up to `max_occurrences_per_word` SBERT
    document embeddings from contexts where the word appears.

    Parameters
    ----------
    random_state : int
        Seed for the per-word subsampling.  Was hardcoded to 42, which made the
        pass reproducible but not controllable via ``SCPTMConfig.random_state``.
    doc_embeddings : array-like | None
        Pre-computed SBERT document embeddings, one row per document (the same
        matrix :func:`build_hetero_graph` already stores on the ``doc`` nodes).
        When provided, SBERT is not called at all.  When ``None`` the documents
        are encoded **in one batched call** (the previous implementation
        encoded them one at a time, ~8x slower).
    batch_size : int
        Batch size for both the SBERT encoding and ``nlp_model.pipe``.

    Returns
    -------
    dict
        word -> torch.Tensor of shape (N, emb_dim)
    """
    print("Collecting contextual embeddings (REV-A)...")
    ctx_embs = defaultdict(list)

    if doc_embeddings is None:
        doc_embeddings = sbert_model.encode(
            documents, batch_size=batch_size, show_progress_bar=True
        )
    if len(doc_embeddings) != len(documents):
        raise ValueError(
            f"doc_embeddings has {len(doc_embeddings)} rows but there are "
            f"{len(documents)} documents."
        )

    parsed_docs = _iter_parsed(nlp_model, documents, batch_size=batch_size)
    for doc_vec, parsed in zip(
        tqdm(doc_embeddings, desc="Contextual embeddings"), parsed_docs
    ):
        seen = set()
        for token in parsed:
            if token.is_stop or not token.is_alpha or len(token.lemma_) <= 2:
                continue
            lemma = token.lemma_.lower()
            if lemma not in vocab or lemma in seen:
                continue
            seen.add(lemma)
            ctx_embs[lemma].append(doc_vec)

    # Subsample to max_occurrences_per_word
    rng = np.random.default_rng(random_state)
    for w in ctx_embs:
        if len(ctx_embs[w]) > max_occurrences_per_word:
            idx = rng.choice(len(ctx_embs[w]), max_occurrences_per_word, replace=False)
            ctx_embs[w] = [ctx_embs[w][i] for i in idx]

    ctx_tensor = {
        w: torch.tensor(np.stack(vecs), dtype=torch.float32)
        for w, vecs in ctx_embs.items()
    }
    coverage = len(ctx_tensor) / max(len(vocab), 1)
    print(f"  Vocabulary coverage: {coverage:.1%} ({len(ctx_tensor)}/{len(vocab)} lemmas)")
    return ctx_tensor


# ---------------------------------------------------------------------------
# Parse cache (edge list + vocabulary) — avoid re-running the NLP parse on reload
# ---------------------------------------------------------------------------

def _corpus_hash(documents: List[str]) -> str:
    """Short, order-sensitive hash of the corpus contents.

    Updated incrementally so a huge corpus never needs a second full copy in
    memory.  Captures chunking/content changes that ``n_docs`` alone misses
    (e.g. editing the text without changing the document count).
    """
    h = hashlib.sha1()
    for doc in documents:
        h.update(str(doc).encode("utf-8", errors="replace"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def _cache_signature(cfg: SCPTMConfig, documents: List[str]) -> dict:
    """
    Everything the parse cache depends on, so a stale cache can never be
    silently reused.

    Cached artifacts and their inputs:
      * vocab_arr / bow_sparse      → corpus content, lang, lemmatizer,
                                      min_df, max_features, token pattern
      * doc/word-word edge lists    → graph_mode
      * doc_embs / word_embs_static → corpus content, vocabulary, lang (SBERT)
    """
    return {
        "content_hash":  _corpus_hash(documents),
        "n_docs":        len(documents),
        "lang":          getattr(cfg, "lang", "eng"),
        "lemmatizer":    getattr(cfg, "lemmatizer", "spacy"),
        "graph_mode":    getattr(cfg, "graph_mode", "filtered"),
        "min_df":        getattr(cfg, "min_df", None),
        "max_features":  getattr(cfg, "max_features", None),
        # Guards against silently reusing a vocabulary built by an older
        # tokenizer rule (e.g. the ASCII-only pattern that dropped accented
        # Italian words).  Bump/change the pattern and caches rebuild.
        "token_pattern": VOCAB_TOKEN_PATTERN,
    }


def _signature_diff(cached: dict, current: dict) -> List[str]:
    """Return human-readable reasons why ``cached`` no longer matches ``current``."""
    reasons = []
    for key, value in current.items():
        old = cached.get(key)
        if old == value:
            continue
        if key == "content_hash":
            reasons.append("corpus content changed (same document count)")
        else:
            reasons.append(f"{key}: {old!r} -> {value!r}")
    return reasons


def _save_parse_cache(
    path: Union[str, Path],
    vocab_arr: np.ndarray,
    bow_sparse,
    doc_word_src: list,
    doc_word_dst: list,
    word_word_src: list,
    word_word_dst: list,
    n_docs: int,
    doc_embs: Optional[np.ndarray] = None,
    word_embs_static: Optional[np.ndarray] = None,
    dep_triples: Optional[dict] = None,
    signature: Optional[dict] = None,
) -> None:
    """Persist the NLP-heavy outputs so they can be reused across runs.

    doc_embs and word_embs_static are optional; when provided they are stored
    so that subsequent calls to build_hetero_graph can skip SBERT entirely.

    ``signature`` records the corpus hash and every config field the cached
    artifacts depend on (see :func:`_cache_signature`), so a cache built with a
    different backend, graph mode, vocabulary setting or corpus is never
    silently reused.
    """
    cache = {
        "_n_docs":       n_docs,
        "vocab_arr":     vocab_arr,
        "bow_sparse":    bow_sparse,
        "doc_word_src":  doc_word_src,
        "doc_word_dst":  doc_word_dst,
        "word_word_src": word_word_src,
        "word_word_dst": word_word_dst,
    }
    if signature is not None:
        cache["_signature"] = signature
        # Flat keys too: the contextual-embedding cache validates them without
        # needing the whole signature.
        cache["_lemmatizer"] = signature.get("lemmatizer", "spacy")
        cache["_lang"]       = signature.get("lang", "eng")
    if doc_embs is not None:
        cache["doc_embs"] = doc_embs
    if word_embs_static is not None:
        cache["word_embs_static"] = word_embs_static
    if dep_triples is not None:
        cache["dep_triples"] = dep_triples
    with open(path, "wb") as f:
        pickle.dump(cache, f)
    print(f"  [EdgeCache] Saved to '{path}'")


def load_ctx_embs_from_cache(
    path: Union[str, Path],
    vocab_size: int,
    lemmatizer: str = "spacy",
    lang: str = "eng",
    max_occurrences: Optional[int] = None,
) -> Optional[list]:
    """
    Try to load a previously saved ``ctx_embs_list`` from the parse cache.

    Returns the list (one entry per vocabulary word) if the cache file exists
    **and** was built with the same vocabulary size, lemmatisation backend and
    ``max_ctx_occurrences``, otherwise returns ``None`` (triggering a fresh
    ``collect_contextual_embeddings`` call).

    Parameters
    ----------
    path : str | Path
        Same file used for the parse cache (ctx embeddings are piggy-backed
        onto the same pickle).
    vocab_size : int
        Expected number of entries in the list.
    lemmatizer : str
        Active lemmatisation backend; contextual embeddings depend on the
        tokenisation/lemmas that produced the vocabulary.
    lang : str
        Active language code.
    max_occurrences : int | None
        Active ``SCPTMConfig.max_ctx_occurrences``; the stored list was
        subsampled with it, so changing it invalidates the cache.
    """
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        cache = pickle.load(f)
    ctx = cache.get("ctx_embs_list")
    if ctx is None:
        return None
    if (cache.get("_lemmatizer", "spacy"), cache.get("_lang", "eng")) != (lemmatizer, lang):
        warnings.warn(
            "[CtxCache] Cache was built with a different lemmatisation backend. "
            "Ignoring cached contextual embeddings."
        )
        return None
    if cache.get("_ctx_vocab_size") != vocab_size:
        warnings.warn(
            f"[CtxCache] Vocabulary size changed ({cache.get('_ctx_vocab_size')} "
            f"→ {vocab_size}).  Ignoring cached contextual embeddings."
        )
        return None
    if max_occurrences is not None and cache.get("_ctx_max_occurrences") != max_occurrences:
        warnings.warn(
            f"[CtxCache] max_ctx_occurrences changed "
            f"({cache.get('_ctx_max_occurrences')} → {max_occurrences}). "
            "Ignoring cached contextual embeddings."
        )
        return None
    print(f"  [CtxCache] Loaded from '{path}' — skipping contextual embedding pass.")
    return ctx


def save_ctx_embs_to_cache(
    path: Union[str, Path],
    ctx_embs_list: list,
    max_occurrences: Optional[int] = None,
) -> None:
    """
    Persist ``ctx_embs_list`` into the existing parse cache file.

    The function loads the current cache, injects the new keys, and writes
    it back atomically.  If the cache file does not exist yet (e.g. mode
    'none' without a prior full parse), a minimal new file is created.

    Parameters
    ----------
    path : str | Path
        Same file used for the parse cache.
    ctx_embs_list : list
        One entry per vocabulary word — ``torch.Tensor | None``.
    max_occurrences : int | None
        ``SCPTMConfig.max_ctx_occurrences`` used to build the list.
    """
    if os.path.exists(path):
        with open(path, "rb") as f:
            cache = pickle.load(f)
    else:
        cache = {}
    cache["ctx_embs_list"]    = ctx_embs_list
    cache["_ctx_vocab_size"]  = len(ctx_embs_list)
    if max_occurrences is not None:
        cache["_ctx_max_occurrences"] = max_occurrences
    with open(path, "wb") as f:
        pickle.dump(cache, f)
    print(f"  [CtxCache] Saved to '{path}'")


def _load_parse_cache(
    path: Union[str, Path],
    signature: dict,
):
    """
    Load a previously saved parse cache, validating it against ``signature``
    (see :func:`_cache_signature`).

    Returns the cache dict on an exact signature match, otherwise ``None``
    (triggering a fresh build) after warning about the concrete reason.
    """
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        cache = pickle.load(f)
    cached = cache.get("_signature")
    if cached is None:
        # Caches written before the config signature existed cannot be
        # validated (min_df / max_features / graph_mode / corpus content are
        # unknown), and accepting them is exactly how a stale vocabulary used
        # to slip through.  Rebuild once; the new cache carries a signature.
        warnings.warn(
            f"[EdgeCache] '{path}' predates the config-signature check and "
            "cannot be validated; rebuilding it once. Subsequent runs will "
            "reuse it.",
            stacklevel=2,
        )
        return None
    reasons = _signature_diff(cached, signature)
    if reasons:
        warnings.warn(
            f"[EdgeCache] Cache invalidated — {'; '.join(reasons)}. Rebuilding.",
            stacklevel=2,
        )
        return None
    print(f"  [EdgeCache] Loaded from '{path}' — skipping lemmatisation/parsing.")
    return cache


# ---------------------------------------------------------------------------
# Memory estimation
# ---------------------------------------------------------------------------

def estimate_graph_memory(
    n_docs: int,
    vocab_size: int,
    n_edges_doc_word: int,
    n_edges_word_word: int,
    emb_dim: int = 384,
    dtype_bytes: int = 4,
) -> dict:
    """Print and return a GPU memory estimate for the heterogeneous graph."""
    node_mem = (n_docs + vocab_size) * emb_dim * dtype_bytes
    edge_mem = (n_edges_doc_word + n_edges_word_word) * 2 * dtype_bytes
    activation_mem = node_mem * 3
    gradient_mem = node_mem * 2
    total_bytes = node_mem + edge_mem + activation_mem + gradient_mem
    total_gb = total_bytes / (1024 ** 3)
    report = {
        "node_features_MB":   round(node_mem / 1024**2, 1),
        "edge_index_MB":      round(edge_mem / 1024**2, 1),
        "activations_MB":     round(activation_mem / 1024**2, 1),
        "gradients_MB":       round(gradient_mem / 1024**2, 1),
        "total_estimated_GB": round(total_gb, 2),
    }
    print("\n[Memory estimate]")
    for k, v in report.items():
        print(f"  {k:25s}: {v}")
    if total_gb > 20:
        print("  CRITICAL WARNING: >20 GB estimated. Use NeighborLoader or GraphSAINT.")
    elif total_gb > 8:
        print("  WARNING: >8 GB estimated. Enable mixed precision + gradient checkpointing.")
    return report


# ---------------------------------------------------------------------------
# PMI-based graph sparsification
# ---------------------------------------------------------------------------

def _pmi_filter_edges(
    word_word_src: list,
    word_word_dst: list,
    doc_word_src: list,
    doc_word_dst: list,
    n_docs: int,
    top_k: int = 15,
) -> Tuple[list, list]:
    """
    Filter word-word edges by Positive PMI (PPMI > 0) and optionally
    keep only the top-k highest-PPMI neighbours per source word.

    Works on the existing syntactic edge list — does NOT recompute from
    scratch; PPMI is estimated from the doc-word occurrence data already
    collected during graph construction.

    Parameters
    ----------
    word_word_src, word_word_dst : lists of word indices (syntactic edges)
    doc_word_src, doc_word_dst   : lists for doc-word occurrence edges
    n_docs : int                 : total number of documents
    top_k  : int                 : max neighbours per word (0 = no limit)

    Returns
    -------
    filtered_src, filtered_dst : filtered edge lists
    """
    if not word_word_src:
        return word_word_src, word_word_dst

    # Build per-word document sets (needed for PPMI estimation)
    word_doc_sets: dict = defaultdict(set)
    for d, w in zip(doc_word_src, doc_word_dst):
        word_doc_sets[w].add(d)

    # Compute PPMI for each unique edge pair, then filter
    pair_ppmi: dict = {}
    filtered_src: list = []
    filtered_dst: list = []
    edge_ppmi_vals: list = []

    for src, dst in zip(word_word_src, word_word_dst):
        key = (min(src, dst), max(src, dst))
        if key not in pair_ppmi:
            docs_s = word_doc_sets.get(src, set())
            docs_d = word_doc_sets.get(dst, set())
            n_s = len(docs_s)
            n_d = len(docs_d)
            n_co = len(docs_s & docs_d)
            if n_co > 0 and n_s > 0 and n_d > 0:
                pmi = math.log(
                    (n_co / n_docs) / ((n_s / n_docs) * (n_d / n_docs))
                )
                pair_ppmi[key] = max(pmi, 0.0)
            else:
                pair_ppmi[key] = 0.0

        ppmi_val = pair_ppmi[key]
        if ppmi_val > 0.0:
            filtered_src.append(src)
            filtered_dst.append(dst)
            edge_ppmi_vals.append(ppmi_val)

    # Top-k neighbours per source node by PPMI score
    if top_k > 0 and filtered_src:
        triples = sorted(
            zip(filtered_src, filtered_dst, edge_ppmi_vals),
            key=lambda x: (x[0], -x[2]),
        )
        src_count: dict = defaultdict(int)
        tk_src: list = []
        tk_dst: list = []
        for s, d, _ in triples:
            if src_count[s] < top_k:
                tk_src.append(s)
                tk_dst.append(d)
                src_count[s] += 1
        filtered_src, filtered_dst = tk_src, tk_dst

    before = len(word_word_src)
    after  = len(filtered_src)
    pct    = (1 - after / max(before, 1)) * 100
    print(f"  PMI sparsification: {before:,} → {after:,} word-word edges ({pct:.0f}% reduction)")
    return filtered_src, filtered_dst


# ---------------------------------------------------------------------------
# Main graph builder
# ---------------------------------------------------------------------------

def build_hetero_graph(
    documents: List[str],
    sbert_model,
    nlp_model,
    stop_words,
    cfg: SCPTMConfig,
    edge_cache_path: Optional[Union[str, Path]] = None,
) -> Tuple[HeteroData, List[str], object, int, int, Dict[Tuple[str, str, str], int]]:
    """
    Build a heterogeneous PyG graph over documents and vocabulary words.

    Node types : "doc", "word"
    Edge types : ("doc","contains","word"),
                 ("word","rev_contains","doc"),
                 ("word","relates","word")

    Returns
    -------
    data        : HeteroData
    vocab_list  : list of vocabulary strings
    bow_sparse  : CountVectorizer sparse BoW matrix (n_docs x vocab_size)
    n_dw        : number of doc-word edges
    n_ww        : number of word-word edges
    dep_triples : {(head_lemma, dep_label, dependent_lemma): count} —
        the same syntactic relations behind the word-word edges, kept with
        their relation label (discarded from the edge tensor itself).
        Empty for graph_mode in {"none", "no_syntax"}. Used by
        keywords.extract_separated_topics for graph-based MWE extraction.
    """
    graph_mode = cfg.graph_mode
    if graph_mode not in GRAPH_MODES:
        raise ValueError(
            f"graph_mode must be one of {list(GRAPH_MODES)}, got '{graph_mode}'"
        )

    print(f"\n[Graph] Mode: '{graph_mode}' — {GRAPH_MODES[graph_mode]}")

    # Cache signature: the cached artifacts depend on the corpus content and on
    # several config fields (lemmatiser, graph_mode, min_df, max_features, lang);
    # all of them must match for a cache hit, otherwise a stale vocabulary could
    # be reused silently.
    cache_signature = _cache_signature(cfg, documents)

    # ---- Try loading pre-cached parse results ----
    _cache = None
    if edge_cache_path is not None:
        _cache = _load_parse_cache(edge_cache_path, cache_signature)

    if _cache is not None:
        # Cache hit: restore vocabulary and (conditionally) edge lists
        vocab_arr  = _cache["vocab_arr"]
        bow_sparse = _cache["bow_sparse"]
        vocab      = {word: idx for idx, word in enumerate(vocab_arr)}
        # Edge lists are valid only when the cache was built by a non-"none" run
        # (doc_word_src non-empty) OR when the current mode is "none" (edges unused).
        # If a "none"-mode run created the cache first, the edge lists are [] and
        # must be rebuilt for any mode that needs them.
        _cache_edges_valid = (
            graph_mode == "none"
            or len(_cache.get("doc_word_src", [])) > 0
        )
        if _cache_edges_valid:
            doc_word_src  = _cache["doc_word_src"]
            doc_word_dst  = _cache["doc_word_dst"]
            word_word_src = _cache["word_word_src"]
            word_word_dst = _cache["word_word_dst"]
            # dep_triples absent in caches written before this field existed —
            # falls back to {} (MWE extraction then uses the n-gram fallback).
            dep_triples   = _cache.get("dep_triples", {})
        print(f"  Vocabulary: {len(vocab)} unique lemmas (from cache).")
        _cache_was_used = True
    else:
        # ---- 1. Build vocabulary on lemmas ----
        print("1/5  Building lemma vocabulary...")
        lemma_docs = []
        parsed_docs = _iter_parsed(nlp_model, documents)
        for doc in tqdm(parsed_docs, total=len(documents), desc="Lemmatisation"):
            lemmas = [
                t.lemma_.lower()
                for t in doc
                if not t.is_stop and t.is_alpha and len(t.lemma_) > 2
            ]
            lemma_docs.append(" ".join(lemmas))

        # stop_words=None: the lemmatiser already filtered stop words during
        # parsing (t.is_stop guard above). Passing the raw list to
        # CountVectorizer triggers sklearn warnings for Italian because
        # lemmatised stop-word forms ("gl", "nient") don't match the original
        # list entries.
        vectorizer = CountVectorizer(
            stop_words=None,
            min_df=cfg.min_df,
            max_features=cfg.max_features,
            token_pattern=VOCAB_TOKEN_PATTERN,
        )
        bow_sparse = vectorizer.fit_transform(lemma_docs)
        vocab_arr  = vectorizer.get_feature_names_out()
        vocab      = {word: idx for idx, word in enumerate(vocab_arr)}
        print(f"  Vocabulary: {len(vocab)} unique lemmas.")
        _cache_was_used = False
        _cache_edges_valid = False

    # ---- 2. Document embeddings ----
    if _cache is not None and _cache.get("doc_embs") is not None:
        print("2/5  Document embeddings loaded from cache.")
        doc_embs = _cache["doc_embs"]
    else:
        print("2/5  Encoding documents (SBERT)...")
        doc_embs = sbert_model.encode(documents, show_progress_bar=True)
    data = HeteroData()
    data["doc"].x = torch.tensor(doc_embs, dtype=torch.float)

    # ---- None mode: no edges ----
    if graph_mode == "none":
        print("  MODE 'none': no edges generated (CTM-like baseline).")
        if _cache is not None and _cache.get("word_embs_static") is not None:
            print("  Word embeddings loaded from cache.")
            word_embs_none = _cache["word_embs_static"]
        else:
            word_embs_none = sbert_model.encode(vocab_arr.tolist(), show_progress_bar=True)
        data["word"].x = torch.tensor(word_embs_none, dtype=torch.float)
        data["doc", "contains", "word"].edge_index     = torch.empty((2, 0), dtype=torch.long)
        data["word", "rev_contains", "doc"].edge_index = torch.empty((2, 0), dtype=torch.long)
        data["word", "relates", "word"].edge_index     = torch.empty((2, 0), dtype=torch.long)
        # Cache vocab/bow/embeddings so future runs skip the NLP parse + SBERT entirely.
        # IMPORTANT: preserve any existing edge lists in the cache — a "none"-mode
        # run must not overwrite real edges built by a prior "filtered"/"full_dep" run.
        if edge_cache_path is not None:
            _needs_save = (
                _cache is None
                or _cache.get("doc_embs") is None
                or _cache.get("word_embs_static") is None
            )
            if _needs_save:
                _dw_src = _cache.get("doc_word_src", []) if _cache else []
                _dw_dst = _cache.get("doc_word_dst", []) if _cache else []
                _ww_src = _cache.get("word_word_src", []) if _cache else []
                _ww_dst = _cache.get("word_word_dst", []) if _cache else []
                _dep_triples = _cache.get("dep_triples", {}) if _cache else {}
                _save_parse_cache(
                    edge_cache_path, vocab_arr, bow_sparse,
                    _dw_src, _dw_dst, _ww_src, _ww_dst, len(documents),
                    doc_embs=doc_embs,
                    word_embs_static=word_embs_none,
                    dep_triples=_dep_triples,
                    signature=cache_signature,
                )
        return (
            data, vocab_arr.tolist(), bow_sparse, 0, 0,
            _cache.get("dep_triples", {}) if _cache else {},
        )

    # ---- Active dependency types ----
    if graph_mode == "filtered":
        active_deps = INFORMATIVE_DEP_TYPES
    elif graph_mode == "full_dep":
        active_deps = ALL_CONTENT_DEP_TYPES
    else:  # no_syntax
        active_deps = frozenset()

    if not _cache_was_used or not _cache_edges_valid:
        # ---- 3. Syntactic parsing ----
        # Runs when: (a) no cache at all, or (b) cache exists but was built with
        # graph_mode="none" (empty edge lists) and current mode needs real edges.
        print("3/5  Syntactic parsing...")
        doc_word_src, doc_word_dst = [], []
        word_word_src, word_word_dst = [], []
        dep_counts = defaultdict(int)
        # (head_lemma, dep_label, dependent_lemma) -> occurrence count.
        # Same edges as word_word_src/dst, but keeping the relation label
        # that would otherwise be discarded — used for graph-based MWE
        # extraction (see keywords.extract_separated_topics) instead of
        # surface n-grams.
        dep_triple_counts: dict = defaultdict(int)

        parsing_docs = _iter_parsed(nlp_model, documents)
        for d_idx, doc in enumerate(
            tqdm(parsing_docs, total=len(documents), desc="Dependency parsing")
        ):
            for token in doc:
                if token.is_stop or not token.is_alpha or len(token.lemma_) <= 2:
                    continue
                lemma = token.lemma_.lower()
                if lemma not in vocab:
                    continue
                w_idx = vocab[lemma]
                doc_word_src.append(d_idx)
                doc_word_dst.append(w_idx)
                if active_deps:
                    head_lemma = token.head.lemma_.lower()
                    if (
                        token.dep_ in active_deps
                        and not token.head.is_stop
                        and token.head.is_alpha
                        and head_lemma != lemma
                        and head_lemma in vocab
                    ):
                        word_word_src.append(w_idx)
                        word_word_dst.append(vocab[head_lemma])
                        dep_counts[token.dep_] += 1
                        dep_triple_counts[(head_lemma, token.dep_, lemma)] += 1

        if dep_counts:
            print("  Syntactic edge distribution:")
            for dep, cnt in sorted(dep_counts.items(), key=lambda x: -x[1]):
                print(f"    {dep:12s}: {cnt:6d}")

        dep_triples = dict(dep_triple_counts)

    else:
        print("3/5  Syntactic parsing skipped (cache hit).")

    # ---- 4. Word embeddings (static) ----
    if _cache is not None and _cache.get("word_embs_static") is not None:
        print("4/5  Word embeddings loaded from cache.")
        word_embs_static = _cache["word_embs_static"]
    else:
        print("4/5  Encoding vocabulary (SBERT static)...")
        word_embs_static = sbert_model.encode(vocab_arr.tolist(), show_progress_bar=True)
    data["word"].x = torch.tensor(word_embs_static, dtype=torch.float)

    # ---- Save / update cache with edge lists + embeddings ----
    if edge_cache_path is not None:
        _needs_save = (
            not _cache_was_used        # no cache at all → save everything
            or not _cache_edges_valid  # cache had empty edges (from a prior "none" run)
            or _cache.get("doc_embs") is None
            or _cache.get("word_embs_static") is None
        )
        if _needs_save:
            _save_parse_cache(
                edge_cache_path, vocab_arr, bow_sparse,
                doc_word_src, doc_word_dst,
                word_word_src, word_word_dst,
                len(documents),
                doc_embs=doc_embs,
                word_embs_static=word_embs_static,
                dep_triples=dep_triples,
                signature=cache_signature,
            )

    # ---- 4b. PMI sparsification of word-word edges ----
    # Applied AFTER cache load/save so raw edges are always cached, and
    # the filter can be re-run with different pmi_top_k_neighbors cheaply.
    if cfg.pmi_sparse_graph and word_word_src and graph_mode != "no_syntax":
        word_word_src, word_word_dst = _pmi_filter_edges(
            word_word_src, word_word_dst,
            doc_word_src, doc_word_dst,
            len(documents), top_k=cfg.pmi_top_k_neighbors,
        )

    # ---- 5. Build edge tensors ----
    print("5/5  Building edge tensors...")
    dw_idx = torch.tensor([doc_word_src, doc_word_dst], dtype=torch.long)
    data["doc", "contains", "word"].edge_index = dw_idx
    # rev_contains is the reverse of contains (word -> doc), letting the
    # encoder's gated word_to_doc_gate pull syntactic/graph context back
    # into document representations.
    data["word", "rev_contains", "doc"].edge_index = torch.tensor(
        [doc_word_dst, doc_word_src], dtype=torch.long
    )

    if word_word_src and graph_mode != "no_syntax":
        data["word", "relates", "word"].edge_index = torch.tensor(
            [word_word_src, word_word_dst], dtype=torch.long
        )
    else:
        data["word", "relates", "word"].edge_index = torch.empty((2, 0), dtype=torch.long)
        if graph_mode == "no_syntax":
            print("  MODE 'no_syntax': word-word edges omitted.")

    n_dw = len(doc_word_src)
    n_ww = len(word_word_src)
    print(f"  Doc-word edges : {n_dw:,}")
    print(f"  Word-word edges: {n_ww:,}")
    return data, vocab_arr.tolist(), bow_sparse, n_dw, n_ww, dep_triples


# ---------------------------------------------------------------------------
# Control condition for the syntax ablation
# ---------------------------------------------------------------------------
#
# The claim "syntactic edges contribute" needs a control with the same graph
# statistics but no linguistic content.  Apply either helper to the syntactic
# edge lists *before* graph construction to obtain that control:
#
#     ctrl_src, ctrl_dst = rewire_word_word_edges(
#         word_word_src, word_word_dst, n_vocab, seed=0, mode="swap"
#     )
#     # then build the graph from ctrl_src / ctrl_dst instead
#
# Two modes, two different null hypotheses:
#
# * "swap"    — 2-edge swap randomisation: every word keeps exactly its degree,
#               the edge count is unchanged, only the topology is destroyed.
#               Asks "is it the *content* of the edges, or just more
#               regularisation?"
# * "permute" — node relabelling: the same graph attached to the wrong words.
#               Asks "is it the *pairing with words*, or any smoothing at all?"


def rewire_word_word_edges(
    word_word_src: list,
    word_word_dst: list,
    n_vocab: int,
    seed: int = 0,
    mode: str = "swap",
    n_swaps: Optional[int] = None,
) -> Tuple[list, list]:
    """
    Return a control version of the syntactic edge lists.

    Parameters
    ----------
    mode : "swap" | "permute"
        "swap" preserves each word's degree exactly (2-edge swap
        randomisation); "permute" relabels the nodes, preserving every graph
        statistic but destroying the word correspondence.
    n_swaps : int, optional
        Number of swap attempts; defaults to ``10 * n_edges``.

    Returns
    -------
    (src, dst) : new edge lists of the same length as the input (``permute``)
        or with the same degree sequence (``swap``).
    """
    if mode not in ("swap", "permute"):
        raise ValueError(f"mode must be 'swap' or 'permute', got '{mode}'")
    if not word_word_src:
        return list(word_word_src), list(word_word_dst)

    rng = np.random.default_rng(seed)
    src = [int(x) for x in word_word_src]
    dst = [int(x) for x in word_word_dst]

    if mode == "permute":
        perm = rng.permutation(int(n_vocab))
        return [int(perm[s]) for s in src], [int(perm[d]) for d in dst]

    # Self-loops are kept untouched (dropping them would change the degree
    # sequence, which is exactly what this control must preserve).
    edges = [(s, d) for s, d in zip(src, dst)]
    present = {(min(a, b), max(a, b)) for a, b in edges if a != b}
    swaps = 10 * len(edges) if n_swaps is None else int(n_swaps)

    for _ in range(swaps):
        i, j = rng.integers(0, len(edges), size=2)
        if i == j:
            continue
        u, v = edges[i]
        x, y = edges[j]
        if u == v or x == y:            # leave self-loops alone
            continue
        if len({u, v, x, y}) < 4:
            continue
        if u == y or x == v:
            continue
        key_i = (min(u, y), max(u, y))
        key_j = (min(x, v), max(x, v))
        if key_i == key_j or key_i in present or key_j in present:
            continue
        present.discard((min(u, v), max(u, v)))
        present.discard((min(x, y), max(x, y)))
        present.add(key_i)
        present.add(key_j)
        edges[i] = (u, y)
        edges[j] = (x, v)

    return [int(a) for a, _ in edges], [int(b) for _, b in edges]
