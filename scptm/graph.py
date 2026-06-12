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
      * Later runs — loads from cache, skips spaCy AND SBERT encode entirely.

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

import math
import os
import pickle
import warnings
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple, Union

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


# ---------------------------------------------------------------------------
# Corpus helpers
# ---------------------------------------------------------------------------

def prepare_corpus(
    source,
    source_type: str = "folder",
    text_col: Optional[str] = None,
    apply_chunking: bool = True,
    max_chunk_chars: int = 800,
) -> List[str]:
    """
    Load and optionally chunk a text corpus.

    Parameters
    ----------
    source : str | pd.DataFrame
        Path to a folder of .txt files, or a DataFrame.
    source_type : str
        "folder" or "dataframe".
    text_col : str | None
        Column name when source_type == "dataframe".
    apply_chunking : bool
        Split long documents into shorter segments.
    max_chunk_chars : int
        Maximum characters per chunk.

    Returns
    -------
    List[str]
        Processed document strings.
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
        docs = [" ".join(t.split()) for t in raw if len(t.strip()) > 10]
        return docs

    docs: List[str] = []
    for text in raw:
        clean = " ".join(text.split())
        sentences = clean.split(". ")
        chunk = ""
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if len(chunk) + len(sent) > max_chunk_chars and chunk:
                docs.append(chunk + ".")
                chunk = sent
            else:
                chunk = chunk + ". " + sent if chunk else sent
        if len(chunk) > 50:
            docs.append(chunk + ".")

    print(f"Final corpus: {len(docs)} segments.")
    return docs


# ---------------------------------------------------------------------------
# Contextual embeddings per word
# ---------------------------------------------------------------------------

def collect_contextual_embeddings(
    documents: List[str],
    nlp_model,
    sbert_model,
    vocab: dict,
    max_occurrences_per_word: int = 50,
) -> dict:
    """
    For each vocabulary word, collect up to `max_occurrences_per_word` SBERT
    document embeddings from contexts where the word appears.

    Returns
    -------
    dict
        word -> torch.Tensor of shape (N, emb_dim)
    """
    print("Collecting contextual embeddings (REV-A)...")
    ctx_embs = defaultdict(list)

    for text in tqdm(documents, desc="Contextual embeddings"):
        doc_vec = sbert_model.encode(text)
        spacy_doc = nlp_model(text)
        seen = set()
        for token in spacy_doc:
            if token.is_stop or not token.is_alpha or len(token.lemma_) <= 2:
                continue
            lemma = token.lemma_.lower()
            if lemma not in vocab or lemma in seen:
                continue
            seen.add(lemma)
            ctx_embs[lemma].append(doc_vec)

    # Subsample to max_occurrences_per_word
    rng = np.random.default_rng(42)
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
# Parse cache (edge list + vocabulary) — avoid re-running spaCy on reload
# ---------------------------------------------------------------------------

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
) -> None:
    """Persist the NLP-heavy outputs so they can be reused across runs.

    doc_embs and word_embs_static are optional; when provided they are stored
    so that subsequent calls to build_hetero_graph can skip SBERT entirely.
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
    if doc_embs is not None:
        cache["doc_embs"] = doc_embs
    if word_embs_static is not None:
        cache["word_embs_static"] = word_embs_static
    with open(path, "wb") as f:
        pickle.dump(cache, f)
    print(f"  [EdgeCache] Saved to '{path}'")


def load_ctx_embs_from_cache(
    path: Union[str, Path],
    vocab_size: int,
) -> Optional[list]:
    """
    Try to load a previously saved ``ctx_embs_list`` from the parse cache.

    Returns the list (one entry per vocabulary word) if the cache file exists
    **and** was built with the same vocabulary size, otherwise returns ``None``
    (triggering a fresh ``collect_contextual_embeddings`` call).

    Parameters
    ----------
    path : str | Path
        Same file used for the parse cache (ctx embeddings are piggy-backed
        onto the same pickle).
    vocab_size : int
        Expected number of entries in the list.
    """
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        cache = pickle.load(f)
    ctx = cache.get("ctx_embs_list")
    if ctx is None:
        return None
    if cache.get("_ctx_vocab_size") != vocab_size:
        warnings.warn(
            f"[CtxCache] Vocabulary size changed ({cache.get('_ctx_vocab_size')} "
            f"→ {vocab_size}).  Ignoring cached contextual embeddings."
        )
        return None
    print(f"  [CtxCache] Loaded from '{path}' — skipping contextual embedding pass.")
    return ctx


def save_ctx_embs_to_cache(
    path: Union[str, Path],
    ctx_embs_list: list,
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
    """
    if os.path.exists(path):
        with open(path, "rb") as f:
            cache = pickle.load(f)
    else:
        cache = {}
    cache["ctx_embs_list"]    = ctx_embs_list
    cache["_ctx_vocab_size"]  = len(ctx_embs_list)
    with open(path, "wb") as f:
        pickle.dump(cache, f)
    print(f"  [CtxCache] Saved to '{path}'")


def _load_parse_cache(path: Union[str, Path], n_docs: int):
    """
    Load a previously saved parse cache.

    Returns the cache dict if the file exists and the corpus size matches,
    otherwise returns None (triggering a fresh build).
    """
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        cache = pickle.load(f)
    if cache.get("_n_docs") != n_docs:
        warnings.warn(
            f"[EdgeCache] Corpus size changed ({cache['_n_docs']} → {n_docs}). "
            "Ignoring stale cache and rebuilding."
        )
        return None
    print(f"  [EdgeCache] Loaded from '{path}' — skipping spaCy parsing.")
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
) -> Tuple[HeteroData, List[str], object, int, int]:
    """
    Build a heterogeneous PyG graph over documents and vocabulary words.

    Node types : "doc", "word"
    Edge types : ("doc","contains","word"),
                 ("word","rev_contains","doc"),
                 ("word","relates","word")

    Returns
    -------
    data       : HeteroData
    vocab_list : list of vocabulary strings
    bow_sparse : CountVectorizer sparse BoW matrix (n_docs x vocab_size)
    n_dw       : number of doc-word edges
    n_ww       : number of word-word edges
    """
    graph_mode = cfg.graph_mode
    assert graph_mode in GRAPH_MODES

    print(f"\n[Graph] Mode: '{graph_mode}' — {GRAPH_MODES[graph_mode]}")

    # ---- Try loading pre-cached parse results ----
    _cache = None
    if edge_cache_path is not None:
        _cache = _load_parse_cache(edge_cache_path, len(documents))

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
        print(f"  Vocabulary: {len(vocab)} unique lemmas (from cache).")
        _cache_was_used = True
    else:
        # ---- 1. Build vocabulary on lemmas ----
        print("1/5  Building lemma vocabulary...")
        lemma_docs = []
        for text in tqdm(documents, desc="Lemmatisation"):
            doc = nlp_model(text)
            lemmas = [
                t.lemma_.lower()
                for t in doc
                if not t.is_stop and t.is_alpha and len(t.lemma_) > 2
            ]
            lemma_docs.append(" ".join(lemmas))

        # stop_words=None: spaCy already filtered stop words during lemmatisation
        # (t.is_stop guard above). Passing the raw spaCy list to CountVectorizer
        # triggers sklearn warnings for Italian because lemmatised stop-word forms
        # ("gl", "nient") don't match the original list entries.
        vectorizer = CountVectorizer(
            stop_words=None,
            min_df=cfg.min_df,
            max_features=cfg.max_features,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
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
        # Cache vocab/bow/embeddings so future runs skip spaCy + SBERT entirely.
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
                _save_parse_cache(
                    edge_cache_path, vocab_arr, bow_sparse,
                    _dw_src, _dw_dst, _ww_src, _ww_dst, len(documents),
                    doc_embs=doc_embs,
                    word_embs_static=word_embs_none,
                )
        return data, vocab_arr.tolist(), bow_sparse, 0, 0

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

        for d_idx, text in enumerate(tqdm(documents, desc="Dependency parsing")):
            doc = nlp_model(text)
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

        if dep_counts:
            print("  Syntactic edge distribution:")
            for dep, cnt in sorted(dep_counts.items(), key=lambda x: -x[1]):
                print(f"    {dep:12s}: {cnt:6d}")

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
    # rev_contains removed from the encoder architecture; keep an empty tensor
    # for backward compatibility with any cached HeteroData that expects the key.
    data["word", "rev_contains", "doc"].edge_index = torch.empty((2, 0), dtype=torch.long)

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
    return data, vocab_arr.tolist(), bow_sparse, n_dw, n_ww
