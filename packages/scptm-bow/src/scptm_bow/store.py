"""
scptm_bow/store.py
-------------------
Corpus hashing and the parse cache.

The cache is validated against a *signature*: a hash of the corpus text plus
every configuration field the cached artefacts depend on.  If anything that
matters changes, the cache is rebuilt and the reason is reported — a stale
vocabulary can never be reused silently.
"""

import hashlib
import os
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

FORMAT_VERSION = 1


def corpus_hash(documents: List[str]) -> str:
    """
    Short, order-sensitive hash of the corpus text.

    Updated incrementally so a large corpus never needs a second full copy in
    memory.  Catches edits that leave the document count unchanged.
    """
    h = hashlib.sha1()
    for doc in documents:
        h.update(str(doc).encode("utf-8", errors="replace"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def cache_signature(cfg, documents: List[str]) -> Dict[str, object]:
    """Everything the cached parse artefacts depend on."""
    return {
        "content_hash": corpus_hash(documents),
        "n_docs": len(documents),
        "lang": cfg.lang,
        "parser_backend": cfg.parser_backend,
        "udpipe_model_path": cfg.udpipe_model_path,
        "min_df": cfg.min_df,
        "max_features": cfg.max_features,
        "apply_chunking": cfg.apply_chunking,
        "max_chunk_chars": cfg.max_chunk_chars,
        "dep_types": tuple(sorted(cfg.dep_types)),
    }


def signature_diff(cached: Dict[str, object], current: Dict[str, object]) -> List[str]:
    """Human-readable reasons why a cached signature no longer matches."""
    reasons: List[str] = []
    for key, value in current.items():
        old = cached.get(key)
        if old == value:
            continue
        if key == "content_hash":
            reasons.append("corpus content changed (same document count)")
        else:
            reasons.append(f"{key}: {old!r} -> {value!r}")
    return reasons


def save_cache(
    path: Union[str, Path],
    payload: Dict[str, object],
    signature: Dict[str, object],
) -> None:
    """Persist parse artefacts (vocabulary, BoW, dependency pairs and triples)."""
    payload = dict(payload)
    payload["_format_version"] = FORMAT_VERSION
    payload["_signature"] = signature
    with open(path, "wb") as fh:
        pickle.dump(payload, fh)
    print(f"  [cache] saved -> {path}")


def load_cache(path: Union[str, Path], signature: Dict[str, object]) -> Optional[Dict]:
    """
    Load the parse cache, or return None (with a warning naming the reason).

    A cache written without a signature cannot be validated, so it is rejected
    once and rebuilt.
    """
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        payload = pickle.load(fh)

    cached = payload.get("_signature")
    if cached is None:
        warnings.warn(
            f"[cache] '{path}' has no signature and cannot be validated; "
            "rebuilding it once.",
            stacklevel=2,
        )
        return None

    version = payload.get("_format_version")
    if version != FORMAT_VERSION:
        warnings.warn(
            f"[cache] format version {version} != {FORMAT_VERSION}; rebuilding.",
            stacklevel=2,
        )
        return None

    reasons = signature_diff(cached, signature)
    if reasons:
        warnings.warn(
            f"[cache] invalidated — {'; '.join(reasons)}. Rebuilding.",
            stacklevel=2,
        )
        return None

    print(f"  [cache] hit -> {path}")
    return payload
