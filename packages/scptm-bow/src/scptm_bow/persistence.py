"""
scptm_bow/persistence.py
-------------------------
Versioned persistence for a fitted :class:`~scptm_bow.model.BowSCPTM`.

Why not just ``pickle.dump(model)``?
------------------------------------
Pickling the object works, but it couples the file to the *class layout* of
whatever version wrote it and gives unhelpful errors when the file is not a
model at all.  This module writes a plain, versioned state dict instead:

* a header with the format version, the kind of object and the package version;
* only the fields needed to use the model (config, vocabulary, BoW, dependency
  graph, word vectors, graph Laplacians, NMF factors);
* an atomic write (temp file + ``os.replace``), so a crash never leaves a
  half-written model behind.

The parser is **not** serialised.  Loading therefore needs neither spaCy nor
UDPipe, only the parser model at the first ``transform()`` call.
"""

from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

import os
import pickle

from .config import BowConfig
from .factorize import NMFResult
from .graphs import Graph
from .parse import ParsedCorpus

# Bump when the state layout changes in a way older readers cannot handle.
SAVE_FORMAT_VERSION = 1

_KIND = "scptm_bow.BowSCPTM"
_REQUIRED = (
    "config",
    "vocab",
    "bow",
    "dep_src",
    "dep_dst",
    "dep_weight",
    "dep_counts",
    "n_docs",
    "result",
)


def _package_version() -> str:
    """Current package version (imported lazily to avoid a circular import)."""
    from . import __version__

    return __version__


# ---------------------------------------------------------------------------
# Small piece-wise converters
# ---------------------------------------------------------------------------

def _graph_to_state(graph: Optional[Graph]) -> Optional[Dict[str, Any]]:
    if graph is None:
        return None
    return {"adj": graph.adj, "degree": graph.degree}


def _graph_from_state(state: Optional[Dict[str, Any]]) -> Optional[Graph]:
    if state is None:
        return None
    return Graph(adj=state["adj"], degree=state["degree"])


def _result_to_state(result: NMFResult) -> Dict[str, Any]:
    return {
        "W": result.W,
        "H": result.H,
        "n_iter": int(result.n_iter),
        "objective": float(result.objective),
        "divergence": result.divergence,
        "history": list(result.history),
    }


def _result_from_state(state: Dict[str, Any]) -> NMFResult:
    # n_iter/objective are metadata: tolerate their absence rather than refusing
    # a file whose factors are intact.
    return NMFResult(
        W=state["W"],
        H=state["H"],
        n_iter=int(state.get("n_iter", 0)),
        objective=float(state.get("objective", float("nan"))),
        divergence=state.get("divergence", "kl"),
        history=list(state.get("history", [])),
    )


def _config_to_state(cfg: BowConfig) -> Dict[str, Any]:
    return asdict(cfg)


def _config_from_state(state: Dict[str, Any]) -> BowConfig:
    """
    Rebuild the config, ignoring fields this version does not know about.

    A model saved by a newer scptm-bow must stay loadable: unknown fields are
    dropped rather than raising, because they cannot affect the fitted factors.
    """
    known = {f.name for f in fields(BowConfig)}
    kwargs = {k: v for k, v in dict(state).items() if k in known}
    if "dep_types" in kwargs:
        kwargs["dep_types"] = frozenset(kwargs["dep_types"])
    return BowConfig(**kwargs)


# ---------------------------------------------------------------------------
# Model <-> state
# ---------------------------------------------------------------------------

def model_to_state(model, include_corpus: bool = False) -> Dict[str, Any]:
    """Everything needed to reconstruct the fitted model, as plain objects."""
    model._check_fitted()
    parsed = model.parsed
    state: Dict[str, Any] = {
        "format": SAVE_FORMAT_VERSION,
        "kind": _KIND,
        "package_version": _package_version(),
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": _config_to_state(model.config),
        "vocab": parsed.vocab,
        "bow": parsed.bow,
        "dep_src": parsed.dep_src,
        "dep_dst": parsed.dep_dst,
        "dep_weight": parsed.dep_weight,
        "dep_counts": dict(parsed.dep_counts),
        "n_docs": int(parsed.n_docs),
        "vectors": model.vectors,
        "syntax_graph": _graph_to_state(model.syntax_graph),
        "embed_graph": _graph_to_state(model.embed_graph),
        "result": _result_to_state(model.result),
        "npmi": model._npmi,
        "syntax_npmi": model._syntax_npmi,
    }
    if include_corpus:
        # The lemmatised documents are only needed to rebuild the vectors or to
        # inspect the training corpus; nothing in the fitted model uses them.
        state["lemma_docs"] = list(parsed.lemma_docs)
    return state


def model_from_state(
    state: Dict[str, Any],
    path: Optional[Union[str, Path]] = None,
) -> "BowSCPTM":
    """Inverse of :func:`model_to_state`.  Does not touch the parser."""
    _validate_state(state, path)

    from .model import BowSCPTM  # lazy: model imports this module's functions

    parsed = ParsedCorpus(
        vocab=state["vocab"],
        bow=state["bow"],
        dep_src=state["dep_src"],
        dep_dst=state["dep_dst"],
        dep_weight=state["dep_weight"],
        dep_counts=dict(state["dep_counts"]),
        lemma_docs=list(state.get("lemma_docs") or []),
        n_docs=int(state["n_docs"]),
    )

    model = BowSCPTM(config=_config_from_state(state["config"]))
    model.parsed = parsed
    model.vectors = state.get("vectors")
    model.syntax_graph = _graph_from_state(state.get("syntax_graph"))
    model.embed_graph = _graph_from_state(state.get("embed_graph"))
    model.result = _result_from_state(state["result"])
    model._npmi = state.get("npmi")
    model._syntax_npmi = state.get("syntax_npmi")
    model._is_fitted = True
    return model


def _validate_state(state: Any, path: Optional[Union[str, Path]] = None) -> None:
    """Turn corrupt/foreign/too-new files into actionable errors."""
    where = f" in {path}" if path is not None else ""
    if not isinstance(state, dict):
        raise ValueError(f"not a scptm_bow model{where}: expected a dict, got {type(state).__name__}")

    kind = state.get("kind")
    if kind != _KIND:
        raise ValueError(
            f"not a scptm_bow model{where}: header says kind={kind!r}, "
            f"expected {_KIND!r}"
        )

    fmt = state.get("format")
    if not isinstance(fmt, int):
        raise ValueError(f"scptm_bow model{where} has no usable format version")
    if fmt > SAVE_FORMAT_VERSION:
        raise ValueError(
            f"scptm_bow model{where} was written with format {fmt}, but this "
            f"version reads up to {SAVE_FORMAT_VERSION}: upgrade scptm-bow "
            f"(saved by {state.get('package_version', 'unknown version')})."
        )

    missing = [key for key in _REQUIRED if key not in state]
    if missing:
        raise ValueError(
            f"scptm_bow model{where} is incomplete: missing {', '.join(missing)}"
        )

    vocab = state["vocab"]
    bow = state["bow"]
    result = state["result"]
    if getattr(bow, "shape", (None, None))[1] != len(vocab):
        raise ValueError(
            f"scptm_bow model{where} is inconsistent: BoW has {bow.shape[1]} "
            f"columns but the vocabulary has {len(vocab)} lemmas"
        )
    W = result.get("W") if isinstance(result, dict) else None
    H = result.get("H") if isinstance(result, dict) else None
    if W is None or H is None:
        raise ValueError(
            f"scptm_bow model{where} is incomplete: the NMF result must carry W and H"
        )
    if H.shape != (W.shape[1], len(vocab)):
        raise ValueError(
            f"scptm_bow model{where} is inconsistent: H has shape "
            f"{H.shape}, expected ({W.shape[1]}, {len(vocab)})"
        )
    if W.shape[0] != bow.shape[0]:
        raise ValueError(
            f"scptm_bow model{where} is inconsistent: theta has "
            f"{W.shape[0]} rows but the BoW has {bow.shape[0]}"
        )


# ---------------------------------------------------------------------------
# Disk I/O
# ---------------------------------------------------------------------------

def save_model(
    model,
    path: Union[str, Path],
    include_corpus: bool = False,
) -> Path:
    """
    Write the fitted model to ``path`` and return it.

    The write is atomic: the state goes to a temporary file in the same
    directory and is renamed over ``path`` only once complete.
    """
    path = Path(path)
    state = model_to_state(model, include_corpus=include_corpus)

    tmp = path.with_name(f".{path.name}.tmp")
    try:
        with open(tmp, "wb") as fh:
            pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        # Leave nothing behind if pickling or the rename failed.
        if tmp.exists():
            tmp.unlink()
    return path


def load_model(path: Union[str, Path]) -> "BowSCPTM":
    """Read a model written by :func:`save_model`; no parser is needed."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such model file: {path}")

    with open(path, "rb") as fh:
        try:
            state = pickle.load(fh)
        except Exception as exc:  # noqa: BLE001 - any unpickling failure is a bad file
            raise ValueError(
                f"{path} is not a scptm_bow model file (unpickling failed: {exc})"
            ) from exc

    return model_from_state(state, path=path)
