"""
scptm/nlp.py
------------
NLP pipeline initialisation: SBERT + a lemmatisation / dependency-parsing
backend.  English and Italian are supported.

Two backends can be selected through ``SCPTMConfig.lemmatizer``:

  * ``"spacy"`` (default) — spaCy statistical pipelines
    (``en_core_web_sm`` / ``it_core_news_sm``).
  * ``"udpipe"`` — UDPipe pretrained Universal Dependencies models
    (Straka & Straková), accessed through the optional ``ufal.udpipe``
    bindings.  Models are downloaded once and cached on disk; see
    ``SCPTMConfig.udpipe_model_path`` and ``SCPTMConfig.udpipe_cache_dir``.

Both backends expose the same minimal token interface consumed by
``scptm.graph`` (``.lemma_``, ``.is_stop``, ``.is_alpha``, ``.dep_``,
``.head``, plus ``.head.is_stop`` / ``.head.is_alpha`` / ``.head.lemma_``),
so the graph builder is completely backend-agnostic.
"""

import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from sentence_transformers import SentenceTransformer

from .config import resolve_device


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

# (lang, lemmatizer, udpipe_model_path, udpipe_cache_dir, device) -> (sbert, nlp, stop)
# Loading SBERT + a spaCy/UDPipe model costs ~4 s; a K sweep would otherwise pay
# that for every K.  See setup_nlp_pipeline / clear_pipeline_cache.
_PIPELINE_CACHE: Dict[tuple, tuple] = {}

LEMMATIZERS = ("spacy", "udpipe")

# SBERT / spaCy model names per language.
_SBERT_MODELS = {
    "eng": "all-MiniLM-L6-v2",
    "ita": "paraphrase-multilingual-MiniLM-L12-v2",
}
_SPACY_MODELS = {
    "eng": "en_core_web_sm",
    "ita": "it_core_news_sm",
}

# ---- UDPipe pretrained models (UD 2.5, 2019-12-06) ------------------------
# Distributed by LINDAT/CLARIN (handle 11234/1-3131).  The legacy
# /xmlui/bitstream/handle/... links were retired when LINDAT migrated to
# DSpace 7, so the canonical DSpace 7 REST "content" endpoints are used and
# a live lookup against the repository API is kept as a fallback if these
# UUIDs ever change.
UDPIPE_ITEM_HANDLE = "11234/1-3131"
UDPIPE_MODEL_FILENAMES = {
    "eng": "english-ewt-ud-2.5-191206.udpipe",
    "ita": "italian-isdt-ud-2.5-191206.udpipe",
}
UDPIPE_MODEL_URLS = {
    "eng": (
        "https://lindat.mff.cuni.cz/repository/server/api/core/bitstreams/"
        "8d51e3d3-1f87-47fe-8975-c197ddf7907b/content"
    ),
    "ita": (
        "https://lindat.mff.cuni.cz/repository/server/api/core/bitstreams/"
        "ee10c62a-01c5-42cb-a8b6-b7751c5da12c/content"
    ),
}
_UDPIPE_API_BASE = "https://lindat.mff.cuni.cz/repository/server/api"
# Environment overrides, useful on clusters with a shared model directory.
_ENV_MODEL_PATH = {"eng": "SCPTM_UDPIPE_MODEL_ENG", "ita": "SCPTM_UDPIPE_MODEL_ITA"}
_ENV_CACHE_DIR = "SCPTM_UDPIPE_MODEL_DIR"


def available_lemmatizers() -> Tuple[str, ...]:
    """Return the supported lemmatisation backends."""
    return LEMMATIZERS


def _load_sentence_transformer(name: str, device: str) -> SentenceTransformer:
    """
    Load SBERT from the local HuggingFace cache, without touching the network.

    ``SentenceTransformer(name)`` performs a Hub round-trip on **every**
    construction — even when the model is already cached — which costs several
    seconds and fails outright when offline (on a plane, on an air-gapped
    cluster).  ``local_files_only=True`` reads the cache directly: measured
    5.4 s -> 0.06 s for a cached model.  The network is used only when the model
    is genuinely missing from the cache.

    Set ``SCPTM_HF_OFFLINE=1`` to never fall back to the network: a missing
    model then raises instead of downloading.
    """
    try:
        return SentenceTransformer(name, device=device, local_files_only=True)
    except Exception:
        if os.environ.get("SCPTM_HF_OFFLINE"):
            raise
        return SentenceTransformer(name, device=device)


def _normalise_lang(lang: str) -> str:
    """Lower-case the language code, rejecting unsupported ones.

    An unknown code used to fall back to English silently, so a typo such as
    ``lang="deu"`` quietly ran the whole pipeline in English.
    """
    normalised = (lang or "eng").lower()
    if normalised not in _SPACY_MODELS:
        raise ValueError(
            f"Unsupported lang '{lang}'. Supported languages: "
            f"{sorted(_SPACY_MODELS)}"
        )
    return normalised


# ---------------------------------------------------------------------------
# Stop words
# ---------------------------------------------------------------------------

# Small built-in fallbacks, used only when spaCy's language data cannot be
# imported (spaCy is a hard dependency, so this should not normally happen).
_FALLBACK_STOP_WORDS = {
    "eng": frozenset({
        "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for",
        "from", "had", "has", "have", "he", "her", "his", "i", "in", "is", "it",
        "its", "of", "on", "or", "that", "the", "their", "they", "this", "to",
        "was", "were", "which", "will", "with", "you", "your",
    }),
    "ita": frozenset({
        "a", "ad", "al", "alla", "che", "chi", "ci", "con", "da", "dal", "dei",
        "del", "della", "di", "e", "ed", "gli", "ha", "ho", "i", "il", "in",
        "io", "la", "le", "lo", "ma", "mi", "ne", "nei", "nel", "nella", "no",
        "non", "o", "per", "più", "si", "sono", "su", "tra", "un", "una", "uno",
    }),
}


def get_stop_words(lang: str) -> Set[str]:
    """
    Return the language stop-word set used to filter tokens.

    The sets are pulled from spaCy's language data (``spacy.lang.*.stop_words``),
    which is identical to the per-model ``nlp.Defaults.stop_words`` but does not
    require loading a full spaCy pipeline.  This is what gives the UDPipe
    backend the same function-word filtering as the spaCy backend.
    """
    lang = _normalise_lang(lang)
    try:
        if lang == "ita":
            from spacy.lang.it.stop_words import STOP_WORDS as _SW
        else:
            from spacy.lang.en.stop_words import STOP_WORDS as _SW
        return set(_SW)
    except Exception:                       # pragma: no cover - spaCy always present
        return set(_FALLBACK_STOP_WORDS[lang])


def _vectorizer_stop_words(lang: str):
    """Stop words in the shape historically returned by ``setup_nlp_pipeline``.

    ``"english"`` for English (sklearn's built-in list) and an explicit list for
    Italian, matching the pre-existing behaviour used by RAKE / CountVectorizer.
    """
    if _normalise_lang(lang) == "ita":
        return sorted(get_stop_words("ita"))
    return "english"


# ---------------------------------------------------------------------------
# UDPipe backend
# ---------------------------------------------------------------------------

def _default_udpipe_cache_dir() -> Path:
    env = os.environ.get(_ENV_CACHE_DIR)
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "scptm" / "udpipe"


def _resolve_udpipe_model_path(
    lang: str,
    model_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    verbose: bool = True,
) -> Path:
    """Return a local path to the UDPipe model, downloading it if necessary."""
    lang = _normalise_lang(lang)

    explicit = model_path or os.environ.get(_ENV_MODEL_PATH[lang])
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"UDPipe model for '{lang}' not found at '{path}'."
            )
        return path

    root = Path(cache_dir).expanduser() if cache_dir else _default_udpipe_cache_dir()
    dest = root / UDPIPE_MODEL_FILENAMES[lang]
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        return dest

    root.mkdir(parents=True, exist_ok=True)
    _download_udpipe_model(lang, dest, verbose=verbose)
    return dest


def _requests_module():
    """Return ``requests`` if importable, else None.

    ``requests`` ships its own CA bundle (certifi) and is already a transitive
    dependency (huggingface_hub → requests).  It is preferred over
    ``urllib.request`` because on some systems Python's default OpenSSL trust
    store cannot verify hosts whose chain includes a corporate/intermediate
    certificate, even though the system store can.
    """
    try:
        import requests
    except ImportError:                     # pragma: no cover - always present here
        return None
    return requests


def _http_get_json(url: str, timeout: int = 60):
    import json

    headers = {"Accept": "application/json", "User-Agent": "scptm"}
    requests = _requests_module()
    if requests is not None:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _lookup_lindat_model_url(filename: str) -> Optional[str]:
    """Resolve a model's download URL through the LINDAT DSpace 7 REST API.

    Used as a fallback when the hard-coded bitstream UUIDs in
    ``UDPIPE_MODEL_URLS`` stop working (e.g. after another repository
    migration).
    """
    try:
        query = urllib.parse.quote(filename)
        search = _http_get_json(
            f"{_UDPIPE_API_BASE}/discover/search/objects?query={query}&size=20"
        )
        objects = (
            search.get("_embedded", {})
            .get("searchResult", {})
            .get("_embedded", {})
            .get("objects", [])
        )
        item_uuid = None
        for obj in objects:
            item = obj.get("_embedded", {}).get("indexableObject", {})
            if item.get("handle") == UDPIPE_ITEM_HANDLE:
                item_uuid = item.get("uuid")
                break
        if not item_uuid:
            return None

        bundles = _http_get_json(f"{_UDPIPE_API_BASE}/core/items/{item_uuid}/bundles")
        for bundle in bundles.get("_embedded", {}).get("bundles", []):
            if bundle.get("name") != "ORIGINAL":
                continue
            bitstreams = _http_get_json(
                f"{_UDPIPE_API_BASE}/core/bundles/{bundle['uuid']}/bitstreams?size=200"
            )
            for bitstream in bitstreams.get("_embedded", {}).get("bitstreams", []):
                if bitstream.get("name") == filename:
                    return bitstream.get("_links", {}).get("content", {}).get("href")
    except Exception:                       # noqa: BLE001 - best-effort fallback
        return None
    return None


def _download_file(url: str, dest: Path, verbose: bool = True) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    headers = {"User-Agent": "scptm"}
    requests = _requests_module()
    try:
        if requests is not None:
            with requests.get(url, headers=headers, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("Content-Length") or 0)
                with open(tmp, "wb") as fh:
                    done = 0
                    for chunk in resp.iter_content(1 << 20):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        done += len(chunk)
                        if verbose and total:
                            pct = 100.0 * done / total
                            sys.stdout.write(
                                f"\r  Downloading {dest.name}: {pct:5.1f}% "
                                f"({done / 1e6:.1f}/{total / 1e6:.1f} MB)"
                            )
                            sys.stdout.flush()
        else:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as fh:
                total = int(resp.headers.get("Content-Length") or 0)
                done = 0
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    done += len(chunk)
                    if verbose and total:
                        pct = 100.0 * done / total
                        sys.stdout.write(
                            f"\r  Downloading {dest.name}: {pct:5.1f}% "
                            f"({done / 1e6:.1f}/{total / 1e6:.1f} MB)"
                        )
                        sys.stdout.flush()
        if verbose:
            sys.stdout.write("\n")
        os.replace(tmp, dest)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise


def _download_udpipe_model(lang: str, dest: Path, verbose: bool = True) -> None:
    filename = UDPIPE_MODEL_FILENAMES[lang]
    if verbose:
        print(f"  UDPipe model '{filename}' not found — downloading...")
    errors: List[str] = []
    candidates = [UDPIPE_MODEL_URLS[lang]]
    for url in candidates:
        try:
            _download_file(url, dest, verbose=verbose)
            return
        except Exception as exc:            # noqa: BLE001 - try next candidate
            errors.append(f"{url}: {exc}")

    resolved = _lookup_lindat_model_url(filename)
    if resolved:
        try:
            _download_file(resolved, dest, verbose=verbose)
            return
        except Exception as exc:            # noqa: BLE001
            errors.append(f"{resolved}: {exc}")

    raise RuntimeError(
        f"Could not download the UDPipe model '{filename}' for '{lang}'.\n"
        "  Tried:\n    " + "\n    ".join(errors) + "\n"
        "  Download it manually from https://lindat.mff.cuni.cz/repository/handle/"
        f"{UDPIPE_ITEM_HANDLE} and pass its path via "
        "SCPTMConfig(udpipe_model_path=...) or the "
        f"{_ENV_MODEL_PATH[lang]} environment variable."
    )


class UDPipeToken:
    """Minimal spaCy-``Token``-compatible view of a UDPipe word."""

    __slots__ = (
        "i", "text", "lemma_", "pos_", "tag_", "dep_",
        "is_alpha", "is_stop", "head", "_head_id",
    )

    def __init__(
        self,
        i: int,
        text: str,
        lemma: str,
        pos: str,
        tag: str,
        dep: str,
        stop_words: frozenset,
    ) -> None:
        self.i = i
        self.text = text
        self.lemma_ = lemma
        self.pos_ = pos
        self.tag_ = tag
        self.dep_ = dep
        self.is_alpha = text.isalpha()
        self.is_stop = text.lower() in stop_words
        self.head = self          # replaced once all tokens are known
        self._head_id = 0

    def __repr__(self) -> str:      # pragma: no cover - debugging aid
        return f"UDPipeToken({self.text!r}, lemma={self.lemma_!r}, dep={self.dep_!r})"


class UDPipeDoc:
    """Iterable of :class:`UDPipeToken`, mirroring a spaCy ``Doc``."""

    __slots__ = ("_tokens",)

    def __init__(self, tokens: Iterable[UDPipeToken]) -> None:
        self._tokens = list(tokens)

    def __iter__(self):
        return iter(self._tokens)

    def __len__(self) -> int:
        return len(self._tokens)

    def __getitem__(self, idx):
        return self._tokens[idx]


class UDPipePipeline:
    """
    Callable UDPipe lemmatiser / dependency parser.

    Mimics the small slice of the spaCy ``Language`` API used by SCPTM:
    calling the object on a string returns an iterable of tokens exposing
    ``.lemma_``, ``.is_stop``, ``.is_alpha``, ``.dep_``, ``.head``.
    """

    def __init__(
        self,
        model,
        stop_words: Optional[Set[str]] = None,
        lang: str = "eng",
        name: str = "udpipe",
    ) -> None:
        self._model = model
        self._stop_words = frozenset(w.lower() for w in (stop_words or ()))
        self.lang = lang
        self.name = name
        # API parity with spaCy (UDPipe itself imposes no such limit).
        self.max_length = 2_000_000

    def __call__(self, text: str) -> UDPipeDoc:
        from ufal.udpipe import ProcessingError, Sentence

        error = ProcessingError()
        tokenizer = self._model.newTokenizer(self._model.DEFAULT)
        if not tokenizer:
            raise RuntimeError("The loaded UDPipe model does not provide a tokenizer.")
        tokenizer.setText(text)

        tokens: List[UDPipeToken] = []
        sentence = Sentence()
        while tokenizer.nextSentence(sentence, error):
            # Tagger and parser are no-ops for models that only ship a
            # tokenizer, but both are present in the standard UD models.
            if not self._model.tag(sentence, self._model.DEFAULT):
                raise RuntimeError(f"UDPipe tagging failed: {error.message}")
            if not self._model.parse(sentence, self._model.DEFAULT):
                raise RuntimeError(f"UDPipe parsing failed: {error.message}")

            by_id = {}
            for word in sentence.words:
                if word.id == 0:        # synthetic <root> node
                    continue
                token = UDPipeToken(
                    i=word.id,
                    text=word.form,
                    lemma=word.lemma or word.form,
                    pos=word.upostag,
                    tag=word.xpostag,
                    dep=word.deprel,
                    stop_words=self._stop_words,
                )
                token._head_id = word.head
                by_id[word.id] = token
                tokens.append(token)

            for token in by_id.values():
                # head == 0 is the sentence root (spaCy: the token is its own
                # head); multi-word/empty nodes never appear in ``words``.
                token.head = by_id.get(token._head_id, token)

            sentence = Sentence()

        if error.occurred():
            raise RuntimeError(f"UDPipe processing failed: {error.message}")
        return UDPipeDoc(tokens)

    def pipe(self, texts: Iterable[str], batch_size: int = 64, **kwargs):
        """
        spaCy-compatible batched iteration.

        UDPipe's C++ pipeline is a stateful, one-document-at-a-time API, so this
        is sequential; ``batch_size`` is accepted for signature parity with
        ``spacy.Language.pipe`` and ignored.
        """
        for text in texts:
            yield self(text)


def load_udpipe_pipeline(
    lang: str = "eng",
    model_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    verbose: bool = True,
) -> UDPipePipeline:
    """
    Load a UDPipe model for ``lang`` and wrap it in a spaCy-like adapter.

    Parameters
    ----------
    lang : str
        "eng" or "ita".
    model_path : str | None
        Explicit path to a ``.udpipe`` model.  Falls back to the
        ``SCPTM_UDPIPE_MODEL_{ENG,ITA}`` environment variable and then to an
        auto-downloaded model inside ``cache_dir``.
    cache_dir : str | None
        Directory used for auto-downloaded models
        (default: ``$SCPTM_UDPIPE_MODEL_DIR`` or ``~/.cache/scptm/udpipe``).
    verbose : bool
        Print download / loading progress.
    """
    try:
        from ufal.udpipe import Model
    except ImportError as exc:              # pragma: no cover - depends on env
        raise ImportError(
            "The 'ufal.udpipe' package is required for lemmatizer='udpipe'. "
            "Install it with:  pip install ufal.udpipe"
        ) from exc

    lang = _normalise_lang(lang)
    path = _resolve_udpipe_model_path(
        lang, model_path=model_path, cache_dir=cache_dir, verbose=verbose
    )
    if verbose:
        print(f"  Loading UDPipe model '{path.name}'...")
    model = Model.load(str(path))
    if not model:
        raise RuntimeError(
            f"ufal.udpipe could not load the model file '{path}'. "
            "The file may be corrupt or use an unsupported format."
        )
    return UDPipePipeline(model, stop_words=get_stop_words(lang), lang=lang,
                          name=path.name)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def setup_nlp_pipeline(
    lang: str = "eng",
    verbose: bool = True,
    lemmatizer: str = "spacy",
    udpipe_model_path: Optional[str] = None,
    udpipe_cache_dir: Optional[str] = None,
    device: str = "auto",
):
    """
    Load SBERT (on the best available accelerator) and a lemmatisation/parsing
    backend.

    Parameters
    ----------
    lang : str
        "eng" (default) or "ita".
    verbose : bool
        Print device/model info.
    lemmatizer : str
        "spacy" (default) or "udpipe".
    udpipe_model_path : str | None
        Path to a ``.udpipe`` model (only used when ``lemmatizer="udpipe"``).
    udpipe_cache_dir : str | None
        Cache directory for auto-downloaded UDPipe models.
    device : str
        SBERT compute device: "auto" (default), "cpu", "cuda" or "mps".
        Resolved through :func:`scptm.config.resolve_device`.

    Returns
    -------
    sbert_model : SentenceTransformer
    nlp : spaCy Language | UDPipePipeline
        Callable lemmatiser/parser exposing a common token interface.
    stop_words : list | str
        Stop word list for CountVectorizer / RAKE.
    """
    if lemmatizer not in LEMMATIZERS:
        raise ValueError(
            f"lemmatizer must be one of {list(LEMMATIZERS)}, got '{lemmatizer}'"
        )

    lang = _normalise_lang(lang)
    dev = resolve_device(device)
    sbert_name = _SBERT_MODELS[lang]

    # Loading SBERT + the spaCy/UDPipe model costs several seconds, and a K
    # sweep (search_k) builds one SCPTM per K.  Both objects are stateless at
    # inference time, so they are safe to share between instances.
    key = (lang, lemmatizer, udpipe_model_path, udpipe_cache_dir, str(dev))
    cached = _PIPELINE_CACHE.get(key)
    if cached is not None:
        sbert, nlp, stop_words = cached
        if verbose:
            print(
                f"NLP pipeline: reusing cached SBERT='{sbert_name}' on "
                f"{sbert.device}, {_backend_desc(nlp, lemmatizer, lang)} on CPU"
            )
        return sbert, nlp, stop_words

    sbert = _load_sentence_transformer(sbert_name, str(dev))

    if lemmatizer == "udpipe":
        nlp = load_udpipe_pipeline(
            lang,
            model_path=udpipe_model_path,
            cache_dir=udpipe_cache_dir,
            verbose=verbose,
        )
    else:
        import spacy

        spacy_model = _SPACY_MODELS[lang]
        try:
            nlp = spacy.load(spacy_model, disable=["ner"])
        except OSError:
            from spacy.cli import download
            download(spacy_model)
            nlp = spacy.load(spacy_model, disable=["ner"])
        nlp.max_length = 2_000_000

    stop_words = _vectorizer_stop_words(lang)
    _PIPELINE_CACHE[key] = (sbert, nlp, stop_words)

    if verbose:
        print(
            f"NLP pipeline: SBERT='{sbert_name}' on {sbert.device}, "
            f"{_backend_desc(nlp, lemmatizer, lang)} on CPU"
        )

    return sbert, nlp, stop_words


def _backend_desc(nlp, lemmatizer: str, lang: str) -> str:
    """Human-readable backend description for log lines."""
    if lemmatizer == "udpipe":
        return f"UDPipe='{getattr(nlp, 'name', 'udpipe')}'"
    return f"spaCy='{_SPACY_MODELS[lang]}'"


def clear_pipeline_cache() -> int:
    """
    Drop the cached (SBERT, lemmatiser) pipelines.

    Returns the number of entries that were freed.  Useful when switching
    device/model configuration in a long-running process, or to release the
    accelerator memory held by cached models.
    """
    count = len(_PIPELINE_CACHE)
    _PIPELINE_CACHE.clear()
    return count
