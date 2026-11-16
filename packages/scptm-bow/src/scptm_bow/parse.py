"""
scptm_bow/parse.py
-------------------
Lemmatisation and dependency parsing — **one pass over the corpus**.

Design note
-----------
The heavy part of any structure-aware topic model is parsing the corpus, so this
module parses exactly once and derives everything from that single pass:

* ``lemma_docs``    -> lemmatised documents (fed to ``CountVectorizer``)
* ``bow``           -> document x term counts; its non-zero pattern *is* the
                       doc-word edge set, so no second pass is needed to build it
* ``dep_pairs``     -> aggregated (head, dependent) lemma pairs, the word-word
                       graph used for regularisation
* ``dep_counts``    -> (head, relation, dependent) triples, kept for MWE
                       extraction (the interpretability output)

Both backends expose the same tiny token interface, so the pass is shared.
"""

import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, FrozenSet, Iterable, List, NamedTuple, Optional, Tuple, Union

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

from .config import BowConfig
from .store import cache_signature, load_cache, save_cache

# Lemmas shorter than this are dropped (same convention as the main package).
MIN_LEMMA_CHARS = 3

# Unicode-aware token pattern: at least two letters, no digits or underscores.
# An ASCII-only pattern would silently drop "città", "però", "più", …
TOKEN_PATTERN = r"(?u)\b[^\W\d_]{2,}\b"

SPACY_MODELS = {"eng": "en_core_web_sm", "ita": "it_core_news_sm"}
UDPIPE_MODEL_ENV = {"eng": "SCPTM_BOW_UDPIPE_ENG", "ita": "SCPTM_BOW_UDPIPE_ITA"}

_FALLBACK_STOPWORDS = {
    "eng": frozenset(
        "a an and are as at be been but by for from had has have he her his i in is it "
        "its of on or that the their they this to was were which will with you your".split()
    ),
    "ita": frozenset(
        "a ad al alla che chi ci con da dal dei del della di e ed gli ha ho i il in io "
        "la le lo ma mi ne nei nel nella no non o per più si sono su tra un una uno".split()
    ),
}


class Tok(NamedTuple):
    """Minimal token view shared by both parser backends."""

    lemma: str
    dep: str
    head_lemma: str
    is_stop: bool
    is_alpha: bool


@dataclass
class ParsedCorpus:
    """Everything the single parse pass produced."""

    vocab: np.ndarray                       # (V,) lemmas
    bow: sparse.csr_matrix                  # (n_docs, V) term counts
    dep_src: np.ndarray                     # (E,) vocab indices
    dep_dst: np.ndarray                     # (E,) vocab indices
    dep_weight: np.ndarray                  # (E,) co-occurrence counts
    dep_counts: Dict[Tuple[str, str, str], int]   # (head, relation, dependent)
    lemma_docs: List[str]                   # lemmatised documents
    n_docs: int

    @property
    def n_vocab(self) -> int:
        return len(self.vocab)

    def term_doc_matrix(self) -> sparse.csr_matrix:
        """Documents x terms, the layout NMF works on."""
        return self.bow.T.tocsr()


def chunk_documents(
    documents: Iterable[str],
    max_chunk_chars: int = 800,
) -> List[str]:
    """
    Split long documents on sentence boundaries into ~``max_chunk_chars`` pieces.

    **Chunking never drops a document**: every non-empty input contributes at
    least one segment, and the trailing piece of a long document is kept even
    when short.  (The neural package used to discard tails under 50 characters,
    which silently deleted whole short documents.)  Sentence splitting covers
    ``.``, ``!`` and ``?`` and never appends a spurious period.
    """
    import re

    segments: List[str] = []
    for text in documents:
        clean = " ".join(str(text).split())
        if not clean:
            continue
        if len(clean) <= max_chunk_chars:
            segments.append(clean)
            continue

        chunk = ""
        for sentence in re.split(r"(?<=[.!?])\s+", clean):
            if not sentence:
                continue
            if chunk and len(chunk) + 1 + len(sentence) > max_chunk_chars:
                segments.append(chunk)
                chunk = sentence
            else:
                chunk = f"{chunk} {sentence}" if chunk else sentence
        if chunk:
            segments.append(chunk)
    return segments


def get_stop_words(lang: str) -> frozenset:
    """Stop words from spaCy's language data when available, else a small set."""
    try:
        if lang == "ita":
            from spacy.lang.it.stop_words import STOP_WORDS
        else:
            from spacy.lang.en.stop_words import STOP_WORDS
        return frozenset(STOP_WORDS)
    except Exception:                       # pragma: no cover - spacy absent
        return _FALLBACK_STOPWORDS[lang]


# ---------------------------------------------------------------------------
# Parser backends
# ---------------------------------------------------------------------------

class _SpacyParser:
    name = "spacy"

    def __init__(self, cfg: BowConfig, stop_words: frozenset):
        try:
            import spacy
        except ImportError as exc:          # pragma: no cover
            raise ImportError(
                "parser_backend='spacy' requires spaCy: pip install spacy"
            ) from exc

        model = SPACY_MODELS[cfg.lang]
        try:
            self._nlp = spacy.load(model, disable=["ner"])
        except OSError as exc:
            raise OSError(
                f"spaCy model '{model}' is not installed. Run:\n"
                f"  python -m spacy download {model}\n"
                "(or use parser_backend='udpipe' with an explicit model path)."
            ) from exc
        self._stop = stop_words

    def pipe(self, texts: Iterable[str], batch_size: int = 64) -> Iterable[List[Tok]]:
        for doc in self._nlp.pipe(texts, batch_size=batch_size):
            yield [
                Tok(t.lemma_.lower(), t.dep_, t.head.lemma_.lower(), t.is_stop, t.is_alpha)
                for t in doc
            ]


class _UDPipeParser:
    name = "udpipe"

    def __init__(self, cfg: BowConfig, stop_words: frozenset):
        try:
            from ufal.udpipe import Model, Pipeline
        except ImportError as exc:          # pragma: no cover
            raise ImportError(
                "parser_backend='udpipe' requires ufal.udpipe: pip install ufal.udpipe"
            ) from exc

        path = cfg.udpipe_model_path or os.environ.get(UDPIPE_MODEL_ENV[cfg.lang])
        if not path or not Path(path).is_file():
            raise FileNotFoundError(
                f"UDPipe model for '{cfg.lang}' not found. Pass "
                f"BowConfig(udpipe_model_path=...) or set "
                f"{UDPIPE_MODEL_ENV[cfg.lang]}."
            )
        model = Model.load(str(path))
        if not model:
            raise RuntimeError(f"ufal.udpipe could not load '{path}'.")
        self._model = model
        self._pipe = Pipeline(model, "tokenize", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu")
        self._stop = stop_words
        self.name = Path(path).name

    def pipe(self, texts: Iterable[str], batch_size: int = 64) -> Iterable[List[Tok]]:
        from ufal.udpipe import ProcessingError

        error = ProcessingError()
        for text in texts:
            conllu = self._pipe.process(text, error)
            if error.occurred():
                raise RuntimeError(f"UDPipe failed: {error.message}")

            # id, lemma, relation, head-id, is_stop, is_alpha
            rows: List[Tuple[int, str, str, int, bool, bool]] = []
            for line in conllu.splitlines():
                if not line or line.startswith("#"):
                    continue
                cols = line.split("\t")
                if len(cols) < 8 or "-" in cols[0] or "." in cols[0]:
                    continue                     # multi-word token / empty node
                form, lemma, dep, head = cols[1], cols[2], cols[7], cols[6]
                lemma = (lemma if lemma != "_" else form).lower()
                rows.append((
                    int(cols[0]),
                    lemma,
                    "" if dep == "_" else dep,
                    int(head) if head.lstrip("-").isdigit() else 0,
                    form.lower() in self._stop,
                    form.isalpha(),
                ))

            by_id = {r[0]: r[1] for r in rows}
            yield [
                Tok(
                    lemma=r[1],
                    dep=r[2],
                    head_lemma=by_id.get(r[3], "") if r[3] > 0 else "",
                    is_stop=r[4],
                    is_alpha=r[5],
                )
                for r in rows
            ]


def build_parser(cfg: BowConfig) -> Callable[..., Iterable[List[Tok]]]:
    """Instantiate the configured parser backend."""
    stop = get_stop_words(cfg.lang)
    if cfg.parser_backend == "udpipe":
        return _UDPipeParser(cfg, stop)
    return _SpacyParser(cfg, stop)


# ---------------------------------------------------------------------------
# The single pass
# ---------------------------------------------------------------------------

def _lemmas_of(tokens) -> List[str]:
    """The lemma sequence of one parsed document (storage words only)."""
    return [
        tok.lemma
        for tok in tokens
        if not tok.is_stop and tok.is_alpha and len(tok.lemma) >= MIN_LEMMA_CHARS
    ]


def lemmatize_documents(
    documents: List[str],
    cfg: BowConfig,
    verbose: bool = False,
) -> List[str]:
    """
    Lemmatise documents **without** building a vocabulary.

    Used to project held-out documents onto an already-fitted vocabulary: the
    vocabulary must not be re-derived from the new documents, or the test set
    would leak into the model (and a small batch would fail the ``min_df``
    pruning outright).
    """
    if cfg.apply_chunking:
        documents = chunk_documents(documents, cfg.max_chunk_chars)
    parser = build_parser(cfg)
    if verbose:
        print(f"  lemmatising {len(documents):,} documents ({parser.name})...")
    stream = tqdm(parser.pipe(documents), total=len(documents), desc="Parsing") \
        if verbose else parser.pipe(documents)
    return [" ".join(_lemmas_of(tokens)) for tokens in stream]


def parse_corpus(
    documents: List[str],
    cfg: BowConfig,
    cache_path: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> ParsedCorpus:
    """
    Parse the corpus once and return every artefact the model needs.

    With ``cache_path`` the result is stored and reused across runs, guarded by
    a signature (corpus hash + parsing/vocabulary configuration).
    """
    if cfg.apply_chunking:
        original = len(documents)
        documents = chunk_documents(documents, cfg.max_chunk_chars)
        if verbose:
            print(
                f"  chunking: {original:,} documents -> {len(documents):,} segments "
                f"(max {cfg.max_chunk_chars} chars)"
            )

    signature = cache_signature(cfg, documents)

    if cache_path is not None:
        cached = load_cache(cache_path, signature)
        if cached is not None:
            return ParsedCorpus(
                vocab=cached["vocab"],
                bow=cached["bow"],
                dep_src=cached["dep_src"],
                dep_dst=cached["dep_dst"],
                dep_weight=cached["dep_weight"],
                dep_counts=cached["dep_counts"],
                lemma_docs=cached["lemma_docs"],
                n_docs=cached["n_docs"],
            )

    parser = build_parser(cfg)
    dep_types: FrozenSet[str] = cfg.dep_types

    lemma_docs: List[str] = []
    pair_counts: Counter = Counter()
    triple_counts: Counter = Counter()

    if verbose:
        print(f"1/3  Parsing + lemmatising ({parser.name}) — single pass...")
    stream = tqdm(parser.pipe(documents), total=len(documents), desc="Parsing") \
        if verbose else parser.pipe(documents)
    for tokens in stream:
        lemmas: List[str] = _lemmas_of(tokens)
        for tok in tokens:
            if tok.is_stop or not tok.is_alpha or len(tok.lemma) < MIN_LEMMA_CHARS:
                continue
            head = tok.head_lemma
            if (
                tok.dep in dep_types
                and head
                and head != tok.lemma
                and len(head) >= MIN_LEMMA_CHARS
            ):
                pair_counts[(head, tok.lemma)] += 1
                triple_counts[(head, tok.dep, tok.lemma)] += 1
        lemma_docs.append(" ".join(lemmas))

    print("2/3  Building vocabulary and BoW...")
    vectorizer = CountVectorizer(
        stop_words=None,               # already filtered during parsing
        min_df=cfg.min_df,
        max_features=cfg.max_features,
        token_pattern=TOKEN_PATTERN,
    )
    bow = vectorizer.fit_transform(lemma_docs).tocsr()
    vocab = vectorizer.get_feature_names_out()
    index = {w: i for i, w in enumerate(vocab)}

    print("3/3  Mapping dependency edges onto the vocabulary...")
    src, dst, weight = [], [], []
    for (a, b), count in pair_counts.items():
        ia = index.get(a)
        ib = index.get(b)
        if ia is None or ib is None:
            continue
        src.append(ia)
        dst.append(ib)
        weight.append(count)

    dep_src = np.asarray(src, dtype=np.int32)
    dep_dst = np.asarray(dst, dtype=np.int32)
    dep_weight = np.asarray(weight, dtype=np.float32)
    print(
        f"  vocabulary: {len(vocab):,} lemmas | docs: {len(lemma_docs):,} | "
        f"dependency edges: {len(dep_src):,} | triples: {len(triple_counts):,}"
    )

    if cache_path is not None:
        save_cache(
            cache_path,
            {
                "vocab": vocab,
                "bow": bow,
                "dep_src": dep_src,
                "dep_dst": dep_dst,
                "dep_weight": dep_weight,
                "dep_counts": dict(triple_counts),
                "lemma_docs": lemma_docs,
                "n_docs": len(lemma_docs),
            },
            signature,
        )

    return ParsedCorpus(
        vocab=vocab,
        bow=bow,
        dep_src=dep_src,
        dep_dst=dep_dst,
        dep_weight=dep_weight,
        dep_counts=dict(triple_counts),
        lemma_docs=lemma_docs,
        n_docs=len(lemma_docs),
    )
