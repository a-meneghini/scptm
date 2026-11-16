"""
scptm_bow/model.py
-------------------
``BowSCPTM`` — the whole pipeline in one object: single-pass parse, word
vectors, graph construction, regularised NMF, MWE explanation.

Deliberately different from the neural SCPTM in three ways: no embeddings are
trained, no graph network is run, and nothing is printed unless you ask for it.
"""

import dataclasses
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import sparse
from scipy.optimize import linear_sum_assignment

from . import metrics as _metrics
from .config import BowConfig
from .explain import extract_mwes, mwe_compactness, topic_labels
from .factorize import GraphRegularizedNMF, NMFResult
from .graphs import Graph, similarity_graph, syntax_graph
from .parse import (
    MIN_LEMMA_CHARS,
    ParsedCorpus,
    build_parser,
    get_stop_words,
    lemmatize_documents,
    parse_corpus,
)
from .stats import bootstrap_ci
from .vectors import build_word_vectors, document_vectors


class BowSCPTM:
    """
    Structure-aware topic model without neural embeddings.

    Examples
    --------
    >>> model = BowSCPTM(num_topics=10)
    >>> model.fit(documents, cache_path="corpus.pkl")
    >>> theta = model.theta
    >>> model.get_topic_info(top_k=8)
    >>> model.save("model.pkl")
    >>> model = BowSCPTM.load("model.pkl")
    """

    def __init__(self, config: Optional[BowConfig] = None, **kwargs):
        if config is not None and kwargs:
            raise TypeError(
                "Pass either config= or individual keyword arguments, not both "
                f"(got config and {sorted(kwargs)})."
            )
        self.config = config if config is not None else BowConfig(**kwargs)
        self._is_fitted = False
        self.parsed: Optional[ParsedCorpus] = None
        self.vectors: Optional[np.ndarray] = None
        self.syntax_graph: Optional[Graph] = None
        self.embed_graph: Optional[Graph] = None
        self.result: Optional[NMFResult] = None
        self._npmi: Optional[np.ndarray] = None
        self._syntax_npmi: Optional[float] = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        documents: Sequence[str],
        cache_path: Optional[Union[str, Path]] = None,
        verbose: bool = True,
    ) -> "BowSCPTM":
        """
        Fit the model.

        Parameters
        ----------
        documents : sequence of str
            The corpus.  With ``config.apply_chunking`` long documents are split
            at sentence boundaries, so a row of ``theta`` is then a segment, not
            necessarily a whole document.
        cache_path : str | Path | None
            Where to store/reuse the parse artefacts.  Validated by a signature
            (corpus hash + parsing/vocabulary config).
        verbose : bool
            Print progress.  Defaults to True because fitting is a long
            operation; set False in scripts and notebooks.
        """
        cfg = self.config
        documents = [str(d) for d in documents]
        if not documents:
            raise ValueError("documents is empty.")

        if verbose:
            print(f"[scptm-bow] {len(documents):,} documents | K={cfg.num_topics} | "
                  f"lang={cfg.lang} | parser={cfg.parser_backend}")

        self.parsed = parse_corpus(documents, cfg, cache_path=cache_path)

        if verbose:
            print("  building word vectors...")
        self.vectors = build_word_vectors(self.parsed, cfg)

        if verbose:
            print("  building graphs...")
        n_vocab = self.parsed.n_vocab
        self.syntax_graph = syntax_graph(
            self.parsed.dep_src, self.parsed.dep_dst, self.parsed.dep_weight,
            n_vocab, top_k=cfg.graph_top_k,
        )
        self.embed_graph = (
            similarity_graph(self.vectors, top_k=cfg.graph_top_k)
            if self.vectors is not None else None
        )
        if verbose:
            print(
                f"  graphs: syntax={self.syntax_graph.n_edges:,} edges | "
                f"semantic="
                f"{self.embed_graph.n_edges if self.embed_graph else 0:,} edges"
            )
            print("  factorising...")

        nmf = GraphRegularizedNMF(
            n_components=cfg.num_topics,
            lambda_syntax=cfg.lambda_syntax,
            lambda_embed=cfg.lambda_embed,
            max_iter=cfg.nmf_max_iter,
            tol=cfg.nmf_tol,
            random_state=cfg.random_state,
            divergence=cfg.divergence,
        )
        self.result = nmf.fit_transform(
            self.parsed.bow,
            syntax_graph=self.syntax_graph,
            embed_graph=self.embed_graph,
        )

        # Document-topic mixtures as distributions.
        W = np.clip(self.result.W, 0.0, None)
        row = W.sum(axis=1, keepdims=True)
        row[row == 0] = 1.0
        self.result.W = W / row
        self._is_fitted = True

        if verbose:
            print(
                f"  done: {self.result.n_iter} iterations | "
                f"objective={self.result.objective:,.1f}"
            )
        return self

    def fit_transform(
        self,
        documents: Sequence[str],
        cache_path: Optional[Union[str, Path]] = None,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Fit and return the document-topic matrix (scikit-learn convention).

        Same as ``fit(...)`` followed by ``model.theta``; provided so a
        ``SCPTM``-style call works unchanged.  Note that a notebook renders the
        returned array, so prefer ``fit()`` + ``model.theta`` if you do not want
        that.
        """
        self.fit(documents, cache_path=cache_path, verbose=verbose)
        return self.theta

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(
        self,
        path: Union[str, Path],
        include_corpus: bool = False,
    ) -> Path:
        """
        Save the fitted model to ``path`` and return the path.

        The file is versioned and written atomically, and loading it needs no
        parser: only the first :meth:`transform` call builds one.

        Parameters
        ----------
        path : str | Path
            Destination file (``.pkl`` by convention).
        include_corpus : bool
            Also store the lemmatised training documents.  They are not needed
            by the fitted model, so the default keeps the file small; set it if
            you want to inspect the parsed corpus or rebuild the word vectors.
        """
        from .persistence import save_model

        return save_model(self, path, include_corpus=include_corpus)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BowSCPTM":
        """
        Load a model saved with :meth:`save`.

        Raises ``FileNotFoundError`` if ``path`` does not exist and
        ``ValueError`` if the file is not a scptm-bow model or was written by a
        newer format version.
        """
        from .persistence import load_model

        return load_model(path)

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Call fit() before using this method.")

    @property
    def theta(self) -> np.ndarray:
        """Document-topic mixtures, shape (n_docs, K)."""
        self._check_fitted()
        return self.result.W

    @property
    def topic_word(self) -> np.ndarray:
        """Topic-term matrix H, shape (K, V), rows summing to 1."""
        self._check_fitted()
        return self.result.H

    @property
    def vocab(self) -> np.ndarray:
        self._check_fitted()
        return self.parsed.vocab

    def _blocked_words(self, extra: Optional[Sequence[str]] = None) -> set:
        """
        Words hidden from the *reported* topic words.

        The vocabulary itself is untouched, so this never changes the fitted
        model, the cached artefacts or the metrics' denominators — it only keeps
        function words out of the lists you read.  The set is the language stop
        words (already filtered at parse time, kept as a safety net) plus
        ``config.topic_stopwords`` plus any per-call ``extra``.
        """
        blocked = set(get_stop_words(self.config.lang))
        if self.config.topic_stopwords:
            blocked |= {str(w).lower() for w in self.config.topic_stopwords}
        if extra:
            blocked |= {str(w).lower() for w in extra}
        return blocked

    def top_words(
        self,
        top_k: int = 10,
        drop_stopwords: bool = True,
        extra_stopwords: Optional[Sequence[str]] = None,
    ) -> List[List[str]]:
        """
        Top-``top_k`` words per topic, ordered by topic-term weight.

        Parameters
        ----------
        top_k : int
        drop_stopwords : bool
            Skip stop words (default True).  They carry no topic information but
            NMF ranks by frequency, so they surface in the top of every topic.
        extra_stopwords : sequence of str | None
            Additional words to hide for this call (case-insensitive).  Use
            :meth:`common_words` to find corpus-specific candidates.
        """
        self._check_fitted()
        vocab = self.parsed.vocab
        weights = self.result.H
        if drop_stopwords:
            blocked = self._blocked_words(extra_stopwords)
            if blocked:
                keep = np.fromiter(
                    (str(w) not in blocked for w in vocab), dtype=bool, count=len(vocab)
                )
                vocab = vocab[keep]
                weights = weights[:, keep]
        k = min(top_k, len(vocab))
        if k == 0:
            return [[] for _ in range(self.result.n_topics)]
        order = np.argsort(-weights, axis=1)[:, :k]
        return [[str(vocab[i]) for i in row] for row in order]

    def common_words(self, top_n: int = 30) -> List[Dict[str, object]]:
        """
        The most document-frequent lemmas in the vocabulary.

        These are the corpus-specific function words that NMF promotes to the top
        of every topic (spaCy's stop list does not contain them: "ciò", "esso",
        "proprio", …).  Use them to fill ``BowConfig.topic_stopwords``.
        """
        self._check_fitted()
        bow = self.parsed.bow
        doc_freq = np.asarray((bow > 0).sum(axis=0)).ravel()
        n_docs = max(bow.shape[0], 1)
        order = np.argsort(-doc_freq)[:top_n]
        return [
            {
                "word": str(self.parsed.vocab[i]),
                "doc_freq": int(doc_freq[i]),
                "share": round(float(doc_freq[i]) / n_docs, 3),
            }
            for i in order
        ]

    def get_topic_info(
        self,
        top_k: int = 10,
        drop_stopwords: bool = True,
        extra_stopwords: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, object]]:
        """
        One record per topic: id, document count, top words, best MWEs.

        Returns a list of dicts (``pandas.DataFrame(model.get_topic_info())``
        gives a table).  Nothing is printed — that was a papercut in the neural
        package.
        """
        self._check_fitted()
        words = self.top_words(top_k, drop_stopwords, extra_stopwords)
        mwes = self.get_mwes(top_k=3, drop_stopwords=drop_stopwords,
                            extra_stopwords=extra_stopwords)
        dominant = self.theta.argmax(axis=1)
        labels = topic_labels(words, mwes, n=3)
        records = []
        for k in range(self.result.n_topics):
            records.append({
                "topic_id": k + 1,
                "size": int((dominant == k).sum()),
                "label": labels[k],
                "top_words": ", ".join(words[k]),
                "mwes": ", ".join(m["phrase"] for m in mwes[f"Topic_{k + 1}"]),
            })
        return sorted(records, key=lambda r: -r["size"])

    def get_mwes(
        self,
        top_k: int = 15,
        drop_stopwords: bool = True,
        extra_stopwords: Optional[Sequence[str]] = None,
    ) -> Dict[str, List[Dict[str, object]]]:
        """Multiword expressions per topic, from the dependency triples."""
        self._check_fitted()
        return extract_mwes(
            self.top_words(max(top_k, 10), drop_stopwords, extra_stopwords),
            self.parsed.dep_counts,
            top_k=top_k,
        )

    # ------------------------------------------------------------------
    # Out-of-sample inference
    # ------------------------------------------------------------------

    def _count_matrix(self, lemma_docs: Sequence[str]) -> sparse.csr_matrix:
        """Turn lemmatised documents into a count matrix over the fitted vocab."""
        index = {w: i for i, w in enumerate(self.parsed.vocab)}
        rows, cols, data = [], [], []
        for r, doc in enumerate(lemma_docs):
            for word, count in Counter(doc.split()).items():
                c = index.get(word)
                if c is not None:
                    rows.append(r)
                    cols.append(c)
                    data.append(count)
        return sparse.csr_matrix(
            (np.asarray(data, dtype=np.float32), (rows, cols)),
            shape=(len(lemma_docs), self.parsed.n_vocab),
        )

    def transform(self, documents: Sequence[str]) -> np.ndarray:
        """
        Infer topic mixtures for new documents (fold-in, no re-training).

        Only lemmatisation is needed — the word vectors and the graphs are not
        recomputed.
        """
        self._check_fitted()
        documents = [str(d) for d in documents]
        parser = build_parser(self.config)
        lemma_docs = [
            " ".join(
                t.lemma for t in tokens
                if not t.is_stop and t.is_alpha and len(t.lemma) >= MIN_LEMMA_CHARS
            )
            for tokens in parser.pipe(documents)
        ]
        return self._fold_in_bow(self._count_matrix(lemma_docs))

    def document_vectors(self) -> Optional[np.ndarray]:
        """Mean word vector per document (for plots / semantic metrics)."""
        self._check_fitted()
        return document_vectors(self.parsed.bow, self.vectors)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, top_k: int = 10) -> Dict[str, float]:
        """
        Topic-quality metrics: NPMI, diversity, redundancy, exclusivity.

        Returns the dict; print it yourself if you want to see it.
        """
        self._check_fitted()
        return _metrics.summary(
            self.top_words(top_k), self.parsed.bow, self.parsed.vocab,
            H=self.result.H, top_k=top_k,
        )

    def evaluate_no_syntax(self, top_k: int = 10) -> Dict[str, float]:
        """
        Same model with the graph penalties switched off.

        The difference against :meth:`evaluate` is the marginal value of the
        syntactic and semantic structure — the question the neural package needs
        a GNN ablation to answer.
        """
        self._check_fitted()
        plain = GraphRegularizedNMF(
            n_components=self.config.num_topics,
            lambda_syntax=0.0,
            lambda_embed=0.0,
            max_iter=self.config.nmf_max_iter,
            tol=self.config.nmf_tol,
            random_state=self.config.random_state,
            divergence=self.config.divergence,
        )
        result = plain.fit_transform(self.parsed.bow)
        vocab = self.parsed.vocab
        H = result.H
        blocked = self._blocked_words()
        keep = np.fromiter((str(w) not in blocked for w in vocab),
                           dtype=bool, count=len(vocab))
        vocab, H = vocab[keep], H[:, keep]
        k = min(top_k, len(vocab))
        order = np.argsort(-H, axis=1)[:, :k]
        words = [[str(vocab[i]) for i in row] for row in order]
        return _metrics.summary(words, self.parsed.bow, vocab, H=result.H, top_k=top_k)

    def mwe_compactness(self, top_k: int = 10) -> float:
        """Mean cosine similarity between the parts of each extracted MWE."""
        self._check_fitted()
        return mwe_compactness(self.get_mwes(), self.vectors, self.parsed.vocab, top_k)

    # ------------------------------------------------------------------
    # sklearn-style parameter protocol
    # ------------------------------------------------------------------

    def get_params(self, deep: bool = True) -> Dict[str, object]:
        """
        Constructor parameters, sklearn-style.

        Returns the config object, which is what ``__init__`` accepts — so
        ``sklearn.base.clone(model)`` works.  This is a *partial* protocol: the
        model does not subclass ``BaseEstimator`` because it holds fitted state
        that sklearn's cloning rules do not cover.
        """
        return {"config": self.config}

    def set_params(self, **params) -> "BowSCPTM":
        """Set ``config`` wholesale, or override individual config fields."""
        if "config" in params:
            self.config = params.pop("config")
        if params:
            valid = {f.name for f in dataclasses.fields(self.config)}
            unknown = set(params) - valid
            if unknown:
                raise ValueError(
                    f"unknown config field(s) {sorted(unknown)}; "
                    f"valid: {sorted(valid)}"
                )
            self.config = dataclasses.replace(self.config, **params)
        return self

    # ------------------------------------------------------------------
    # Held-out evaluation
    # ------------------------------------------------------------------

    def encode_documents(
        self,
        documents: Sequence[str],
        cache_path: Optional[Union[str, Path]] = None,
        verbose: bool = False,
    ) -> sparse.csr_matrix:
        """
        Project new documents onto the **fitted** vocabulary.

        The documents are parsed with the same backend and counted against
        ``self.parsed.vocab``; lemmas outside the training vocabulary are dropped
        (they carry no information for the fitted topics).  This is what makes a
        genuine held-out evaluation possible: the vocabulary must not be rebuilt,
        or the test data would leak into the model.
        """
        self._check_fitted()
        docs = [str(d) for d in documents]
        if not docs:
            raise ValueError("documents is empty.")
        lemma_docs = lemmatize_documents(docs, self.config, verbose=verbose)
        return self._count_matrix(lemma_docs)

    def _fold_in_bow(self, bow: sparse.csr_matrix) -> np.ndarray:
        """Topic mixtures for a count matrix over the fitted vocabulary."""
        W = GraphRegularizedNMF(
            n_components=self.config.num_topics,
            random_state=self.config.random_state,
            divergence=self.result.divergence,
        ).transform(bow, self.result.H)
        row = W.sum(axis=1, keepdims=True)
        row[row == 0] = 1.0
        return W / row

    def heldout_perplexity(
        self,
        documents: Sequence[str],
        cache_path: Optional[Union[str, Path]] = None,
    ) -> float:
        """
        Held-out perplexity under the fitted topics.

        New documents are folded in (``W`` only, ``H`` fixed), the predicted word
        distribution ``p(w|d)`` is read off ``W H``, and

            perplexity = exp( -sum_{d,w: X_dw > 0} X_dw log p_dw / sum_{d,w} X_dw )

        Lower is better.  This is the standard held-out criterion, and it is only
        meaningful *because* ``H`` and the vocabulary are frozen — refitting the
        vocabulary on the test documents would leak information and inflate it.
        """
        self._check_fitted()
        bow = self.encode_documents(documents, cache_path=cache_path)
        if bow.nnz == 0:
            raise ValueError(
                "no held-out token overlaps the fitted vocabulary; "
                "the perplexity is undefined."
            )
        W = self._fold_in_bow(bow)
        recon = np.asarray(W @ self.result.H, dtype=np.float64)
        row_sums = recon.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        logp = np.log(np.maximum(recon / row_sums, 1e-300))

        coo = bow.tocoo()
        observed = coo.data.astype(np.float64)
        log_lik = -float(np.sum(observed * logp[coo.row, coo.col]))
        return float(np.exp(log_lik / observed.sum()))

    # ------------------------------------------------------------------
    # Bootstrap uncertainty
    # ------------------------------------------------------------------

    @staticmethod
    def _match_topics(H_ref: np.ndarray, H_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Align ``H_new`` rows to ``H_ref`` rows by maximum cosine similarity."""
        a = H_ref / np.maximum(np.linalg.norm(H_ref, axis=1, keepdims=True), 1e-12)
        b = H_new / np.maximum(np.linalg.norm(H_new, axis=1, keepdims=True), 1e-12)
        sim = a @ b.T
        rows, cols = linear_sum_assignment(-sim)
        return cols, sim[rows, cols]

    def bootstrap(self, n_resamples: int = 20, seed: int = 0) -> Dict[str, object]:
        """
        Refit on document resamples and report how stable the topics are.

        Only the factorization is repeated — parsing, vocabulary and graphs stay
        fixed — so this is cheap.  Each resample's topics are aligned to the
        reference by an optimal assignment on cosine similarity, and the mean
        matched similarity is the stability estimate (1.0 = identical topics).
        """
        self._check_fitted()
        cfg = self.config
        X = self.parsed.bow
        n_docs = X.shape[0]
        H_ref = self.result.H
        rng = np.random.default_rng(seed)

        per_resample: List[np.ndarray] = []
        prevalences: List[np.ndarray] = []
        for b in range(n_resamples):
            idx = rng.integers(0, n_docs, size=n_docs)
            res = GraphRegularizedNMF(
                n_components=cfg.num_topics,
                lambda_syntax=cfg.lambda_syntax,
                lambda_embed=cfg.lambda_embed,
                max_iter=cfg.nmf_max_iter,
                tol=cfg.nmf_tol,
                random_state=cfg.random_state + b + 1,
                divergence=cfg.divergence,
            ).fit_transform(X[idx], self.syntax_graph, self.embed_graph)
            _, sims = self._match_topics(H_ref, res.H)
            per_resample.append(sims)
            theta_b = res.W / np.maximum(res.W.sum(axis=1, keepdims=True), 1e-12)
            prevalences.append(theta_b.mean(axis=0))

        sims = np.asarray(per_resample)                      # (B, K)
        prev = np.asarray(prevalences)                       # (B, K)
        return {
            "topic_stability": float(sims.mean()),
            "per_topic_stability": sims.mean(axis=0),
            "stability_sd": sims.std(axis=0, ddof=1) if len(sims) > 1 else np.zeros(sims.shape[1]),
            "prevalence_sd": prev.std(axis=0, ddof=1) if len(prev) > 1 else np.zeros(prev.shape[1]),
            "n_resamples": int(n_resamples),
        }

    # ------------------------------------------------------------------
    # Covariates (STM's estimateEffect, post-hoc)
    # ------------------------------------------------------------------

    def estimate_effect(
        self,
        covariates: Union[np.ndarray, Sequence[float]],
        names: Optional[Sequence[str]] = None,
        n_bootstrap: int = 200,
        level: float = 0.95,
        seed: int = 0,
    ) -> List[Dict[str, object]]:
        """
        Regress each topic's prevalence on document covariates, with bootstrap CIs.

        This is the post-hoc counterpart of STM's ``estimateEffect``: covariates
        do **not** enter the factorization, they are regressed on ``theta``
        afterwards.  That keeps the topic solution independent of the covariates
        (no feedback into how topics are formed) at the price of not modelling
        the covariate effect jointly — state which one you are claiming.
        """
        self._check_fitted()
        theta = self.theta
        Z = np.asarray(covariates, dtype=np.float64)
        if Z.ndim == 1:
            Z = Z[:, None]
        n_docs, n_cov = Z.shape
        if n_docs != theta.shape[0]:
            raise ValueError(
                f"covariates has {n_docs} rows but theta has {theta.shape[0]}; "
                "with apply_chunking=True a row is a segment, not a document."
            )
        design = np.column_stack([np.ones(n_docs), Z])
        term_names = ["intercept"] + list(
            names if names is not None else [f"covariate_{i + 1}" for i in range(n_cov)]
        )
        if len(term_names) != design.shape[1]:
            raise ValueError("names must have one entry per covariate")

        point, *_ = np.linalg.lstsq(design, theta, rcond=None)      # (p+1, K)
        rng = np.random.default_rng(seed)
        draws = np.empty((n_bootstrap, design.shape[1], theta.shape[1]), dtype=np.float64)
        for b in range(n_bootstrap):
            idx = rng.integers(0, n_docs, size=n_docs)
            coef, *_ = np.linalg.lstsq(design[idx], theta[idx], rcond=None)
            draws[b] = coef

        alpha = (1.0 - level) / 2.0
        low = np.quantile(draws, alpha, axis=0)
        high = np.quantile(draws, 1.0 - alpha, axis=0)

        out: List[Dict[str, object]] = []
        for k in range(theta.shape[1]):
            for j, term in enumerate(term_names):
                sample = draws[:, j, k]
                p_two_sided = 2.0 * min(
                    float((sample <= 0.0).mean()), float((sample >= 0.0).mean())
                )
                out.append(
                    {
                        "topic": k + 1,
                        "term": term,
                        "coef": float(point[j, k]),
                        "ci_low": float(low[j, k]),
                        "ci_high": float(high[j, k]),
                        "p": min(p_two_sided, 1.0),
                        "level": level,
                        "n_bootstrap": int(n_bootstrap),
                    }
                )
        return out

    # ------------------------------------------------------------------
    # Temporal structure
    # ------------------------------------------------------------------

    def _slice_labels(self, timestamps: Sequence[float], n_slices: int) -> np.ndarray:
        ts = np.asarray(timestamps, dtype=np.float64).ravel()
        if ts.shape[0] != self.theta.shape[0]:
            raise ValueError(
                f"timestamps has {ts.shape[0]} entries but there are "
                f"{self.theta.shape[0]} rows; with apply_chunking=True a row is a "
                "segment, so pass one timestamp per segment."
            )
        edges = np.quantile(ts, np.linspace(0.0, 1.0, n_slices + 1))[1:-1]
        return np.clip(np.searchsorted(edges, ts), 0, n_slices - 1)

    def temporal_prevalence(
        self,
        timestamps: Sequence[float],
        n_slices: int = 5,
        level: float = 0.95,
        seed: int = 0,
    ) -> List[Dict[str, object]]:
        """
        Topic prevalence per time slice, with bootstrap CIs and a trend slope.

        The topic solution is held fixed, so this measures how the *mixture* moves
        over time — not how the topics themselves change (that is
        :meth:`temporal_drift`).
        """
        self._check_fitted()
        theta = self.theta
        labels = self._slice_labels(timestamps, n_slices)
        out: List[Dict[str, object]] = []
        for s in range(n_slices):
            mask = labels == s
            if not mask.any():
                continue
            for k in range(theta.shape[1]):
                est = bootstrap_ci(theta[mask, k], n_resamples=2_000, level=level, seed=seed)
                out.append(
                    {
                        "slice": s,
                        "topic": k + 1,
                        "n": int(mask.sum()),
                        "mean": est.estimate,
                        "ci_low": est.low,
                        "ci_high": est.high,
                    }
                )
        return out

    def temporal_drift(
        self,
        timestamps: Sequence[float],
        n_slices: int = 4,
        seed: int = 0,
    ) -> List[Dict[str, object]]:
        """
        Refit the topics on each time slice and measure how far they moved.

        Each slice is factorized independently on the **shared vocabulary and
        graphs**, its topics are aligned to the first slice by optimal assignment,
        and the reported ``drift`` is ``1 - mean matched cosine`` (0 = the topics
        are identical, 1 = orthogonal).  Requires one row per document, i.e.
        ``apply_chunking=False``.
        """
        self._check_fitted()
        if self.config.apply_chunking:
            raise ValueError(
                "temporal_drift needs one row per document; set apply_chunking=False."
            )
        cfg = self.config
        X = self.parsed.bow
        labels = self._slice_labels(timestamps, n_slices)

        reference: Optional[np.ndarray] = None
        out: List[Dict[str, object]] = []
        for s in range(n_slices):
            mask = labels == s
            if int(mask.sum()) < cfg.num_topics:
                continue
            res = GraphRegularizedNMF(
                n_components=cfg.num_topics,
                lambda_syntax=cfg.lambda_syntax,
                lambda_embed=cfg.lambda_embed,
                max_iter=cfg.nmf_max_iter,
                tol=cfg.nmf_tol,
                random_state=cfg.random_state,
                divergence=cfg.divergence,
            ).fit_transform(X[mask], self.syntax_graph, self.embed_graph)
            if reference is None:
                reference = res.H
                drift, stability = 0.0, 1.0
            else:
                _, sims = self._match_topics(reference, res.H)
                stability = float(sims.mean())
                drift = float(1.0 - stability)
            out.append(
                {
                    "slice": s,
                    "n": int(mask.sum()),
                    "n_iter": res.n_iter,
                    "objective": res.objective,
                    "stability_vs_first": stability,
                    "drift": drift,
                }
            )
        return out

    # ------------------------------------------------------------------
    # Topic selection
    # ------------------------------------------------------------------

    @classmethod
    def search_k(
        cls,
        docs: Sequence[str],
        k_grid: Sequence[int] = (5, 10, 15, 20, 25, 30),
        cache_path: Optional[Union[str, Path]] = None,
        finetune: bool = True,
        verbose: bool = True,
        **kwargs,
    ):
        """
        Grid search for the optimal number of topics K.

        Same criterion as the neural package — NPMI × diversity — with an
        optional refinement between the neighbours of the best grid point.

        The parse cache is built once and reused for every K, which is what
        makes this cheap: after the first fit each additional K only re-runs the
        NMF factorisation.

        Parameters
        ----------
        docs : sequence of str
        k_grid : sequence of int, default (5, 10, 15, 20, 25, 30)
        cache_path : str | Path | None
            Parse cache; strongly recommended (shared across K and across runs).
        finetune : bool
            Refine between the grid neighbours of the best K.
        verbose : bool
            Print the score table.
        **kwargs
            :class:`BowConfig` parameters (``lang``, ``parser_backend``, …).

        Returns
        -------
        best_k : int
        best_model : BowSCPTM
        results : dict
            ``{K: {"model", "npmi", "div", "score"}}``
        """
        from .selection import search_k as _search_k

        return _search_k(
            docs=docs,
            k_grid=k_grid,
            cache_path=str(cache_path) if cache_path is not None else None,
            finetune=finetune,
            verbose=verbose,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def plot_semantic_2d(
        self,
        top_k: int = 12,
        save_path: Union[str, Path] = "semantic_space_2d.png",
        **kwargs,
    ) -> Path:
        """
        2D UMAP map of the word/topic semantic space, as a PNG for papers.

        Topics are stars, their top words and MWEs are labelled dots, and each
        topic is linked to its own words.  Requires word vectors and the
        ``[plots]`` extra (matplotlib + umap-learn).
        """
        from .visualize import plot_semantic_2d

        return plot_semantic_2d(self, top_k=top_k, save_path=save_path, **kwargs)

    def visualize_2d(
        self,
        top_k: int = 12,
        save_path: Union[str, Path] = "semantic_space_2d.png",
        **kwargs,
    ) -> Path:
        """Alias of :meth:`plot_semantic_2d`, matching the neural package's name."""
        return self.plot_semantic_2d(top_k=top_k, save_path=save_path, **kwargs)

    def topic_vectors(self) -> np.ndarray:
        """Topic centroids in word-vector space, shape (K, D)."""
        from .visualize import topic_vectors

        self._check_fitted()
        return topic_vectors(self)
