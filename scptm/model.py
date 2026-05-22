"""
scptm/model.py
--------------
Main SCPTM class with scikit-learn compatible API.

Key additions vs original script:
  - fit() / transform() / fit_transform() interface
  - save() / load() with full state persistence
  - transform() for out-of-sample documents
  - Iterative refinement (optional) — pulls doc embeddings toward topic centroids
  - run_ablation_study() class method
"""

import pickle
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .config import GRAPH_MODES, SCPTMConfig
from .evaluation import SCPTMEvaluator, interpret_mc_uncertainty
from .graph import (
    build_hetero_graph,
    collect_contextual_embeddings,
    estimate_graph_memory,
    load_ctx_embs_from_cache,
    prepare_corpus,
    save_ctx_embs_to_cache,
)
from .keywords import extract_separated_topics, extract_top_words
from .nlp import setup_nlp_pipeline
from .nn import VariationalGraphTopicModel
from .training import train
from .visualization import (
    plot_training_history,
    view_semantic_2d_paper,
    view_semantic_constellations_3d,
)


class SCPTM:
    """
    Structural Contextual Probabilistic Topic Model.

    A VAE-based topic model that leverages a heterogeneous GNN over syntactic
    dependency graphs and contextual word embeddings.

    Parameters
    ----------
    config : SCPTMConfig | None
        Model configuration. Defaults to SCPTMConfig() with sensible defaults.

    Examples
    --------
    >>> model = SCPTM()
    >>> labels = model.fit_transform(documents)
    >>> print(model.get_topic_info())

    >>> # Out-of-sample inference
    >>> new_theta = model.transform(new_docs)
    """

    def __init__(self, config: Optional[SCPTMConfig] = None, **kwargs):
        self.config = config or SCPTMConfig(**kwargs)
        self._is_fitted: bool = False

        # State filled during fit()
        self._sbert   = None
        self._nlp     = None
        self._stop    = None
        self._nn: Optional[VariationalGraphTopicModel] = None
        self._graph_data = None
        self._static_word_embs: Optional[torch.Tensor] = None
        self._ctx_embs_list: Optional[list] = None
        self._vocab: Optional[List[str]] = None
        self._vocab_idx: Optional[dict] = None
        self._bow_sparse = None
        self._corpus: Optional[List[str]] = None
        self._theta: Optional[torch.Tensor] = None
        self._theta_uncertainty: Optional[torch.Tensor] = None
        self._history: Optional[dict] = None
        self._device: Optional[torch.device] = None
        self._topics_dict: Optional[dict] = None

    # ------------------------------------------------------------------
    # scikit-learn compatible API
    # ------------------------------------------------------------------

    def fit(
        self,
        source,
        source_type: str = "list",
        text_col: Optional[str] = None,
        iterative_refinement: bool = False,
        n_refinement_steps: int = 2,
        refinement_blend: float = 0.2,
        edge_cache_path: Optional[str] = None,
    ) -> "SCPTM":
        """
        Fit SCPTM on a corpus.

        Parameters
        ----------
        source : list[str] | str (folder path) | pd.DataFrame
        source_type : "list" | "folder" | "dataframe"
        text_col : required when source_type == "dataframe"
        iterative_refinement : bool
            If True, alternate between training and blending doc embeddings
            toward topic centroids (TriTopic-inspired).
        n_refinement_steps : int
            Number of refine → retrain cycles.
        refinement_blend : float
            Alpha in:  emb_refined = (1-alpha)*emb_orig + alpha*centroid
        edge_cache_path : str | None
            Path to a parse cache file. If the file exists the spaCy parsing
            steps are skipped on load. If it doesn't exist, it is created
            after the first parse so subsequent runs are fast.

        Returns
        -------
        self
        """
        cfg = self.config

        # ---- 1. NLP pipeline ----
        self._sbert, self._nlp, self._stop = setup_nlp_pipeline(cfg.lang)

        # ---- 2. Corpus ----
        self._corpus = prepare_corpus(
            source,
            source_type=source_type,
            text_col=text_col,
            apply_chunking=cfg.apply_chunking,
            max_chunk_chars=cfg.max_chunk_chars,
        )

        # ---- 3. Graph ----
        self._graph_data, self._vocab, self._bow_sparse, n_dw, n_ww = build_hetero_graph(
            self._corpus, self._sbert, self._nlp, self._stop, cfg,
            edge_cache_path=edge_cache_path,
        )
        self._vocab_idx = {w: i for i, w in enumerate(self._vocab)}
        emb_dim = self._graph_data["doc"].x.shape[1]
        estimate_graph_memory(len(self._corpus), len(self._vocab), n_dw, n_ww, emb_dim)

        # ---- 4. Device ----
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._graph_data = self._graph_data.to(self._device)
        self._static_word_embs = self._graph_data["word"].x.clone().to(self._device)

        # ---- 5. Contextual embeddings (with cache support) ----
        self._ctx_embs_list = None
        if edge_cache_path is not None:
            self._ctx_embs_list = load_ctx_embs_from_cache(
                edge_cache_path, len(self._vocab)
            )
        if self._ctx_embs_list is None:
            _ctx_dict = collect_contextual_embeddings(
                self._corpus, self._nlp, self._sbert, self._vocab_idx,
                cfg.max_ctx_occurrences,
            )
            self._ctx_embs_list = [
                _ctx_dict[w].to("cpu") if w in _ctx_dict else None
                for w in self._vocab
            ]
            if edge_cache_path is not None:
                save_ctx_embs_to_cache(edge_cache_path, self._ctx_embs_list)

        # ---- 6. Model ----
        self._nn = VariationalGraphTopicModel(
            emb_dim, cfg.hidden_channels, cfg.num_topics,
            len(self._vocab), graph_mode=cfg.graph_mode,
            beta_temperature=cfg.beta_temperature,
        ).to(self._device)

        # ---- 6b. K-means initialisation of topic embeddings ----
        # Topic embeddings live in the SAME space as word embeddings (SBERT
        # single-word encodings).  Clustering WORD embeddings (not document
        # embeddings) guarantees high cosine similarity between each topic
        # centroid and its cluster members, so beta is discriminative from
        # epoch 1 and the reconstruction gradient flows properly.
        #
        # Document-centroid init looks intuitive but fails: a centroid of
        # ~900 sentence embeddings has cosine sim only ~0.15 with individual
        # word embeddings → logit ≈ 1.5 at T=0.1 → beta still near-uniform.
        try:
            from sklearn.cluster import MiniBatchKMeans

            word_embs_np = self._static_word_embs.cpu().numpy()  # (V, D)
            kmeans = MiniBatchKMeans(
                n_clusters=cfg.num_topics,
                random_state=cfg.random_state,
                n_init=5,
                max_iter=300,
            ).fit(word_embs_np)
            centers = F.normalize(
                torch.tensor(kmeans.cluster_centers_, dtype=torch.float32),
                p=2, dim=-1,
            ).to(self._device)
            with torch.no_grad():
                self._nn.topic_embeddings.data.copy_(centers)

            # Quick quality check: mean cosine sim of each centroid vs its cluster
            sims = torch.matmul(centers, F.normalize(self._static_word_embs, p=2, dim=-1).T)
            mean_sim = sims.max(dim=0).values.mean().item()
            print(
                f"  Topic embeddings initialised from word-embedding k-means "
                f"(mean top-sim per word: {mean_sim:.3f})"
            )
        except Exception as e:  # pragma: no cover
            print(f"  [WARN] K-means init failed ({e}); using random init.")

        # ---- 7. Training (+ optional iterative refinement) ----
        if iterative_refinement and n_refinement_steps > 0:
            self._history = self._fit_with_refinement(
                n_refinement_steps, refinement_blend
            )
        else:
            self._history = train(
                self._graph_data, self._bow_sparse, self._nn,
                self._ctx_embs_list, self._static_word_embs,
                cfg, self._device,
                vocab=self._vocab,
            )

        # ---- 8. Final inference ----
        self._theta, self._theta_uncertainty = self._infer_all()
        self._is_fitted = True
        return self

    def transform(self, documents: List[str]) -> torch.Tensor:
        """
        Infer topic mixtures for new (out-of-sample) documents.

        Strategy: encode new documents with SBERT, build a local doc-only
        feature tensor, run through the trained encoder MLP/GNN using the
        original word embeddings as context, return theta.

        For graph modes this uses the full graph plus new document nodes
        appended to the doc feature matrix (no new edge construction —
        approximation suitable for production use).

        Parameters
        ----------
        documents : list of strings

        Returns
        -------
        theta : torch.Tensor, shape (len(documents), num_topics)
        """
        self._check_fitted()
        cfg = self.config

        # Encode new docs
        new_doc_embs = torch.tensor(
            self._sbert.encode(documents, show_progress_bar=False),
            dtype=torch.float32,
        ).to(self._device)

        self._nn.eval()
        with torch.no_grad():
            if cfg.graph_mode == "none":
                # Pure MLP: only doc features needed
                mu, _ = self._nn.encode(
                    {"doc": new_doc_embs, "word": self._static_word_embs},
                    {},
                )
            else:
                # Append new docs to existing graph features, use existing edges
                # (approximation: new docs have no edges, attend over word nodes)
                n_orig = self._graph_data["doc"].x.shape[0]
                extended_doc = torch.cat(
                    [self._graph_data["doc"].x, new_doc_embs], dim=0
                )
                x_dict_ext = {
                    "doc":  extended_doc,
                    "word": self._graph_data["word"].x,
                }
                mu_all, _ = self._nn.encode(x_dict_ext, self._graph_data.edge_index_dict)
                mu = mu_all[n_orig:]   # slice out the new docs

            theta = F.softmax(mu, dim=-1)

        return theta.cpu()

    def fit_transform(
        self,
        source,
        source_type: str = "list",
        text_col: Optional[str] = None,
        edge_cache_path: Optional[str] = None,
        **fit_kwargs,
    ) -> torch.Tensor:
        """Fit and return topic mixtures for training documents."""
        self.fit(
            source, source_type=source_type, text_col=text_col,
            edge_cache_path=edge_cache_path, **fit_kwargs,
        )
        return self._theta

    # ------------------------------------------------------------------
    # Iterative refinement
    # ------------------------------------------------------------------

    def _fit_with_refinement(
        self,
        n_steps: int,
        blend: float,
    ) -> dict:
        """
        TriTopic-inspired iterative refinement:
          1. Train for cfg.epochs
          2. Blend doc embeddings toward dominant topic centroid
          3. Re-train from current weights for cfg.epochs
          4. Repeat n_steps times
        """
        cfg = self.config
        history = {"loss": [], "recon": [], "kl": [], "kl_weight": [],
                   "coherence_npmi": [], "topic_diversity": [], "metric_epochs": []}

        for step in range(n_steps):
            print(f"\n[Iterative Refinement] Step {step+1}/{n_steps}")
            h = train(
                self._graph_data, self._bow_sparse, self._nn,
                self._ctx_embs_list, self._static_word_embs,
                cfg, self._device,
                vocab=self._vocab,
            )
            # Merge history
            for k in ["loss", "recon", "kl", "kl_weight"]:
                history[k].extend(h[k])
            for k in ["coherence_npmi", "topic_diversity"]:
                history[k].extend(h.get(k, []))
            offset = step * cfg.epochs
            history["metric_epochs"].extend(
                [e + offset for e in h.get("metric_epochs", [])]
            )

            if step < n_steps - 1:
                self._refine_embeddings(blend)

        return history

    def _refine_embeddings(self, blend: float):
        """
        Blend document embeddings toward their dominant topic centroid.
        emb_new = (1-blend) * emb_orig + blend * centroid
        """
        self._nn.eval()
        with torch.no_grad():
            mu, _ = self._nn.encode(
                self._graph_data.x_dict, self._graph_data.edge_index_dict
            )
            theta = F.softmax(mu, dim=-1)                          # (N, K)
            dominant = theta.argmax(dim=-1)                        # (N,)
            topic_embs_norm = F.normalize(
                self._nn.topic_embeddings, p=2, dim=-1
            )                                                       # (K, D)
            # Build centroid matrix: for each doc, its dominant topic embedding
            doc_centroids = topic_embs_norm[dominant]              # (N, D)
            orig = self._graph_data["doc"].x
            refined = (1 - blend) * orig + blend * doc_centroids
            refined = F.normalize(refined, p=2, dim=-1)
            self._graph_data["doc"].x = refined
        print(f"  Embeddings refined (blend={blend:.2f})")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _infer_all(self):
        """Run final inference over all training documents."""
        cfg = self.config
        self._nn.eval()

        # Ensure beta is fresh
        self._nn.compute_contextual_beta(self._ctx_embs_list, self._static_word_embs)

        with torch.no_grad():
            mu, logvar = self._nn.encode(
                self._graph_data.x_dict, self._graph_data.edge_index_dict
            )
            theta_map = F.softmax(mu, dim=-1)

            if cfg.n_mc_samples > 1:
                mc = []
                for _ in range(cfg.n_mc_samples):
                    z = self._nn.reparameterize(mu, logvar)
                    mc.append(F.softmax(z, dim=-1).unsqueeze(0))
                mc_stack = torch.cat(mc, dim=0)
                theta_mean = mc_stack.mean(dim=0)
                theta_std  = mc_stack.std(dim=0)
                print(f"MC uncertainty (S={cfg.n_mc_samples}): "
                      f"mean std = {theta_std.mean().item():.4f}")
                return theta_mean.cpu(), theta_std.cpu()
            else:
                return theta_map.cpu(), None

    # ------------------------------------------------------------------
    # Results helpers
    # ------------------------------------------------------------------

    def get_topic_info(self, top_k: int = 10, method: Optional[str] = None) -> pd.DataFrame:
        """
        Return a DataFrame with topic index, top words, and dominant doc count.

        Parameters
        ----------
        top_k : int
        method : "cosine" | "ctfidf" | None
            Override config.keyword_method for this call.
        """
        self._check_fitted()
        kw_method = method or self.config.keyword_method
        top_words = extract_top_words(
            self._nn, self._vocab, top_k=top_k,
            method=kw_method,
            bow_sparse=self._bow_sparse if kw_method == "ctfidf" else None,
            theta=self._theta if kw_method == "ctfidf" else None,
        )
        dominant  = self._theta.argmax(dim=-1).numpy()
        rows = []
        for k, words in enumerate(top_words):
            n_docs = int((dominant == k).sum())
            rows.append({
                "topic_id":  k + 1,
                "size":      n_docs,
                "keywords":  ", ".join(words),
            })
        df = pd.DataFrame(rows).sort_values("size", ascending=False).reset_index(drop=True)
        print(df.to_string(index=False))
        return df

    def get_topics_dict(self, top_k: int = 5, method: Optional[str] = None) -> dict:
        """
        Return separated single-word + phrase topic keywords.

        Parameters
        ----------
        top_k : int
        method : "cosine" | "ctfidf" | None
            Override config.keyword_method for this call.
        """
        self._check_fitted()
        kw_method = method or self.config.keyword_method
        # Cache is keyed on (top_k, method) so different calls don't collide
        cache_key = (top_k, kw_method)
        if self._topics_dict is None or getattr(self, "_topics_dict_key", None) != cache_key:
            self._topics_dict, _, _ = extract_separated_topics(
                self._corpus, self._nn, self._vocab,
                self._static_word_embs, self._sbert,
                self._stop, top_k=top_k,
                method=kw_method,
                bow_sparse=self._bow_sparse if kw_method == "ctfidf" else None,
                theta=self._theta if kw_method == "ctfidf" else None,
            )
            self._topics_dict_key = cache_key
        return self._topics_dict

    def get_document_topics(self) -> pd.DataFrame:
        """Return per-document dominant topic and probability."""
        self._check_fitted()
        dominant = self._theta.argmax(dim=-1).numpy()
        max_prob = self._theta.max(dim=-1).values.numpy()
        return pd.DataFrame({
            "doc_id":          range(len(self._corpus)),
            "dominant_topic":  dominant + 1,
            "dominant_prob":   np.round(max_prob, 3),
            "text_preview":    [
                t[:80] + "..." if len(t) > 80 else t
                for t in self._corpus
            ],
        })

    def get_uncertainty_report(self) -> pd.DataFrame:
        """Return MC uncertainty classification (requires n_mc_samples > 1)."""
        self._check_fitted()
        if self._theta_uncertainty is None:
            raise RuntimeError(
                "Run with config.n_mc_samples > 1 to enable uncertainty reports."
            )
        topic_names = [f"Topic_{k+1}" for k in range(self.config.num_topics)]
        return interpret_mc_uncertainty(
            self._theta, self._theta_uncertainty,
            self._corpus, topic_names,
        )

    def evaluate(
        self,
        true_labels: Optional[np.ndarray] = None,
        theta_runs: Optional[list] = None,
    ) -> dict:
        """Run evaluation metrics. Optionally pass ground-truth labels."""
        self._check_fitted()
        evaluator = SCPTMEvaluator(
            self._nn, self._vocab, self._bow_sparse, self._corpus
        )
        return evaluator.evaluate(
            self._theta,
            true_labels=true_labels,
            theta_uncertainty=self._theta_uncertainty,
            theta_runs=theta_runs,
        )

    # ------------------------------------------------------------------
    # Visualisations
    # ------------------------------------------------------------------

    def plot_training(self, save_path: str = "training_history.png"):
        """Plot training loss curves and metrics."""
        self._check_fitted()
        plot_training_history(self._history, save_path=save_path)

    def visualize_3d(self):
        """Interactive 3D Plotly semantic constellation."""
        self._check_fitted()
        td, mwe_embs, mwe_vocab = extract_separated_topics(
            self._corpus, self._nn, self._vocab,
            self._static_word_embs, self._sbert, self._stop,
        )
        view_semantic_constellations_3d(
            self._nn, self._static_word_embs, self._vocab,
            mwe_embs, mwe_vocab, td,
            random_state=self.config.random_state,
        )

    def visualize_2d(self, save_path: str = "semantic_space_2d.png"):
        """High-resolution 2D UMAP plot for publications."""
        self._check_fitted()
        td, mwe_embs, mwe_vocab = extract_separated_topics(
            self._corpus, self._nn, self._vocab,
            self._static_word_embs, self._sbert, self._stop,
        )
        view_semantic_2d_paper(
            self._nn, self._static_word_embs, self._vocab,
            mwe_embs, mwe_vocab, td,
            save_path=save_path,
            random_state=self.config.random_state,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]):
        """Pickle the full model state to disk."""
        self._check_fitted()
        state = {
            "config":           self.config,
            "nn_state_dict":    self._nn.state_dict(),
            "vocab":            self._vocab,
            "bow_sparse":       self._bow_sparse,
            "corpus":           self._corpus,
            "theta":            self._theta,
            "theta_uncertainty":self._theta_uncertainty,
            "history":          self._history,
            "cached_beta":      self._nn._cached_beta,
            "static_word_embs": self._static_word_embs.cpu(),
            "ctx_embs_list":    self._ctx_embs_list,
            "graph_doc_x":      self._graph_data["doc"].x.cpu(),
            "graph_word_x":     self._graph_data["word"].x.cpu(),
            # Edge indices — persisted so loaded models can run full GNN inference
            "edge_dw":  self._graph_data["doc", "contains", "word"].edge_index.cpu(),
            "edge_wd":  self._graph_data["word", "rev_contains", "doc"].edge_index.cpu(),
            "edge_ww":  self._graph_data["word", "relates", "word"].edge_index.cpu(),
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SCPTM":
        """Load a saved SCPTM model."""
        with open(path, "rb") as f:
            state = pickle.load(f)

        model = cls(config=state["config"])
        model._sbert, model._nlp, model._stop = setup_nlp_pipeline(
            state["config"].lang, verbose=False
        )
        model._vocab      = state["vocab"]
        model._vocab_idx  = {w: i for i, w in enumerate(state["vocab"])}
        model._bow_sparse = state["bow_sparse"]
        model._corpus     = state["corpus"]
        model._theta      = state["theta"]
        model._theta_uncertainty = state["theta_uncertainty"]
        model._history    = state["history"]

        model._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        emb_dim       = state["graph_doc_x"].shape[1]

        model._nn = VariationalGraphTopicModel(
            emb_dim,
            state["config"].hidden_channels,
            state["config"].num_topics,
            len(state["vocab"]),
            graph_mode=state["config"].graph_mode,
            beta_temperature=getattr(state["config"], "beta_temperature", 0.1),
        ).to(model._device)
        model._nn.load_state_dict(state["nn_state_dict"])
        model._nn._cached_beta = (
            state["cached_beta"].to(model._device)
            if state["cached_beta"] is not None
            else None
        )

        model._static_word_embs = state["static_word_embs"].to(model._device)
        model._ctx_embs_list    = state["ctx_embs_list"]

        # Rebuild graph data for transform() — restore edge indices if available
        # (old pickles without edge keys fall back to empty tensors gracefully)
        from torch_geometric.data import HeteroData
        _empty = torch.empty((2, 0), dtype=torch.long)
        gd = HeteroData()
        gd["doc"].x  = state["graph_doc_x"].to(model._device)
        gd["word"].x = state["graph_word_x"].to(model._device)
        gd["doc", "contains", "word"].edge_index     = state.get("edge_dw", _empty).to(model._device)
        gd["word", "rev_contains", "doc"].edge_index = state.get("edge_wd", _empty).to(model._device)
        gd["word", "relates", "word"].edge_index     = state.get("edge_ww", _empty).to(model._device)
        model._graph_data = gd

        model._is_fitted = True
        print(f"Model loaded from {path}")
        return model

    # ------------------------------------------------------------------
    # Ablation study
    # ------------------------------------------------------------------

    @classmethod
    def run_ablation_study(
        cls,
        documents: List[str],
        epochs: int = 50,
        **base_kwargs,
    ) -> pd.DataFrame:
        """
        Train one SCPTM per graph_mode and compare metrics.

        Parameters
        ----------
        documents : list of strings
        epochs : int
        **base_kwargs : additional SCPTMConfig parameters

        Returns
        -------
        pd.DataFrame with one row per mode.
        """
        results = []
        for mode in GRAPH_MODES:
            print(f"\n{'='*60}\nABLATION — mode: {mode}\n{'='*60}")
            cfg = SCPTMConfig(graph_mode=mode, epochs=epochs, n_mc_samples=1, **base_kwargs)
            m   = cls(config=cfg)
            m.fit(documents, source_type="list")
            metrics = m.evaluate()
            row = {
                "graph_mode":       mode,
                "description":      GRAPH_MODES[mode],
                "final_loss":       round(m._history["loss"][-1], 3),
                "npmi_coherence":   metrics.get("npmi_coherence", float("nan")),
                "topic_diversity":  metrics.get("topic_diversity", float("nan")),
            }
            results.append(row)
            print(f"  NPMI={row['npmi_coherence']} | Div={row['topic_diversity']}")

        df = pd.DataFrame(results)
        print("\n[Ablation Study Results]")
        print(df.to_string(index=False))
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Call fit() before using this method.")

    @property
    def topic_embeddings(self) -> torch.Tensor:
        """Learnable topic embedding matrix, shape (K, D)."""
        self._check_fitted()
        return self._nn.topic_embeddings.detach().cpu()

    @property
    def theta(self) -> torch.Tensor:
        """Document-topic mixtures, shape (n_docs, K)."""
        self._check_fitted()
        return self._theta

    @property
    def vocab(self) -> List[str]:
        """Vocabulary list."""
        self._check_fitted()
        return self._vocab
