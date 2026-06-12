"""
scptm/nn.py
-----------
Neural network components for SCPTM.

Classes
-------
VariationalGraphEncoder
    Encodes document embeddings to (μ, logσ²) in topic space.
    Graph modes use a 1-layer HeteroConv/GAT (2 heads).
    Mode 'none' uses a 2-layer MLP over document features only.

VariationalGraphTopicModel
    Full VAE-GNN topic model.  Key methods:

    encode(x_dict, edge_index_dict) → (μ, logσ²)
    reparameterize(μ, logσ²) → z
    decode_train(θ, word_embs) → recon_probs
        Differentiable path used during training.  Beta is computed
        on-the-fly as softmax(topic_norm @ word_norm.T / T) so gradients
        flow all the way back to topic_embeddings.
    decode(θ) → recon_probs
        Cached contextual beta path used at evaluation time.
    compute_contextual_beta(ctx_embs_list, static_word_embs) → beta
        Attention-pooled contextual embeddings → (K, V) beta matrix.
    topic_diversity_loss() → scalar
        Mean pairwise cosine similarity between topic embeddings
        (minimise to push topics apart).

Design notes
------------
* Beta temperature (default T=0.1): cosine similarities in R^384 concentrate
  near 0 (std ≈ 1/√384 ≈ 0.051).  Dividing by T maps the range to ≈[−10,+10]
  which produces a peaked, discriminative softmax and non-zero gradients.
  Without this, reconstruction loss is stuck at log(V) forever.

* Topic embedding init: topic_embeddings are initialised from k-means
  centroids of the *word* embedding space in model.fit() (not document
  embeddings).  This guarantees high cosine similarity between each topic
  and its vocabulary cluster from epoch 1.

* invalidate_beta() / _beta_dirty flag: beta is marked stale after every
  optimizer step and lazily recomputed before evaluation, ensuring
  encoder and decoder are always in sync.
"""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv


class VariationalGraphEncoder(nn.Module):
    """
    Encoder: document embeddings -> (mu, logvar) in topic space.

    In graph modes the encoder is a 1-layer HeteroConv (GAT, 2 heads).
    In 'none' mode the encoder is a 2-layer MLP over document features only.

    Parameters
    ----------
    in_channels : int
        Input embedding dimension (SBERT output size).
    hidden_channels : int
        Hidden units per GAT head (total output = hidden_channels * 2).
    num_topics : int
        Dimensionality of the latent topic space.
    graph_mode : str
        One of {"none", "no_syntax", "full_dep", "filtered"}.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_topics: int,
        graph_mode: str = "filtered",
        encoder_residual: bool = True,
    ):
        super().__init__()
        self.graph_mode = graph_mode
        self.encoder_residual = encoder_residual
        gat_out = hidden_channels * 2   # 2 heads → concatenated

        if graph_mode == "none":
            # Pure MLP: in_channels -> hidden -> gat_out
            self.mlp = nn.Sequential(
                nn.Linear(in_channels, hidden_channels * 2),
                nn.LayerNorm(hidden_channels * 2),
                nn.ReLU(),
                nn.Linear(hidden_channels * 2, gat_out),
                nn.ReLU(),
            )
        else:
            # NOTE: word→doc (rev_contains) removed — it propagated word
            # representations back into doc nodes, causing over-smoothing
            # and degrading NMI. Doc nodes are updated via the residual
            # connection below instead.
            # Suppress the PyG warning about doc nodes not being updated
            # by message passing — intentional, handled by residual_proj.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", "There exist node types"
                )
                self.conv = HeteroConv(
                    {
                        ("doc",  "contains", "word"): GATConv(
                            (in_channels, in_channels), hidden_channels,
                            heads=2, add_self_loops=False
                        ),
                        ("word", "relates",  "word"): GATConv(
                            in_channels, hidden_channels,
                            heads=2, add_self_loops=False
                        ),
                    },
                    aggr="mean",
                )
            # Residual projection: maps raw doc input to gat_out dim so it
            # can be added to the GAT output, preserving document identity.
            if encoder_residual:
                self.residual_proj = nn.Linear(in_channels, gat_out, bias=False)

        self.mu_layer     = nn.Linear(gat_out, num_topics)
        self.logvar_layer = nn.Linear(gat_out, num_topics)

    def forward(self, x_dict: dict, edge_index_dict: dict):
        """
        Returns
        -------
        mu, logvar : torch.Tensor, shape (n_docs, num_topics)
        """
        if self.graph_mode == "none":
            h = self.mlp(x_dict["doc"])
        else:
            h_dict = self.conv(x_dict, edge_index_dict)
            h_dict = {k: F.leaky_relu(v) for k, v in h_dict.items()}
            # After removing rev_contains, doc nodes are no longer updated by
            # HeteroConv. Use the residual projection of the raw doc input as
            # the doc representation, optionally adding word-side context via
            # any future doc-targeting relation.
            doc_input = x_dict["doc"]
            if self.encoder_residual:
                h = self.residual_proj(doc_input)
                # Add word→doc signal if present (e.g. from future relations)
                if "doc" in h_dict:
                    h = h + h_dict["doc"]
            else:
                h = h_dict.get("doc", self.residual_proj(doc_input)
                               if hasattr(self, "residual_proj") else doc_input)
        return self.mu_layer(h), self.logvar_layer(h)


class VariationalGraphTopicModel(nn.Module):
    """
    Full VAE-GNN topic model.

    Parameters
    ----------
    in_channels : int
        SBERT embedding dimension.
    hidden_channels : int
        GNN hidden size per head.
    num_topics : int
        Number of topics K.
    vocab_size : int
        Vocabulary size V.
    graph_mode : str
        Graph construction mode.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_topics: int,
        vocab_size: int,
        graph_mode: str = "filtered",
        beta_temperature: float = 0.1,
        trainable_word_embeddings: bool = True,
        encoder_residual: bool = True,
    ):
        super().__init__()
        self.num_topics                = num_topics
        self.vocab_size                = vocab_size
        self.graph_mode                = graph_mode
        self.beta_temperature          = beta_temperature
        self.trainable_word_embeddings = trainable_word_embeddings

        self.encoder = VariationalGraphEncoder(
            in_channels, hidden_channels, num_topics, graph_mode,
            encoder_residual=encoder_residual,
        )

        # Learnable topic vectors in embedding space.
        # Initialised with unit-norm to improve early convergence.
        raw = torch.randn(num_topics, in_channels)
        self.topic_embeddings = nn.Parameter(
            F.normalize(raw, p=2, dim=-1)
        )

        # Trainable word embeddings for the decoder.
        # When True, initialised from SBERT in model.fit() and updated
        # end-to-end via the reconstruction loss.  This allows β to adapt
        # to corpus co-occurrence statistics, significantly improving NPMI.
        # When False, decode_train() receives static SBERT embeddings instead.
        if trainable_word_embeddings:
            self.word_embeddings = nn.Parameter(torch.zeros(vocab_size, in_channels))
        else:
            self.word_embeddings = None  # type: ignore[assignment]

        # [FIX-2] beta is always None; computed on demand before decode
        self._cached_beta: torch.Tensor | None = None
        self._beta_dirty: bool = True

    # ------------------------------------------------------------------
    # Reparameterisation
    # ------------------------------------------------------------------

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Standard Gaussian reparameterisation."""
        if self.training:
            return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return mu

    # ------------------------------------------------------------------
    # Contextual beta computation
    # ------------------------------------------------------------------

    def compute_contextual_beta(
        self,
        ctx_embs_list: list,
        static_word_embs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute beta matrix (K x V) using attention-pooled contextual word
        embeddings.  Called explicitly before training and refreshed periodically.

        For words with no contextual embeddings, falls back to static SBERT vector.

        Parameters
        ----------
        ctx_embs_list : list[Tensor | None]
            One entry per vocabulary word; each is shape (N_ctx, emb_dim) or None.
        static_word_embs : torch.Tensor
            Shape (V, emb_dim) — static SBERT word embeddings.

        Returns
        -------
        beta : torch.Tensor, shape (K, V), softmax-normalised over V.
        """
        device = self.topic_embeddings.device
        V = self.vocab_size
        K = self.num_topics
        beta_matrix = torch.zeros(K, V, device=device)

        with torch.no_grad():
            topic_embs = self.topic_embeddings   # (K, D)
            for w_idx, ctx_vecs in enumerate(ctx_embs_list):
                if ctx_vecs is None or len(ctx_vecs) == 0:
                    w_static = static_word_embs[w_idx].to(device)
                    sims = F.cosine_similarity(topic_embs, w_static.unsqueeze(0))
                else:
                    ctx = ctx_vecs.to(device)                    # (N, D)
                    attn_logits = torch.matmul(topic_embs, ctx.T) # (K, N)
                    attn = F.softmax(attn_logits, dim=-1)
                    repr_kw = torch.matmul(attn, ctx)            # (K, D)
                    sims = F.cosine_similarity(topic_embs, repr_kw, dim=-1)
                beta_matrix[:, w_idx] = sims

        self._cached_beta = F.softmax(beta_matrix / self.beta_temperature, dim=-1)
        self._beta_dirty = False
        return self._cached_beta

    def invalidate_beta(self):
        """Mark beta as dirty after a parameter update."""
        self._beta_dirty = True

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def encode(self, x_dict: dict, edge_index_dict: dict):
        """Encode documents to (mu, logvar)."""
        return self.encoder(x_dict, edge_index_dict)

    def decode_train(
        self,
        theta_d: torch.Tensor,
        word_embs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Differentiable decode used **during training**.

        Uses self.word_embeddings (trainable) when available, otherwise
        falls back to the passed static SBERT word embeddings.

        Parameters
        ----------
        theta_d : torch.Tensor, shape (B, K)
        word_embs : torch.Tensor, shape (V, D)
            Static SBERT word embeddings — used as fallback when
            trainable_word_embeddings=False, and as initialisation source.

        Returns
        -------
        recon : torch.Tensor, shape (B, V)
        """
        # Use trainable embeddings when available — they carry gradients from
        # the reconstruction loss and adapt to corpus co-occurrence statistics.
        embs = self.word_embeddings if self.word_embeddings is not None else word_embs
        topic_norm = F.normalize(self.topic_embeddings, p=2, dim=-1)  # (K, D)
        word_norm  = F.normalize(embs,                  p=2, dim=-1)  # (V, D)
        beta_live = F.softmax(
            torch.matmul(topic_norm, word_norm.T) / self.beta_temperature,
            dim=-1,
        )                                                              # (K, V)
        return torch.matmul(theta_d, beta_live)                       # (B, V)

    def decode(self, theta_d: torch.Tensor) -> torch.Tensor:
        """
        Decode using the cached contextual beta — used at **evaluation time**
        (keyword extraction, get_topic_info).  Does NOT carry gradients.

        Parameters
        ----------
        theta_d : torch.Tensor, shape (B, K)

        Returns
        -------
        recon : torch.Tensor, shape (B, V)
        """
        if self._cached_beta is None:
            raise RuntimeError(
                "Call compute_contextual_beta() before the first forward pass."
            )
        return torch.matmul(theta_d, self._cached_beta)

    # ------------------------------------------------------------------
    # [FIX-3] Topic diversity / repulsion loss
    # ------------------------------------------------------------------

    def topic_diversity_loss(self) -> torch.Tensor:
        """
        Penalise cosine similarity between topic embedding pairs.
        Encourages topics to occupy different regions of the embedding space.

        Returns a scalar loss (mean pairwise cosine similarity).
        """
        K = self.num_topics
        if K < 2:
            return torch.tensor(0.0, device=self.topic_embeddings.device)
        normed = F.normalize(self.topic_embeddings, p=2, dim=-1)   # (K, D)
        sim_matrix = torch.matmul(normed, normed.T)                 # (K, K)
        # Off-diagonal elements only
        mask = ~torch.eye(K, dtype=torch.bool, device=sim_matrix.device)
        return sim_matrix[mask].mean()

    def we_coherence_loss(self, word_embs: torch.Tensor) -> torch.Tensor:
        """
        Differentiable WE-Coherence loss.

        For each topic, computes a soft-weighted mean word embedding (using
        the current β as soft attention weights) and maximises its cosine
        similarity with the topic embedding.  This encourages the decoder to
        select words that are semantically compact around the topic centroid.

        Parameters
        ----------
        word_embs : torch.Tensor, shape (V, D)
            Used as fallback when trainable word embeddings are absent.

        Returns
        -------
        Scalar loss — negative mean WE-coherence (minimise to maximise coh.).
        """
        embs = self.word_embeddings if self.word_embeddings is not None else word_embs
        topic_norm = F.normalize(self.topic_embeddings, p=2, dim=-1)  # (K, D)
        word_norm  = F.normalize(embs, p=2, dim=-1)                   # (V, D)
        # Soft top-k selection via the same β used in decode_train
        beta_scores = torch.matmul(topic_norm, word_norm.T) / self.beta_temperature  # (K, V)
        soft_w = F.softmax(beta_scores, dim=-1)                       # (K, V)
        # Weighted mean word embedding per topic
        mean_word = torch.matmul(soft_w, word_norm)                   # (K, D)
        mean_word_norm = F.normalize(mean_word, p=2, dim=-1)
        # WE-coherence: cosine similarity between topic centroid and its
        # soft-selected word neighbourhood
        we_coh = F.cosine_similarity(topic_norm, mean_word_norm, dim=-1)  # (K,)
        return -we_coh.mean()  # minimise ≡ maximise WE-coherence

    def npmi_coherence_loss(
        self,
        ppmi_tensor: torch.Tensor,
        word_embs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Differentiable NPMI coherence loss.

        Uses a pre-computed PPMI matrix (fixed, not trained) as a reward signal:
        the loss pushes β to concentrate probability mass on word pairs that
        co-occur in the corpus, directly optimising NPMI.

        Parameters
        ----------
        ppmi_tensor : torch.Tensor, shape (V, V)
            Positive PMI matrix pre-computed from the corpus BoW.
        word_embs : torch.Tensor, shape (V, D)
            Fallback word embeddings when trainable_word_embeddings=False.

        Returns
        -------
        Scalar loss — negative mean NPMI coherence (minimise to maximise).
        """
        embs = self.word_embeddings if self.word_embeddings is not None else word_embs
        topic_norm = F.normalize(self.topic_embeddings, p=2, dim=-1)
        word_norm  = F.normalize(embs, p=2, dim=-1)
        beta_scores = torch.matmul(topic_norm, word_norm.T) / self.beta_temperature
        soft_w = F.softmax(beta_scores, dim=-1)                       # (K, V)
        # Coherence: for each topic k, sum_{i,j} β[k,i] * PPMI[i,j] * β[k,j]
        # = diag(β @ PPMI @ β.T)
        coh = torch.sum(torch.matmul(soft_w, ppmi_tensor) * soft_w, dim=-1)  # (K,)
        return -coh.mean()  # minimise ≡ maximise NPMI coherence
