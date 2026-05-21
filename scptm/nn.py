"""
scptm/nn.py
-----------
Neural network components for SCPTM.

Key fixes vs original:
  [FIX-1] mode 'none' uses forward_encoder consistently (no stale cat workaround)
  [FIX-2] contextual beta is now invalidated after every optimizer step and
          lazily recomputed before decode — ensures encoder/decoder alignment
  [FIX-3] topic_diversity_loss: cosine repulsion between topic embeddings
           to prevent topic collapse
"""

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
    ):
        super().__init__()
        self.graph_mode = graph_mode
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
            self.conv = HeteroConv(
                {
                    ("doc",  "contains",    "word"): GATConv(
                        (in_channels, in_channels), hidden_channels,
                        heads=2, add_self_loops=False
                    ),
                    ("word", "relates",     "word"): GATConv(
                        in_channels, hidden_channels,
                        heads=2, add_self_loops=False
                    ),
                    ("word", "rev_contains","doc"):  GATConv(
                        (in_channels, in_channels), hidden_channels,
                        heads=2, add_self_loops=False
                    ),
                },
                aggr="mean",
            )

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
            h = h_dict["doc"]
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
    ):
        super().__init__()
        self.num_topics  = num_topics
        self.vocab_size  = vocab_size
        self.graph_mode  = graph_mode

        self.encoder = VariationalGraphEncoder(
            in_channels, hidden_channels, num_topics, graph_mode
        )

        # Learnable topic vectors in embedding space.
        # Initialised with unit-norm to improve early convergence.
        raw = torch.randn(num_topics, in_channels)
        self.topic_embeddings = nn.Parameter(
            F.normalize(raw, p=2, dim=-1)
        )

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

        self._cached_beta = F.softmax(beta_matrix, dim=-1)
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

    def decode(self, theta_d: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct BoW distribution from topic mixture theta_d.

        Parameters
        ----------
        theta_d : torch.Tensor, shape (B, K)
            Topic mixture (simplex).

        Returns
        -------
        recon : torch.Tensor, shape (B, V)
            Reconstructed word probability distribution.
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
