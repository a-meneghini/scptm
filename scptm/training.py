"""
scptm/training.py
-----------------
Training loop for SCPTM.

Public API
----------
train(graph_data, bow_sparse, model, ctx_embs_list, static_word_embs,
      cfg, device, vocab=None) → history dict

    Trains the model and returns per-epoch metrics:
    loss, recon, kl, kl_weight, coherence_npmi, topic_diversity,
    metric_epochs.

    vocab : list[str] | None
        Pass the full vocabulary list to get correct NPMI coherence in the
        training log.  If None, NPMI is reported as nan.

compute_kl_weight(epoch, kl_warmup_epochs, kl_max, strategy) → float
normalise_bow(bow, strategy) → Tensor

Design notes
------------
* decode_train() is used in the loss (not the cached beta) so the
  reconstruction gradient flows back to topic_embeddings via the
  differentiable cosine-similarity beta.

* beta is invalidated (model.invalidate_beta()) after every optimizer step
  and fully recomputed every cfg.beta_refresh_epochs epochs.

* BoW target is normalised (TF / log1p / none) before cross-entropy so that
  the loss is not dominated by document length.

* Gradient clipping at max_norm=5.0 to prevent exploding gradients in the GNN.

* AMP (mixed precision) is enabled on CUDA when cfg.use_mixed_precision=True.
"""

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm


# ---------------------------------------------------------------------------
# PPMI matrix helper (for differentiable NPMI coherence loss)
# ---------------------------------------------------------------------------

def compute_ppmi_tensor(
    bow_csr,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute a dense Positive PMI matrix from a BoW sparse matrix.

    PPMI[i, j] = max(0, log P(i,j) / (P(i)*P(j)))
    where co-occurrence is measured at the document level.

    Note: returns a (V, V) dense tensor — memory-intensive for large V.
    Only call this when npmi_coherence_weight > 0.

    Parameters
    ----------
    bow_csr : scipy sparse matrix, shape (n_docs, V)
    device  : target torch device

    Returns
    -------
    ppmi : torch.Tensor, shape (V, V), on `device`
    """
    n_docs = bow_csr.shape[0]
    bow_bin = (bow_csr > 0).astype(np.float32)
    # Co-occurrence: C[i,j] = # docs containing both word i and word j
    cooc = (bow_bin.T @ bow_bin).toarray()          # (V, V)
    p_w = cooc.diagonal() / n_docs                  # P(w)
    p_joint = cooc / n_docs                         # P(wi, wj)
    outer = np.outer(p_w, p_w)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.where(outer > 0, np.log(p_joint / (outer + 1e-10) + 1e-10), 0.0)
    ppmi = np.maximum(pmi, 0.0)
    np.fill_diagonal(ppmi, 0.0)
    return torch.tensor(ppmi, dtype=torch.float32, device=device)

from .config import SCPTMConfig
from .nn import VariationalGraphTopicModel


# ---------------------------------------------------------------------------
# KL annealing schedule
# ---------------------------------------------------------------------------

def compute_kl_weight(
    epoch: int,
    kl_warmup_epochs: int,
    kl_max: float,
    strategy: str = "linear",
) -> float:
    """Return the KL weight for the current epoch."""
    if strategy == "constant":
        return kl_max
    if strategy == "linear":
        return min(kl_max, kl_max * epoch / max(kl_warmup_epochs, 1))
    if strategy == "cyclical":
        t = epoch % kl_warmup_epochs
        return kl_max * min(1.0, t / (kl_warmup_epochs / 2))
    raise ValueError(f"Unknown KL strategy: {strategy}")


# ---------------------------------------------------------------------------
# BoW normalisation helper
# ---------------------------------------------------------------------------

def normalise_bow(
    bow: torch.Tensor,
    strategy: str = "tf",
) -> torch.Tensor:
    """
    Normalise a raw BoW count matrix before computing reconstruction loss.

    Parameters
    ----------
    bow : torch.Tensor, shape (B, V) — raw integer counts
    strategy : "none" | "tf" | "log1p"

    Returns
    -------
    Normalised tensor, same shape.
    """
    if strategy == "none":
        return bow
    if strategy == "tf":
        row_sum = bow.sum(dim=-1, keepdim=True).clamp(min=1.0)
        return bow / row_sum
    if strategy == "log1p":
        return torch.log1p(bow)
    raise ValueError(f"Unknown bow_normalization: {strategy}")


# ---------------------------------------------------------------------------
# Single training step (shared between full-batch and neighbour-sampling)
# ---------------------------------------------------------------------------

def _compute_loss(
    model: VariationalGraphTopicModel,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    batch_indices: torch.Tensor,
    bow_sparse,
    cfg: SCPTMConfig,
    device: torch.device,
    static_word_embs: torch.Tensor,
    ppmi_tensor: Optional[torch.Tensor] = None,
    prior_mu_b: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute ELBO loss for a mini-batch, plus optional coherence terms.

    Parameters
    ----------
    prior_mu_b : Tensor, shape (B, K) or None
        Covariate-informed prior mean for this batch.
        When None, uses the standard N(0, I) prior.

    Returns
    -------
    recon_loss, kl_loss, we_coh_loss, npmi_coh_loss : scalar tensors
    """
    B = len(batch_indices)
    idx_np = batch_indices.cpu().numpy()

    # ---- Reparameterise & decode (differentiable path) ----
    z = model.reparameterize(mu, logvar)
    theta_d = F.softmax(z, dim=-1)
    recon_probs = model.decode_train(theta_d, static_word_embs)     # (B, V)

    # ---- BoW target ----
    bow_raw = torch.tensor(
        bow_sparse[idx_np].toarray(), dtype=torch.float32, device=device
    )
    bow_target = normalise_bow(bow_raw, cfg.bow_normalization)

    # Reconstruction loss: cross-entropy on normalised BoW
    recon_loss = -torch.sum(bow_target * torch.log(recon_probs + 1e-10)) / B

    # ---- KL divergence with free bits ----
    # When a covariate prior is given, shift the N(0,I) prior to N(Γx, I):
    #   KL = -0.5 * sum(1 + logvar - (mu - prior_mu)^2 - exp(logvar))
    if prior_mu_b is not None:
        kl_per_dim = -0.5 * (
            1 + logvar - (mu - prior_mu_b).pow(2) - logvar.exp()
        )
    else:
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    if cfg.free_bits > 0.0:
        kl_per_dim = torch.clamp(kl_per_dim, min=cfg.free_bits)
    kl_loss = kl_per_dim.sum() / B

    # ---- Optional: WE-coherence loss ----
    if cfg.we_coherence_weight > 0.0:
        we_coh_loss = model.we_coherence_loss(static_word_embs)
    else:
        we_coh_loss = torch.tensor(0.0, device=device)

    # ---- Optional: NPMI coherence loss ----
    if cfg.npmi_coherence_weight > 0.0 and ppmi_tensor is not None:
        npmi_coh_loss = model.npmi_coherence_loss(ppmi_tensor, static_word_embs)
    else:
        npmi_coh_loss = torch.tensor(0.0, device=device)

    return recon_loss, kl_loss, we_coh_loss, npmi_coh_loss


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    graph_data: HeteroData,
    bow_sparse,
    model: VariationalGraphTopicModel,
    ctx_embs_list: list,
    static_word_embs: torch.Tensor,
    cfg: SCPTMConfig,
    device: torch.device,
    vocab: Optional[List[str]] = None,
    covariate_tensor: Optional[torch.Tensor] = None,
) -> dict:
    """
    Train SCPTM and return a history dict with per-epoch metrics.

    Parameters
    ----------
    graph_data : HeteroData
        Already on `device`.
    bow_sparse : scipy sparse matrix
        Raw BoW counts, shape (n_docs, vocab_size).
    model : VariationalGraphTopicModel
        Already on `device`.
    ctx_embs_list : list
        Per-word contextual embeddings (CPU tensors).
    static_word_embs : torch.Tensor
        Static SBERT word embeddings on `device`.
    cfg : SCPTMConfig
    device : torch.device
    vocab : list[str] | None
        Vocabulary for NPMI coherence logging.  When None the metric is
        skipped (avoids the index-vs-string mismatch that previously always
        returned 0.0).
    covariate_tensor : Tensor, shape (n_segments, n_covariates) or None
        Per-segment covariate values (already on CPU; moved to device inside).
        When provided, conditions the KL prior à la STM: KL(q || N(Γx, I)).

    Returns
    -------
    history : dict with keys loss, recon, kl, kl_weight,
              coherence_npmi, topic_diversity, metric_epochs
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scaler = (
        torch.cuda.amp.GradScaler()
        if cfg.use_mixed_precision and device.type == "cuda"
        else None
    )

    bow_csr = bow_sparse.tocsr()
    n_docs = graph_data["doc"].x.shape[0]
    history = {
        "loss": [], "recon": [], "kl": [], "kl_weight": [],
        "coherence_npmi": [], "topic_diversity": [], "metric_epochs": [],
    }

    # Pre-compute PPMI tensor if NPMI coherence loss is enabled.
    # This is a one-time O(n_docs * V) operation; skipped when weight == 0.
    ppmi_tensor: Optional[torch.Tensor] = None
    if cfg.npmi_coherence_weight > 0.0:
        print("  Pre-computing PPMI matrix for coherence loss...")
        ppmi_tensor = compute_ppmi_tensor(bow_csr, device)

    # Build NeighborLoader once if requested
    use_loader = cfg.use_neighbor_sampling and cfg.graph_mode != "none"
    loader = None
    if use_loader:
        loader = NeighborLoader(
            graph_data,
            num_neighbors=[10, 10],
            batch_size=cfg.batch_size,
            input_nodes=("doc", None),
            shuffle=True,
        )

    # Initial beta computation
    model.compute_contextual_beta(ctx_embs_list, static_word_embs)

    print(f"\nTraining — mode={cfg.graph_mode}, device={device}")
    print(f"  Mixed precision: {cfg.use_mixed_precision} | Neighbor sampling: {use_loader}")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        kl_weight = compute_kl_weight(
            epoch, cfg.kl_warmup_epochs, cfg.kl_max, cfg.kl_strategy
        )

        # ---- Adaptive KL boost (anti-collapse) ----
        # When topic entropy is below threshold (topics collapsing), temporarily
        # increase KL weight to push the posterior back towards the prior.
        if cfg.adaptive_kl and model._cached_beta is not None:
            with torch.no_grad():
                beta = model._cached_beta                            # (K, V)
                entropy = -(beta * torch.log(beta + 1e-10)).sum(dim=-1).mean()
                if entropy.item() < cfg.min_topic_entropy:
                    kl_weight = min(kl_weight + cfg.adaptive_kl_boost,
                                    cfg.kl_max * 2.0)

        epoch_loss = epoch_recon = epoch_kl = 0.0
        num_batches = 0

        # ---- Refresh beta at start of epoch ----
        if epoch % cfg.beta_refresh_epochs == 0:
            model.compute_contextual_beta(ctx_embs_list, static_word_embs)

        # Move covariate to device once per epoch (cheap if already there)
        cov_dev = covariate_tensor.to(device) if covariate_tensor is not None else None

        if loader is not None:
            # ------ Neighbour-sampling path ------
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad()

                with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                    mu, logvar = model.encode(batch.x_dict, batch.edge_index_dict)
                    doc_indices = batch["doc"].n_id
                    prior_mu_b = (
                        model.compute_prior_mean(cov_dev[doc_indices.cpu()])
                        if cov_dev is not None else None
                    )
                    recon_loss, kl_loss, we_loss, npmi_loss = _compute_loss(
                        model, mu, logvar, doc_indices, bow_csr, cfg, device,
                        static_word_embs, ppmi_tensor, prior_mu_b=prior_mu_b,
                    )
                    div_loss = model.topic_diversity_loss()
                    loss = (
                        recon_loss
                        + kl_weight * kl_loss
                        + cfg.topic_diversity_weight * div_loss
                        + cfg.we_coherence_weight   * we_loss
                        + cfg.npmi_coherence_weight * npmi_loss
                    )

                _backward(loss, optimizer, scaler, model)
                model.invalidate_beta()

                epoch_loss  += loss.item()
                epoch_recon += recon_loss.item()
                epoch_kl    += kl_loss.item()
                num_batches += 1

        else:
            # ------ Standard full-graph path ------
            perm = torch.randperm(n_docs)
            n_batches = math.ceil(n_docs / cfg.batch_size)
            num_batches = n_batches

            for b in range(n_batches):
                start = b * cfg.batch_size
                batch_indices = perm[start: min(start + cfg.batch_size, n_docs)]
                optimizer.zero_grad()

                with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                    mu, logvar = model.encode(
                        graph_data.x_dict, graph_data.edge_index_dict
                    )
                    mu_b     = mu[batch_indices]
                    logvar_b = logvar[batch_indices]
                    # Compute prior_mu fresh per batch so each backward() has its
                    # own graph — computing it once outside the loop and indexing
                    # would free the graph after the first batch.
                    prior_mu_b = (
                        model.compute_prior_mean(cov_dev[batch_indices])
                        if cov_dev is not None else None
                    )
                    recon_loss, kl_loss, we_loss, npmi_loss = _compute_loss(
                        model, mu_b, logvar_b, batch_indices, bow_csr, cfg, device,
                        static_word_embs, ppmi_tensor, prior_mu_b=prior_mu_b,
                    )
                    div_loss = model.topic_diversity_loss()
                    loss = (
                        recon_loss
                        + kl_weight * kl_loss
                        + cfg.topic_diversity_weight * div_loss
                        + cfg.we_coherence_weight   * we_loss
                        + cfg.npmi_coherence_weight * npmi_loss
                    )

                _backward(loss, optimizer, scaler, model)
                model.invalidate_beta()

                epoch_loss  += loss.item()
                epoch_recon += recon_loss.item()
                epoch_kl    += kl_loss.item()

        # ---- Logging ----
        history["loss"].append(epoch_loss / num_batches)
        history["recon"].append(epoch_recon / num_batches)
        history["kl"].append(epoch_kl / num_batches)
        history["kl_weight"].append(kl_weight)

        if epoch % cfg.metrics_every_n_epochs == 0:
            # Lazy metrics: imported here to avoid circular import
            from .evaluation import compute_npmi_coherence, compute_topic_diversity
            from .keywords import extract_top_words
            model.eval()
            # Recompute beta after the epoch's invalidations
            model.compute_contextual_beta(ctx_embs_list, static_word_embs)
            top_words = extract_top_words(model, vocab, top_k=10)
            # compute_npmi_coherence needs word strings; skip when vocab absent
            if vocab is not None and top_words and top_words[0] and isinstance(top_words[0][0], str):
                npmi = compute_npmi_coherence(top_words, bow_csr, vocab)
            else:
                npmi = float("nan")
            div = compute_topic_diversity(top_words if vocab is not None else
                                          [[str(i) for i in row] for row in top_words])
            history["coherence_npmi"].append(npmi)
            history["topic_diversity"].append(div)
            history["metric_epochs"].append(epoch)
            print(
                f"Epoch {epoch:03d}/{cfg.epochs}  "
                f"Loss={epoch_loss/num_batches:.3f}  "
                f"Recon={epoch_recon/num_batches:.3f}  "
                f"KL={epoch_kl/num_batches:.3f}  "
                f"KL-w={kl_weight:.3f}  "
                f"NPMI={npmi:.3f}  Div={div:.3f}"
            )
            model.train()
        elif epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d}/{cfg.epochs}  "
                f"Loss={epoch_loss/num_batches:.3f}  "
                f"Recon={epoch_recon/num_batches:.3f}  "
                f"KL={epoch_kl/num_batches:.3f}  "
                f"KL-w={kl_weight:.3f}"
            )

    return history


def _backward(loss, optimizer, scaler, model):
    """Gradient step with optional AMP scaler and gradient clipping."""
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
