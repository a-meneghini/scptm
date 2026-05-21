"""
scptm/training.py
-----------------
Training loop for SCPTM with all fixes applied:
  [FIX-1] BoW normalisation before reconstruction loss
  [FIX-2] beta invalidated after every optimizer step, recomputed lazily
           (full recompute every beta_refresh_epochs — expensive but correct)
  [FIX-3] topic_diversity_weight applied to repulsion loss
"""

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute total ELBO loss for a mini-batch.

    Returns
    -------
    loss, recon_loss, kl_loss : all scalar tensors
    """
    B = len(batch_indices)
    idx_np = batch_indices.cpu().numpy()

    # ---- Reparameterise & decode ----
    z = model.reparameterize(mu, logvar)
    theta_d = F.softmax(z, dim=-1)
    recon_probs = model.decode(theta_d)                             # (B, V)

    # ---- BoW target  [FIX-1] ----
    bow_raw = torch.tensor(
        bow_sparse[idx_np].toarray(), dtype=torch.float32, device=device
    )
    bow_target = normalise_bow(bow_raw, cfg.bow_normalization)

    # Reconstruction loss: cross-entropy on normalised BoW
    recon_loss = -torch.sum(bow_target * torch.log(recon_probs + 1e-10)) / B

    # ---- KL divergence with free bits ----
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())   # (B, K)
    if cfg.free_bits > 0.0:
        kl_per_dim = torch.clamp(kl_per_dim, min=cfg.free_bits)
    kl_loss = kl_per_dim.sum() / B

    return recon_loss, kl_loss


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
        epoch_loss = epoch_recon = epoch_kl = 0.0
        num_batches = 0

        # ---- [FIX-2] Refresh beta at start of epoch (full recompute) ----
        if epoch % cfg.beta_refresh_epochs == 0:
            model.compute_contextual_beta(ctx_embs_list, static_word_embs)

        if loader is not None:
            # ------ Neighbour-sampling path ------
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad()

                with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                    mu, logvar = model.encode(batch.x_dict, batch.edge_index_dict)
                    doc_indices = batch["doc"].n_id
                    recon_loss, kl_loss = _compute_loss(
                        model, mu, logvar, doc_indices, bow_csr, cfg, device
                    )
                    div_loss = model.topic_diversity_loss()
                    loss = (
                        recon_loss
                        + kl_weight * kl_loss
                        + cfg.topic_diversity_weight * div_loss
                    )

                _backward(loss, optimizer, scaler, model)
                # [FIX-2] mark beta stale after weight update
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
                    # [FIX-1] encoder always goes through model.encode
                    mu, logvar = model.encode(
                        graph_data.x_dict, graph_data.edge_index_dict
                    )
                    mu_b     = mu[batch_indices]
                    logvar_b = logvar[batch_indices]
                    recon_loss, kl_loss = _compute_loss(
                        model, mu_b, logvar_b, batch_indices, bow_csr, cfg, device
                    )
                    div_loss = model.topic_diversity_loss()
                    loss = (
                        recon_loss
                        + kl_weight * kl_loss
                        + cfg.topic_diversity_weight * div_loss
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
            top_words = extract_top_words(model, None, top_k=10)
            npmi = compute_npmi_coherence(top_words, bow_csr, list(range(len(top_words[0]))))
            div  = compute_topic_diversity(top_words)
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
