"""
scptm/visualization.py
----------------------
Visualisation utilities for SCPTM.

[FIX-5] All UMAP calls pass random_state from cfg for reproducibility.
"""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn.functional as F
import umap


# ---------------------------------------------------------------------------
# Training history
# ---------------------------------------------------------------------------

def plot_training_history(history: dict, save_path: str = "training_history.png"):
    """Plot loss curves and per-epoch metrics."""
    has_metrics = len(history.get("coherence_npmi", [])) > 0
    n_plots = 5 if has_metrics else 3
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))

    axes[0].plot(history["loss"],  color="#e63946", label="Total")
    axes[0].plot(history["recon"], color="#457b9d", linestyle="--", label="Recon")
    axes[0].set_title("Training Loss")
    axes[0].legend()

    axes[1].plot(history["kl"],        color="#2a9d8f")
    axes[1].set_title("KL Divergence")

    axes[2].plot(history["kl_weight"], color="#e9c46a")
    axes[2].set_title("KL Annealing Weight")

    if has_metrics:
        me = history["metric_epochs"]
        axes[3].plot(me, history["coherence_npmi"],  color="#8338ec", marker="o", markersize=4)
        axes[3].axhline(0.1, color="gray", linestyle=":", label="good threshold")
        axes[3].set_title("NPMI Coherence")

        axes[4].plot(me, history["topic_diversity"], color="#fb5607", marker="s", markersize=4)
        axes[4].axhline(0.5, color="gray", linestyle=":", label="min threshold")
        axes[4].set_title("Topic Diversity")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Helper: build UMAP projection
# ---------------------------------------------------------------------------

def _build_umap_projection(
    topic_vecs: np.ndarray,
    word_vecs: np.ndarray,
    n_components: int = 3,
    random_state: int = 42,
) -> np.ndarray:
    """Joint UMAP projection of topic + word vectors."""
    all_vecs = np.vstack([topic_vecs, word_vecs])
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=15,
        min_dist=0.1 if n_components == 3 else 0.4,
        metric="cosine",
        random_state=random_state,   # [FIX-5] reproducibility
    )
    return reducer.fit_transform(all_vecs)


# ---------------------------------------------------------------------------
# 3D interactive (Plotly)
# ---------------------------------------------------------------------------

def view_semantic_constellations_3d(
    model,
    orig_word_embs: torch.Tensor,
    orig_vocab: List[str],
    mwe_embs_tensor: torch.Tensor,
    mwe_vocab,
    topics_dict: dict,
    random_state: int = 42,
):
    """Interactive 3D Plotly visualisation of topic semantic space."""
    print("Computing 3D UMAP projection...")
    full_vocab  = orig_vocab + list(mwe_vocab)
    full_embs   = torch.cat([orig_word_embs, mwe_embs_tensor], dim=0)
    topic_vecs  = model.topic_embeddings.detach().cpu().numpy()

    # Collect only words referenced by topics
    target_words = set()
    for t_data in topics_dict.values():
        target_words.update(t_data["single"] + t_data["phrases"])
    target_indices = [full_vocab.index(w) for w in target_words if w in full_vocab]

    emb_3d = _build_umap_projection(
        topic_vecs,
        full_embs[target_indices].cpu().numpy(),
        n_components=3,
        random_state=random_state,
    )
    t_coords = emb_3d[: len(topic_vecs)]
    w_dict = {
        full_vocab[target_indices[i]]: emb_3d[len(topic_vecs) + i]
        for i in range(len(target_indices))
    }

    fig = go.Figure()
    colors = px.colors.qualitative.Dark24

    for k, (t_name, t_data) in enumerate(topics_dict.items()):
        color = colors[k % len(colors)]
        tc    = t_coords[k]
        ex, ey, ez, wx, wy, wz, wt = [], [], [], [], [], [], []

        for w in t_data["single"] + t_data["phrases"]:
            if w in w_dict:
                wc = w_dict[w]
                ex.extend([tc[0], wc[0], None])
                ey.extend([tc[1], wc[1], None])
                ez.extend([tc[2], wc[2], None])
                wx.append(wc[0])
                wy.append(wc[1])
                wz.append(wc[2])
                wt.append(w)

        fig.add_trace(go.Scatter3d(
            x=ex, y=ey, z=ez, mode="lines",
            line=dict(color=color, width=1.5), opacity=0.4, showlegend=False,
        ))
        fig.add_trace(go.Scatter3d(
            x=wx, y=wy, z=wz, mode="markers+text",
            marker=dict(color=color, size=5, line=dict(color="white", width=0.5)),
            text=wt, textposition="top center",
            textfont=dict(color="#222222", size=10), showlegend=False,
        ))
        fig.add_trace(go.Scatter3d(
            x=[tc[0]], y=[tc[1]], z=[tc[2]], mode="markers+text",
            marker=dict(color=color, size=14, symbol="diamond",
                        line=dict(color="#333", width=1)),
            text=[t_name], textposition="bottom center",
            textfont=dict(color="#000000", size=14, family="Arial Black"),
            name=t_name,
        ))

    fig.update_layout(
        title=dict(text="Semantic Space — Topic Network",
                   font=dict(color="#333", size=28)),
        paper_bgcolor="#f8f9fa", plot_bgcolor="#f8f9fa",
        margin=dict(l=0, r=0, b=0, t=50),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor="#f8f9fa",
        ),
    )
    fig.show()


# ---------------------------------------------------------------------------
# 2D static (matplotlib, for papers)
# ---------------------------------------------------------------------------

def view_semantic_2d_paper(
    model,
    orig_word_embs: torch.Tensor,
    orig_vocab: List[str],
    mwe_embs_tensor: torch.Tensor,
    mwe_vocab,
    topics_dict: dict,
    save_path: str = "semantic_space_2d.png",
    random_state: int = 42,
):
    """High-resolution 2D static UMAP map for paper figures."""
    print("Computing 2D UMAP projection...")
    full_vocab  = orig_vocab + list(mwe_vocab)
    full_embs   = torch.cat([orig_word_embs, mwe_embs_tensor], dim=0)
    topic_vecs  = model.topic_embeddings.detach().cpu().numpy()

    target_words = set()
    for t_data in topics_dict.values():
        target_words.update(t_data["single"] + t_data["phrases"])
    target_indices = [full_vocab.index(w) for w in target_words if w in full_vocab]
    target_vocab   = [full_vocab[i] for i in target_indices]

    emb_2d = _build_umap_projection(
        topic_vecs,
        full_embs[target_indices].cpu().numpy(),
        n_components=2,
        random_state=random_state,
    )
    t_coords = emb_2d[: len(topic_vecs)]
    w_dict   = {target_vocab[i]: emb_2d[len(topic_vecs) + i] for i in range(len(target_vocab))}

    plt.figure(figsize=(14, 10), dpi=300)
    colors = plt.cm.tab10.colors

    for k, (t_name, t_data) in enumerate(topics_dict.items()):
        color = colors[k % len(colors)]
        tc = t_coords[k]
        for w in t_data["single"] + t_data["phrases"]:
            if w in w_dict:
                wc = w_dict[w]
                plt.plot([tc[0], wc[0]], [tc[1], wc[1]],
                         color=color, alpha=0.15, linewidth=1, zorder=1)
                plt.scatter(wc[0], wc[1], color=color, s=30,
                            alpha=0.7, edgecolors="none", zorder=2)
                plt.text(wc[0], wc[1] + 0.08, w, fontsize=8,
                         color="#333333", alpha=0.9, ha="center", va="bottom", zorder=3)
        plt.scatter(tc[0], tc[1], color=color, s=350, marker="*",
                    edgecolors="black", linewidths=1, zorder=4)
        plt.text(tc[0], tc[1] - 0.15, t_name, fontsize=12, fontweight="bold",
                 color="black", ha="center", va="top", zorder=5,
                 bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2))

    plt.title("UMAP 2D Semantic Space + Topics", fontsize=16, fontweight="bold", pad=20)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, format="png", bbox_inches="tight", dpi=300)
    print(f"High-resolution figure saved: {save_path}")
    plt.show()
