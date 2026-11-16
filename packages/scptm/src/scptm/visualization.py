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
    n_neighbors: int = 15,
    min_dist: Optional[float] = None,
    spread: float = 1.0,
) -> np.ndarray:
    """
    Joint UMAP projection of topic + word vectors.

    ``min_dist`` defaults to 0.1 in 3D and 0.35 in 2D: the 2D figure has to fit
    readable labels, so it must not collapse semantically similar words onto the
    same pixel.
    """
    if min_dist is None:
        min_dist = 0.1 if n_components == 3 else 0.35
    all_vecs = np.vstack([topic_vecs, word_vecs])
    n = all_vecs.shape[0]
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=min(max(2, n_neighbors), max(2, n - 1)),
        min_dist=min_dist,
        spread=spread,
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

# Offsets in typographic points: (dx, dy, horizontal alignment, vertical alignment).
_LABEL_OFFSETS = (
    (0, 6, "center", "bottom"),
    (0, -6, "center", "top"),
    (6, 0, "left", "center"),
    (-6, 0, "right", "center"),
    (6, 6, "left", "bottom"),
    (-6, 6, "right", "bottom"),
    (6, -6, "left", "top"),
    (-6, -6, "right", "top"),
    (0, 13, "center", "bottom"),
    (0, -13, "center", "top"),
    (13, 0, "left", "center"),
    (-13, 0, "right", "center"),
    (11, 11, "left", "bottom"),
    (-11, 11, "right", "bottom"),
    (11, -11, "left", "top"),
    (-11, -11, "right", "top"),
    (0, 22, "center", "bottom"),
    (0, -22, "center", "top"),
    (22, 0, "left", "center"),
    (-22, 0, "right", "center"),
)

# Last-resort ring for topic labels, drawn with a thin leader line.
_FAR_OFFSETS = tuple(
    (r * np.cos(a), r * np.sin(a), "center", "center")
    for r in (38, 58, 82, 112, 150, 195, 250, 320, 400, 480, 560)
    for a in np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, 16, endpoint=False)
)


def _ring_offsets(max_radius_pt: float, step: float = 16.0, angles: int = 16):
    """Dense ring of candidate offsets, used only when the fixed rings fail."""
    rings = np.arange(20.0, max(40.0, max_radius_pt), step)
    return tuple(
        (r * np.cos(a), r * np.sin(a), "center", "center")
        for r in rings
        for a in np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, angles, endpoint=False)
    )


def _palette(n: int, cmap=None) -> List:
    """Distinct colours for ``n`` topics (tab20 repeats past 20)."""
    if cmap is not None:
        return [cmap(i / max(n - 1, 1)) for i in range(n)]
    if n <= 10:
        return list(plt.cm.tab10.colors[:max(n, 1)])
    if n <= 20:
        return list(plt.cm.tab20.colors[:max(n, 1)])
    return [plt.cm.turbo(i / max(n - 1, 1)) for i in range(n)]


def _text_box(ax, renderer, xy, text, *, fontsize, fontweight, fontstyle,
              dx, dy, ha, va, dpi, pad_pt=2.0):
    """Display-space bbox of ``text`` at a candidate offset, from text metrics."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.transforms import Bbox

    prop = FontProperties(size=fontsize, weight=fontweight, style=fontstyle)
    width, height, _descent = renderer.get_text_width_height_descent(
        text, prop, ismath=False
    )
    px, py = ax.transData.transform(xy)
    ox, oy = dx * dpi / 72.0, dy * dpi / 72.0
    if ha == "left":
        x0 = px + ox
    elif ha == "right":
        x0 = px + ox - width
    else:
        x0 = px + ox - width / 2.0
    if va == "bottom":
        y0 = py + oy
    elif va == "top":
        y0 = py + oy - height
    else:
        y0 = py + oy - height / 2.0
    pad = pad_pt * dpi / 72.0
    return Bbox.from_extents(x0 - pad, y0 - pad, x0 + width + pad, y0 + height + pad)


def _point_bbox(ax, xy, size_pt: float, dpi: float, pad: float = 0.0):
    """Display-space box around a scatter marker (``size_pt`` = marker area)."""
    from matplotlib.transforms import Bbox

    px, py = ax.transData.transform(xy)
    radius = (np.sqrt(max(size_pt, 1.0)) / 2.0 + pad) * dpi / 72.0
    return Bbox.from_extents(px - radius, py - radius, px + radius, py + radius)


def _place(ax, fig, renderer, xy, text, *, occupied, fontsize, color,
           fontweight="normal", fontstyle="normal", offsets=_LABEL_OFFSETS,
           leader=False, zorder=6, dpi=300.0):
    """
    Place ``text`` near ``xy`` at the first collision-free offset, or return
    ``None``.  ``leader=True`` draws a leader line for far offsets.
    """
    canvas = fig.bbox
    for dx, dy, ha, va in offsets:
        box = _text_box(ax, renderer, xy, text, fontsize=fontsize,
                        fontweight=fontweight, fontstyle=fontstyle,
                        dx=dx, dy=dy, ha=ha, va=va, dpi=dpi)
        # A little outside the axes is fine: bbox_inches="tight" grows the
        # canvas, and crowded figures need every ring they can get.
        mx = 0.05 * (canvas.x1 - canvas.x0)
        my = 0.05 * (canvas.y1 - canvas.y0)
        inside = (box.x0 >= canvas.x0 - mx and box.x1 <= canvas.x1 + mx
                  and box.y0 >= canvas.y0 - my and box.y1 <= canvas.y1 + my)
        if not inside or any(box.overlaps(other) for other in occupied):
            continue

        kwargs = {}
        if leader and (abs(dx) > 25 or abs(dy) > 25):
            kwargs["arrowprops"] = dict(arrowstyle="-", color=color, alpha=0.55,
                                        linewidth=0.6, shrinkA=0, shrinkB=4)
        artist = ax.annotate(
            text, xy=xy, xytext=(dx, dy), textcoords="offset points",
            fontsize=fontsize, color=color, fontweight=fontweight,
            fontstyle=fontstyle, ha=ha, va=va, zorder=zorder,
            annotation_clip=False,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.4),
            **kwargs,
        )
        # Verify with the real text extent: backend metrics can differ slightly
        # from the estimate, and this is what guarantees no overlap in the output.
        from matplotlib.text import Text

        # Annotation positions are only resolved during draw(); do it now so the
        # measured box is in display coordinates rather than offset coordinates.
        update = getattr(artist, "update_positions", None)
        if update is not None:
            update(renderer)
        real = Text.get_window_extent(artist, renderer)
        if not any(real.overlaps(other) for other in occupied):
            occupied.append(real)
            return artist
        artist.remove()
    return None


def view_semantic_2d_paper(
    model,
    orig_word_embs: torch.Tensor,
    orig_vocab: List[str],
    mwe_embs_tensor: torch.Tensor,
    mwe_vocab,
    topics_dict: dict,
    save_path: str = "semantic_space_2d.png",
    random_state: int = 42,
    adjust_labels: bool = True,
    show_mwes: bool = True,
    min_dist: float = 0.35,
    n_neighbors: int = 30,
    figsize: tuple = (14.0, 10.0),
    dpi: int = 300,
    cmap=None,
):
    """
    High-resolution 2D static UMAP map for paper figures.

    Labels are placed by a deterministic greedy layout that never overlaps:
    each label tries a ring of offsets around its anchor and keeps the first one
    whose box hits no other label, dot or star; words that cannot be placed are
    dropped rather than stacked.  Topic labels that find no room nearby move
    further out on a thin leader line.  ``adjust_labels=False`` reproduces the
    old fixed-offset behaviour, for comparison.
    """
    print("Computing 2D UMAP projection...")
    full_vocab = list(orig_vocab) + list(mwe_vocab)
    full_embs = torch.cat([orig_word_embs, mwe_embs_tensor], dim=0)
    index = {w: i for i, w in enumerate(full_vocab)}
    topic_vecs = model.topic_embeddings.detach().cpu().numpy()
    topic_names = list(topics_dict)

    # Labels once each, in order of first appearance, so the projection is stable.
    labels: List[str] = []
    seen = set()
    for name in topic_names:
        t_data = topics_dict[name]
        for word in list(t_data["single"]) + list(t_data["phrases"]):
            if word not in seen and word in index:
                seen.add(word)
                labels.append(word)

    vecs = full_embs[[index[w] for w in labels]].detach().cpu().numpy()
    emb_2d = _build_umap_projection(
        topic_vecs, vecs,
        n_components=2, random_state=random_state,
        n_neighbors=n_neighbors, min_dist=min_dist,
    )
    t_coords = emb_2d[: len(topic_vecs)]
    coords = {w: emb_2d[len(topic_vecs) + i] for i, w in enumerate(labels)}

    n_topics = len(topic_names)
    colors = _palette(n_topics, cmap)
    line_alpha = 0.12 if n_topics <= 10 else 0.07
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    word_colors: Dict[str, object] = {}
    per_topic_labels: List[List[str]] = []
    for k, name in enumerate(topic_names):
        t_data = topics_dict[name]
        color = colors[k]
        tc = t_coords[k]
        topic_labels = [w for w in list(t_data["single"]) + list(t_data["phrases"])
                        if w in coords]
        per_topic_labels.append(topic_labels)
        for word in topic_labels:
            word_colors.setdefault(word, color)
            wc = coords[word]
            ax.plot([tc[0], wc[0]], [tc[1], wc[1]], color=color,
                    alpha=line_alpha, linewidth=0.8, zorder=1)

        singles = [w for w in t_data["single"] if w in coords]
        phrases = [w for w in t_data["phrases"] if w in coords] if show_mwes else []
        if singles:
            ax.scatter([coords[w][0] for w in singles],
                       [coords[w][1] for w in singles],
                       color=color, s=30, alpha=0.8, edgecolors="white",
                       linewidths=0.4, zorder=2)
        if phrases:
            ax.scatter([coords[w][0] for w in phrases],
                       [coords[w][1] for w in phrases],
                       color=color, s=26, alpha=0.85, marker="D",
                       edgecolors="white", linewidths=0.4, zorder=2)
        ax.scatter(tc[0], tc[1], color=color, s=350, marker="*",
                   edgecolors="black", linewidths=1.0, zorder=4)

    # Fix the layout *before* placing labels: tight_layout rescales the axes and
    # text does not, which would reintroduce overlaps after the fact.
    ax.set_title("UMAP 2D Semantic Space with Topics",
                 fontsize=16, fontweight="bold", pad=18)
    ax.axis("off")
    fig.tight_layout()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    occupied: list = []
    if adjust_labels:
        for tc in t_coords:
            occupied.append(_point_bbox(ax, tc, 350.0, dpi, pad=1.0))
        for word in labels:
            occupied.append(_point_bbox(ax, coords[word], 30.0, dpi))

        for tc, name, color in zip(t_coords, topic_names, colors):
            artist = _place(ax, fig, renderer, tc, name, occupied=occupied,
                            fontsize=11.0, color="black", fontweight="bold",
                            zorder=7, dpi=dpi)
            if artist is None:
                artist = _place(ax, fig, renderer, tc, name, occupied=occupied,
                                fontsize=11.0, color="black", fontweight="bold",
                                offsets=_FAR_OFFSETS, leader=True, zorder=7, dpi=dpi)
            if artist is None:          # pack anywhere that is still free
                max_radius = float(np.hypot(fig.bbox.width, fig.bbox.height)) / dpi * 72.0
                _place(ax, fig, renderer, tc, name, occupied=occupied,
                       fontsize=9.5, color="black", fontweight="bold",
                       offsets=_ring_offsets(max_radius), leader=True,
                       zorder=7, dpi=dpi)

        # Words by rank across topics: rank-1 words win the space around their topic.
        placed_words = set()
        max_rank = max((len(t_data["single"]) for t_data in topics_dict.values()),
                       default=0)
        for rank in range(max_rank):
            for name in topic_names:
                singles = [w for w in topics_dict[name]["single"] if w in coords]
                if rank >= len(singles):
                    continue
                word = singles[rank]
                if word in placed_words:
                    continue
                if _place(ax, fig, renderer, coords[word], word,
                          occupied=occupied, fontsize=8.0,
                          color=word_colors.get(word, "#333333"), dpi=dpi):
                    placed_words.add(word)

        if show_mwes:
            placed_phrases = set()
            for name in topic_names:
                for phrase in topics_dict[name]["phrases"]:
                    if (phrase in placed_words or phrase in placed_phrases
                            or phrase not in coords):
                        continue
                    if _place(ax, fig, renderer, coords[phrase], phrase,
                              occupied=occupied, fontsize=7.0,
                              color=word_colors.get(phrase, "#333333"),
                              fontstyle="italic", dpi=dpi):
                        placed_phrases.add(phrase)
    else:                                   # legacy fixed-offset layout
        for k, name in enumerate(topic_names):
            color = colors[k]
            tc = t_coords[k]
            for word in per_topic_labels[k]:
                wc = coords[word]
                ax.text(wc[0], wc[1] + 0.08, word, fontsize=8, color="#333333",
                        alpha=0.9, ha="center", va="bottom", zorder=3)
            ax.text(tc[0], tc[1] - 0.15, name, fontsize=12, fontweight="bold",
                    color="black", ha="center", va="top", zorder=5,
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2))

    fig.savefig(save_path, format="png", bbox_inches="tight", dpi=dpi)
    print(f"High-resolution figure saved: {save_path}")
    plt.show()
