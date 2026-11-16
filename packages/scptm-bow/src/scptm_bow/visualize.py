"""
scptm_bow/visualize.py
----------------------
The 2D semantic-space figure, mirroring ``scptm.visualization.view_semantic_2d_paper``:
topics as stars, their top words as dots, multiword expressions as italic labels,
and a line from each topic to each of its words.

The projection needs word vectors, so it requires ``vector_backend != "none"``.
UMAP and matplotlib are imported lazily: they are optional extras
(``pip install "scptm-bow[plots]"``).  ``reducer`` can be replaced with any
callable mapping ``(n_samples, dim) -> (n_samples, 2)``, which is how the test
suite exercises the figure without UMAP.

Label layout
------------
Word vectors of topically related words project on top of each other, so a
naive ``ax.text`` at every point produces unreadable piles of labels (the
"clusterfuck" figure).  Labels are therefore placed by a deterministic greedy
layout: each one tries a fixed set of offsets around its anchor and keeps the
first position whose bounding box hits no already-placed label, dot or star.
Words that cannot be placed are skipped, not drawn on top of something else;
topics always get a label, moving further out with a leader line when their
immediate neighbourhood is crowded.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .explain import extract_mwes

MWE_PER_TOPIC = 3

# Offsets in typographic points: (dx, dy, horizontal alignment, vertical alignment).
_LABEL_OFFSETS: Tuple[Tuple[float, float, str, str], ...] = (
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
_FAR_OFFSETS: Tuple[Tuple[float, float, str, str], ...] = tuple(
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


def topic_vectors(model, normalize: bool = True) -> np.ndarray:
    """
    Embed each topic in word-vector space as the phi-weighted centroid of its
    terms: ``topic_k = sum_w phi_kw * vec_w``.

    That is the natural counterpart of the neural model's learned
    ``topic_embeddings``: it lives in the same space as the words, so topics and
    words can be projected together.
    """
    if model.vectors is None:
        raise ValueError(
            "The 2D semantic projection needs word vectors, but this model was "
            "fitted with vector_backend='none'. Refit with 'fasttext' or "
            "'ppmi_svd'."
        )
    phi = np.asarray(model.topic_word, dtype=np.float64)      # (K, V)
    vecs = np.asarray(model.vectors, dtype=np.float64)        # (V, D)
    mass = phi.sum(axis=1, keepdims=True)
    mass[mass == 0] = 1.0
    centroids = (phi / mass) @ vecs
    if normalize:
        norm = np.linalg.norm(centroids, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        centroids = centroids / norm
    return centroids


def build_projection(
    vectors: np.ndarray,
    n_components: int = 2,
    random_state: int = 42,
    n_neighbors: int = 30,
    min_dist: float = 0.35,
    spread: float = 1.0,
    reducer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> np.ndarray:
    """
    Project ``(n, D)`` vectors to ``(n, n_components)`` with UMAP by default.

    The defaults are deliberately less tight than UMAP's own (``n_neighbors=15,
    min_dist=0.1``): this figure has to fit readable labels, not preserve local
    micro-structure.
    """
    if reducer is not None:
        return np.asarray(reducer(vectors))
    try:
        import umap
    except ImportError as exc:              # pragma: no cover - optional dep
        raise ImportError(
            "The 2D projection needs umap-learn: pip install \"scptm-bow[plots]\""
        ) from exc

    n = vectors.shape[0]
    return umap.UMAP(
        n_components=n_components,
        n_neighbors=min(max(2, n_neighbors), max(2, n - 1)),
        min_dist=min_dist,
        spread=spread,
        metric="cosine",
        random_state=random_state,
    ).fit_transform(vectors)


# ---------------------------------------------------------------------------
# Label layout
# ---------------------------------------------------------------------------

def _palette(n: int, cmap=None) -> List:
    """Distinct colours for up to ``n`` topics (tab20 repeats past 20)."""
    import matplotlib.pyplot as plt

    if cmap is not None:
        return [cmap(i / max(n - 1, 1)) for i in range(n)]
    if n <= 10:
        return list(plt.cm.tab10.colors[:max(n, 1)])
    if n <= 20:
        return list(plt.cm.tab20.colors[:max(n, 1)])
    return [plt.cm.turbo(i / max(n - 1, 1)) for i in range(n)]


def _text_box(
    ax,
    renderer,
    xy,
    text: str,
    *,
    fontsize: float,
    fontweight: str,
    fontstyle: str,
    dx: float,
    dy: float,
    ha: str,
    va: str,
    dpi: float,
    pad_pt: float = 2.0,
):
    """
    Display-space bounding box of ``text`` at a candidate offset, computed from
    the renderer's text metrics.

    Measuring text is much cheaper than creating an artist and asking for its
    extent, and the result is the same box matplotlib will lay out: that matters
    when a crowded figure tries ~20 offsets for a few hundred labels.
    """
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
    return Bbox.from_extents(
        x0 - pad, y0 - pad, x0 + width + pad, y0 + height + pad
    )


def _point_bbox(ax, xy, size_pt: float, dpi: float, pad: float = 0.0):
    """Display-space box around a scatter marker (``size_pt`` = marker area)."""
    from matplotlib.transforms import Bbox

    px, py = ax.transData.transform(xy)
    radius = (np.sqrt(max(size_pt, 1.0)) / 2.0 + pad) * dpi / 72.0
    return Bbox.from_extents(px - radius, py - radius, px + radius, py + radius)


def _place(
    ax,
    fig,
    renderer,
    xy,
    text: str,
    *,
    occupied: list,
    fontsize: float,
    color,
    fontweight: str = "normal",
    fontstyle: str = "normal",
    offsets=_LABEL_OFFSETS,
    leader: bool = False,
    zorder: int = 6,
    dpi: float = 300.0,
):
    """
    Place ``text`` near ``xy`` at the first collision-free offset, or return
    ``None``.  ``leader=True`` draws a thin line from the anchor to the label,
    used by the far-offset fallback of topic labels.
    """
    canvas = fig.bbox
    for dx, dy, ha, va in offsets:
        box = _text_box(
            ax, renderer, xy, text,
            fontsize=fontsize, fontweight=fontweight, fontstyle=fontstyle,
            dx=dx, dy=dy, ha=ha, va=va, dpi=dpi,
        )
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
            kwargs["arrowprops"] = dict(
                arrowstyle="-", color=color, alpha=0.55,
                linewidth=0.6, shrinkA=0, shrinkB=4,
            )
        artist = ax.annotate(
            text, xy=xy, xytext=(dx, dy), textcoords="offset points",
            fontsize=fontsize, color=color, fontweight=fontweight,
            fontstyle=fontstyle, ha=ha, va=va, zorder=zorder,
            annotation_clip=False,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.4),
            **kwargs,
        )
        # The estimated box is only a cheap filter: the metrics of the active
        # backend can differ slightly (notably on non-Agg canvases), so the
        # artist that survives the filter is checked again with its real extent.
        # Text.get_window_extent is used explicitly because an Annotation's own
        # extent includes its leader arrow (a line back to the topic star), which
        # would make every far placement look like it collides with everything.
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


def _draw_labels(
    ax,
    fig,
    renderer,
    *,
    topic_coords,
    topic_names,
    topic_colors,
    word_coords: Dict[str, np.ndarray],
    word_colors: Dict[str, object],
    per_topic_words: Sequence[Sequence[str]],
    mwes: Dict[str, List[Dict[str, object]]],
    show_mwes: bool,
    occupied: list,
    dpi: float,
    label_size: float,
    topic_label_size: float,
):
    """Topic labels first, then words by rank, then MWE phrases."""
    # Stars occupy space: a label must not sit on top of a marker.
    for xy, size in zip(topic_coords, [350.0] * len(topic_coords)):
        occupied.append(_point_bbox(ax, xy, size, dpi, pad=1.0))
    for xy in word_coords.values():
        occupied.append(_point_bbox(ax, xy, 30.0, dpi))

    for xy, name, color in zip(topic_coords, topic_names, topic_colors):
        artist = _place(
            ax, fig, renderer, xy, name, occupied=occupied,
            fontsize=topic_label_size, color="black", fontweight="bold",
            zorder=7, dpi=dpi,
        )
        if artist is None:                      # crowded: move out with a leader
            artist = _place(
                ax, fig, renderer, xy, name, occupied=occupied,
                fontsize=topic_label_size, color="black", fontweight="bold",
                offsets=_FAR_OFFSETS, leader=True, zorder=7, dpi=dpi,
            )
        if artist is None:                      # pack anywhere that is free
            max_radius = float(np.hypot(fig.bbox.width, fig.bbox.height)) / dpi * 72.0
            artist = _place(
                ax, fig, renderer, xy, name, occupied=occupied,
                fontsize=topic_label_size - 1.5, color="black", fontweight="bold",
                offsets=_ring_offsets(max_radius), leader=True, zorder=7, dpi=dpi,
            )

    # Words by rank across topics: rank-1 words win the space around their topic.
    placed_words = set()
    max_k = max((len(words) for words in per_topic_words), default=0)
    for rank in range(max_k):
        for words in per_topic_words:
            if rank >= len(words):
                continue
            label = words[rank]
            if label in placed_words or label not in word_coords:
                continue
            if _place(
                ax, fig, renderer, word_coords[label], label,
                occupied=occupied, fontsize=label_size,
                color=word_colors.get(label, "#333333"), dpi=dpi,
            ):
                placed_words.add(label)

    if not show_mwes:
        return
    placed_phrases = set()
    for topic_key, entries in mwes.items():
        for entry in entries:
            phrase = str(entry["phrase"])
            if phrase in placed_phrases or phrase in placed_words:
                continue
            parts = [p for p in phrase.split() if p in word_coords]
            if not parts:
                continue
            anchor = np.mean([word_coords[p] for p in parts], axis=0)
            color = word_colors.get(parts[0], "#333333")
            if _place(
                ax, fig, renderer, anchor, phrase,
                occupied=occupied, fontsize=label_size - 1.0,
                color=color, fontstyle="italic", dpi=dpi,
            ):
                placed_phrases.add(phrase)


# ---------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------

def plot_semantic_2d(
    model,
    top_k: int = 12,
    mwe_per_topic: int = MWE_PER_TOPIC,
    save_path: Union[str, Path] = "semantic_space_2d.png",
    random_state: int = 42,
    figsize: Tuple[float, float] = (14.0, 10.0),
    dpi: int = 300,
    reducer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    show: bool = False,
    adjust_labels: bool = True,
    show_mwes: bool = True,
    n_neighbors: int = 30,
    min_dist: float = 0.35,
    cmap=None,
) -> Path:
    """
    Draw the 2D semantic space and save it as a PNG.

    Parameters
    ----------
    model : BowSCPTM
        A fitted model with word vectors.
    top_k : int
        Words shown per topic.  Labels that cannot be placed without overlapping
        are dropped, so fewer may appear; the dots always do.
    mwe_per_topic : int
        Multiword expressions annotated per topic (italic, at the centroid of
        their parts).
    save_path : str | Path
        Output PNG (300 dpi by default).
    figsize, dpi : tuple, int
        Figure size and resolution.
    reducer : callable | None
        ``(n, D) -> (n, 2)``.  Defaults to UMAP.
    show : bool
        Call ``plt.show()`` after saving (useful in notebooks).
    adjust_labels : bool
        Greedy collision-free label layout (default).  ``False`` reproduces the
        old behaviour and is only useful for debugging the layout.
    show_mwes : bool
        Annotate the extracted multiword expressions.
    n_neighbors, min_dist : int, float
        UMAP settings.  ``min_dist=0.35`` is deliberately larger than UMAP's
        default: the plot must fit labels, not preserve micro-structure.
    cmap : matplotlib colormap | None
        Topic colourmap.  Defaults to tab10/tab20 up to 20 topics, turbo above.

    Returns
    -------
    pathlib.Path
        The file that was written.
    """
    model._check_fitted()
    if model.vectors is None:
        raise ValueError(
            "plot_semantic_2d needs word vectors; refit with "
            "vector_backend='fasttext' or 'ppmi_svd'."
        )

    try:
        # Deliberately no matplotlib.use("Agg") here: it is a *global, permanent*
        # switch, so calling plot_semantic_2d(show=False) once would silently
        # disable inline rendering for the rest of a notebook session.  savefig()
        # works on any backend, and headless environments fall back to Agg on
        # their own.
        import matplotlib.pyplot as plt
    except ImportError as exc:              # pragma: no cover - optional dep
        raise ImportError(
            "Plotting needs matplotlib: pip install \"scptm-bow[plots]\""
        ) from exc

    index = {w: i for i, w in enumerate(model.vocab)}
    vectors = np.asarray(model.vectors, dtype=np.float64)

    top_words = model.top_words(top_k)
    mwes = extract_mwes(top_words, model.parsed.dep_counts, top_k=mwe_per_topic) \
        if show_mwes else {}

    # Words once each, in order of first appearance, so the projection is stable.
    word_labels: List[str] = []
    seen = set()
    for words in top_words:
        for word in words:
            if word in index and word not in seen:
                seen.add(word)
                word_labels.append(word)
    per_topic_words = [[w for w in words if w in index] for words in top_words]

    topic_vecs = topic_vectors(model)
    coords = build_projection(
        np.vstack([topic_vecs] + [vectors[index[w]] for w in word_labels]),
        n_components=2,
        random_state=random_state,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        reducer=reducer,
    )
    topic_coords = coords[: len(topic_vecs)]
    word_coords = {w: coords[len(topic_vecs) + i] for i, w in enumerate(word_labels)}

    # ---- draw lines and markers (labels come later, once the canvas is fixed) --
    n_topics = len(topic_coords)
    colors = _palette(n_topics, cmap)
    line_alpha = 0.12 if n_topics <= 10 else 0.07
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    word_colors: Dict[str, object] = {}
    for k in range(n_topics):
        color = colors[k]
        txy = topic_coords[k]
        for word in per_topic_words[k]:
            wxy = word_coords[word]
            word_colors.setdefault(word, color)
            ax.plot([txy[0], wxy[0]], [txy[1], wxy[1]],
                    color=color, alpha=line_alpha, linewidth=0.8, zorder=1)
        ax.scatter([word_coords[w][0] for w in per_topic_words[k]],
                   [word_coords[w][1] for w in per_topic_words[k]],
                   color=color, s=30, alpha=0.8, edgecolors="white",
                   linewidths=0.4, zorder=2)
        ax.scatter(txy[0], txy[1], color=color, s=350, marker="*",
                   edgecolors="black", linewidths=1.0, zorder=4)

    # ---- now labels, with the renderer available for real bounding boxes ----
    # Fix the layout *before* placing labels: tight_layout rescales the axes, and
    # text does not scale with it, which would reintroduce overlaps after the fact.
    ax.set_title("UMAP 2D Semantic Space with Topics",
                 fontsize=16, fontweight="bold", pad=18)
    ax.axis("off")
    fig.tight_layout()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    occupied: list = []
    if adjust_labels:
        _draw_labels(
            ax, fig, renderer,
            topic_coords=topic_coords,
            topic_names=[f"Topic_{k + 1}" for k in range(n_topics)],
            topic_colors=colors,
            word_coords=word_coords,
            word_colors=word_colors,
            per_topic_words=per_topic_words,
            mwes=mwes,
            show_mwes=show_mwes,
            occupied=occupied,
            dpi=dpi,
            label_size=8.0,
            topic_label_size=11.0,
        )
    else:                                   # legacy behaviour, for comparison
        for k in range(n_topics):
            txy, color = topic_coords[k], colors[k]
            for word in per_topic_words[k]:
                wxy = word_coords[word]
                ax.text(wxy[0], wxy[1] + 0.08, word, fontsize=8,
                        color="#333333", ha="center", va="bottom", zorder=3)
            ax.text(txy[0], txy[1] - 0.15, f"Topic_{k + 1}", fontsize=11,
                    fontweight="bold", ha="center", va="top", zorder=5,
                    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2))

    path = Path(save_path)
    fig.savefig(path, format="png", bbox_inches="tight", dpi=dpi)
    if show:
        plt.show()
    plt.close(fig)
    print(f"  figure saved -> {path}")
    return path
