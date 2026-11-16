"""
qualitative_comparison.py
--------------------------
Runs all topic models on the novels corpus and produces:
  - topic_words_<model>.png  : grid of top-10 words per topic (one fig per model)
  - topic_words_all.png      : combined overview (top-5 words per topic, all models)

Models compared
---------------
  CTM           SCPTM graph_mode="none"        (VAE baseline)
  TriTopic-like SCPTM graph_mode="none" + refine
  SCPTM         SCPTM graph_mode="filtered"
  SCPTM+refine  SCPTM graph_mode="filtered" + refine
  BERTopic      standard BERTopic, CountVectorizer with English stop words
  TriTopic      native TriTopic (Leiden clustering + iterative refinement)
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from tritopic import TriTopic

from scptm import SCPTM, SCPTMConfig
from scptm.evaluation import compute_npmi_coherence, compute_topic_diversity
from scptm.graph import prepare_corpus

# ── Config ───────────────────────────────────────────────────────────────────
NOVELS_PATH     = "/Users/alemeneghini/Dropbox/stat/CPTM/10_english_novels"
# Cache lives in scptm/ — resolved relative to this script regardless of cwd
_script_dir = Path(__file__).resolve().parent
CACHE = str(_script_dir / "scptm" / "novels_cache.pkl")
# If you run from inside scptm/: CACHE = "novels_cache.pkl"
K               = 15
TOP_K           = 10          # words per topic in per-model figures
TOP_K_COMBINED  = 5           # words per topic in combined overview
OUT_DIR         = Path("topic_plots")
OUT_DIR.mkdir(exist_ok=True)

BASE = dict(
    num_topics      = K,
    lang            = "eng",
    epochs          = 50,
    apply_chunking  = False,   # corpus already chunked below
    min_df          = 10,
    max_features    = 20_000,
    random_state    = 42,
)

# ── Palette — one colour per model ───────────────────────────────────────────
PALETTE = {
    "CTM":           "#4C72B0",
    "TriTopic-like": "#55A868",
    "SCPTM":         "#C44E52",
    "SCPTM+refine":  "#8172B2",
    "BERTopic":      "#CCB974",
    "TriTopic":      "#64B5CD",
}


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _topic_grid(topics: dict, title: str, colour: str,
                top_k: int = 10, ncols: int = 5,
                figsize_per_cell=(2.8, 2.6)) -> plt.Figure:
    """
    One figure: K panels (one per topic), each listing top-k words.

    Parameters
    ----------
    topics  : {topic_id: [word, ...]}
    title   : figure suptitle (model name)
    colour  : hex colour for topic header background
    """
    ids    = sorted(topics.keys())
    K_plot = len(ids)
    nrows  = math.ceil(K_plot / ncols)
    fw     = figsize_per_cell[0] * ncols
    fh     = figsize_per_cell[1] * nrows + 0.6   # extra for suptitle

    fig, axes = plt.subplots(nrows, ncols, figsize=(fw, fh))
    axes = np.array(axes).flatten()

    for i, tid in enumerate(ids):
        ax    = axes[i]
        words = topics[tid][:top_k]

        # Header band
        ax.add_patch(mpatches.FancyBboxPatch(
            (0, 0.88), 1, 0.12,
            boxstyle="square,pad=0",
            transform=ax.transAxes,
            color=colour, zorder=2, clip_on=False,
        ))
        ax.text(0.5, 0.94, f"Topic {tid}",
                transform=ax.transAxes,
                ha="center", va="center",
                fontsize=9, fontweight="bold", color="white", zorder=3)

        # Word list
        body = "\n".join(f"{j+1:2d}. {w}" for j, w in enumerate(words))
        ax.text(0.05, 0.84, body,
                transform=ax.transAxes,
                va="top", ha="left",
                fontsize=7.5, fontfamily="monospace",
                linespacing=1.45)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Hide unused panels
    for ax in axes[K_plot:]:
        ax.axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.patch.set_facecolor("#F8F8F8")
    fig.tight_layout()
    return fig


def plot_combined(all_topics: dict, top_k: int = 5) -> plt.Figure:
    """
    One figure: rows = models, columns = topics.
    Each cell shows top-k words.  Topics numbered 1…K.
    """
    model_names = list(all_topics.keys())
    n_models    = len(model_names)
    topic_ids   = sorted(next(iter(all_topics.values())).keys())
    n_topics    = len(topic_ids)

    cell_w, cell_h = 2.1, 1.5
    fig, axes = plt.subplots(
        n_models, n_topics,
        figsize=(cell_w * n_topics, cell_h * n_models + 0.5),
    )
    if n_models == 1:
        axes = axes[np.newaxis, :]

    # Column headers (topic numbers)
    for j, tid in enumerate(topic_ids):
        axes[0, j].set_title(f"T{tid}", fontsize=8, fontweight="bold", pad=2)

    for i, mname in enumerate(model_names):
        colour  = PALETTE.get(mname, "#888888")
        topics  = all_topics[mname]
        # Row label
        axes[i, 0].text(
            -0.15, 0.5, mname,
            transform=axes[i, 0].transAxes,
            ha="right", va="center",
            fontsize=8, fontweight="bold", color=colour,
            rotation=0,
        )
        for j, tid in enumerate(topic_ids):
            ax    = axes[i, j]
            words = topics.get(tid, [])[:top_k]
            body  = "\n".join(words)
            ax.text(0.05, 0.95, body,
                    transform=ax.transAxes,
                    va="top", ha="left",
                    fontsize=6.5, fontfamily="monospace",
                    linespacing=1.3, color="#222222")
            ax.set_facecolor(colour + "18")   # very light tint
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")

    fig.suptitle("Topic words — qualitative comparison (top 5 per topic)",
                 fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 1. Corpus
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Loading corpus …")
docs = prepare_corpus(
    NOVELS_PATH,
    source_type     = "folder",
    apply_chunking  = True,
    max_chunk_chars = 800,
)
print(f"Corpus: {len(docs):,} segments")


# ─────────────────────────────────────────────────────────────────────────────
# 2. SCPTM models  (embedding + parse cache → fast from run 2 onward)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Fitting SCPTM models …")

# CTM baseline
m_ctm = SCPTM(**BASE, graph_mode="none")
m_ctm.fit_transform(docs, edge_cache_path=CACHE)

# TriTopic-like
m_tril = SCPTM(**BASE, graph_mode="none")
m_tril.fit(docs, edge_cache_path=CACHE,
           iterative_refinement=True, n_refinement_steps=3, refinement_blend=0.2)

# SCPTM filtered
m_scptm = SCPTM(**BASE, graph_mode="filtered")
m_scptm.fit_transform(docs, edge_cache_path=CACHE)

# SCPTM + refine
m_best = SCPTM(**BASE, graph_mode="filtered")
m_best.fit(docs, edge_cache_path=CACHE,
           iterative_refinement=True, n_refinement_steps=3, refinement_blend=0.2)

# Helper: extract top words from a fitted SCPTM model
def scptm_words(model, top_k=TOP_K):
    # get_topic_info() returns DataFrame with columns: topic_id, size, keywords
    # 'keywords' is a comma-separated string — split it back to a list
    info = model.get_topic_info(top_k=top_k)
    out  = {}
    for _, row in info.iterrows():
        tid   = int(row["topic_id"])
        words = [w.strip() for w in row["keywords"].split(",")]
        out[tid] = words[:top_k]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. BERTopic  (with English stop words in the vectorizer)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Fitting BERTopic …")

sbert = SentenceTransformer("all-MiniLM-L6-v2")
bt_vectorizer = CountVectorizer(
    stop_words  = "english",
    min_df      = 10,
    max_features= 20_000,
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
)
bt = BERTopic(
    embedding_model        = sbert,
    nr_topics              = K,
    vectorizer_model       = bt_vectorizer,
    language               = "english",
    calculate_probabilities= True,
    verbose                = True,
)
bt.fit_transform(docs)

def bertopic_words(model, top_k=TOP_K):
    out = {}
    for tid in sorted(model.get_topics().keys()):
        if tid == -1:   # outlier cluster — skip
            continue
        words = [w for w, _ in model.get_topic(tid)[:top_k]]
        out[tid + 1] = words   # shift to 1-based
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. TriTopic native
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Fitting TriTopic …")

tt = TriTopic(
    n_topics           = K,
    embedding_model    = "all-MiniLM-L6-v2",
    mode               = "hybrid",
    iterative          = True,
    random_state       = 42,
    verbose            = True,
)
tt.fit(docs)

def tritopic_words(model, top_k=TOP_K):
    df  = model.get_topic_info()
    out = {}
    for _, row in df.iterrows():
        tid   = int(row["Topic"])
        words = row["All_Keywords"][:top_k]
        out[tid] = words
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 5. Collect all word lists
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Extracting top words …")

all_topics = {
    "CTM":           scptm_words(m_ctm),
    "TriTopic-like": scptm_words(m_tril),
    "SCPTM":         scptm_words(m_scptm),
    "SCPTM+refine":  scptm_words(m_best),
    "BERTopic":      bertopic_words(bt),
    "TriTopic":      tritopic_words(tt),
}

# Quick sanity check — print to console too
for mname, topics in all_topics.items():
    print(f"\n{'─'*50}")
    print(f"  {mname}")
    print(f"{'─'*50}")
    for tid in sorted(topics.keys()):
        words = ", ".join(topics[tid][:8])
        print(f"  Topic {tid:2d}: {words}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Per-model figures
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Plotting …")

for mname, topics in all_topics.items():
    colour = PALETTE[mname]
    fig    = _topic_grid(topics, title=mname, colour=colour, top_k=TOP_K)
    slug   = mname.lower().replace("+", "plus").replace(" ", "_")
    path   = OUT_DIR / f"topic_words_{slug}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#F8F8F8")
    plt.close(fig)
    print(f"  Saved {path}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Combined overview
# ─────────────────────────────────────────────────────────────────────────────
fig_all = plot_combined(all_topics, top_k=TOP_K_COMBINED)
path_all = OUT_DIR / "topic_words_all.png"
fig_all.savefig(path_all, dpi=150, bbox_inches="tight")
plt.close(fig_all)
print(f"  Saved {path_all}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Metrics summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Metrics\n")

vec_eval = CountVectorizer(
    stop_words   = "english",
    min_df       = 10,
    max_features = 20_000,
    token_pattern= r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
)
bow_eval  = vec_eval.fit_transform(docs)
vocab_eval= vec_eval.get_feature_names_out().tolist()

rows = []
for mname, topics in all_topics.items():
    word_lists = [topics[t] for t in sorted(topics.keys())]
    npmi = compute_npmi_coherence(word_lists, bow_eval, vocab_eval)
    div  = compute_topic_diversity(word_lists)
    rows.append({"model": mname,
                 "npmi":      round(npmi, 3),
                 "diversity": round(div,  3)})

df_metrics = pd.DataFrame(rows)
print(df_metrics.to_string(index=False))
print()
