# -*- coding: utf-8 -*-
"""
gen_fig_topics2d.py
Classic 2D document-level topic map for the paper.
Fits SCPTM K=4 on 20NG (4 balanced categories), projects docs via UMAP,
colors by dominant topic, labels cluster centroids.
"""

import os, sys, pickle, numpy as np, matplotlib.pyplot as plt, matplotlib.patches as mpatches
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, "/Users/alemeneghini/Dropbox/stat/CPTM/260521_SCPTM/scptm")

from pathlib import Path
from collections import defaultdict
from sklearn.datasets import fetch_20newsgroups

EDGE_CACHE  = Path("benchmark_cache/demo_20ng.pkl")        # preprocessed graph data
MODEL_CACHE = Path("benchmark_cache/demo_20ng_model.pkl")  # fitted SCPTM model
OUT_PNG  = Path("/Users/alemeneghini/Dropbox/stat/CPTM/q&q/paper_data/fig_topics2d.png")
OUT_PDF  = Path("/Users/alemeneghini/Dropbox/stat/CPTM/q&q/paper_data/fig_topics2d.pdf")

CATS     = ["sci.space", "rec.sport.hockey", "talk.politics.guns", "comp.graphics"]
CAT_NICE = ["Space", "Ice Hockey", "Firearms/Politics", "Computer Graphics"]
K        = 4
N_PER    = 150

# ── 1. Load 20NG sample ──────────────────────────────────────────────────────

ng = fetch_20newsgroups(subset="all", categories=CATS,
                        remove=("headers", "footers", "quotes"))
by_cat = defaultdict(list)
for text, label in zip(ng.data, ng.target):
    if len(text.split()) >= 30:
        by_cat[label].append(text)

docs, true_labels = [], []
for cat_id in range(K):
    sample = by_cat[cat_id][:N_PER]
    docs   += sample
    true_labels += [cat_id] * len(sample)
print(f"Loaded {len(docs)} documents from {K} categories")

# ── 2. Fit SCPTM ─────────────────────────────────────────────────────────────

if MODEL_CACHE.exists():
    print(f"Loading fitted model from {MODEL_CACHE}")
    with open(MODEL_CACHE, "rb") as f:
        model = pickle.load(f)
    theta = model._theta.numpy()
else:
    print("Fitting SCPTM (K=4, filtered, 40 epochs) …")
    from scptm import SCPTM
    model = SCPTM(num_topics=K, graph_mode="filtered", lang="eng",
                  epochs=40, batch_size=128, lr=5e-3,
                  min_df=3, max_features=8000,
                  random_state=42, metrics_every_n_epochs=40)
    theta = model.fit_transform(docs, edge_cache_path=str(EDGE_CACHE))
    theta = model._theta.numpy()
    with open(MODEL_CACHE, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {MODEL_CACHE}")

print(f"theta shape: {theta.shape}")

# ── 3. Topic labels ──────────────────────────────────────────────────────────

topics_dict = model.get_topics_dict(top_k=5)
topic_labels = {}
for k_idx, (t_name, t_data) in enumerate(topics_dict.items()):
    words  = t_data.get("single", [])[:3]
    phrases = t_data.get("phrases", [])[:1]
    kw = (phrases + words)[:3]
    topic_labels[k_idx] = "\n".join(kw) if kw else f"Topic {k_idx+1}"
print("Topic labels:")
for k, v in topic_labels.items():
    print(f"  {k}: {v!r}")

# ── 4. SBERT embeddings for segments ─────────────────────────────────────────
# SCPTM uses 1469 segments from 600 docs. Load pre-computed embeddings from cache.

import pickle as _pkl
with open(EDGE_CACHE, "rb") as _f:
    _cache = _pkl.load(_f)
if "doc_embs" in _cache:
    import torch as _torch
    embs = _cache["doc_embs"]
    if hasattr(embs, "numpy"):
        embs = embs.numpy()
    print(f"Loaded doc_embs from edge cache: {embs.shape}")
else:
    print("Encoding segments with SBERT …")
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    embs  = sbert.encode(model._corpus, show_progress_bar=True, batch_size=64)
    print(f"Embeddings: {embs.shape}")

# ── 5. UMAP 2D ───────────────────────────────────────────────────────────────

print("Running UMAP 2D …")
import umap
reducer = umap.UMAP(n_components=2, n_neighbors=20, min_dist=0.1,
                    metric="cosine", random_state=42)
xy = reducer.fit_transform(embs)         # (n_segments, 2)

# ── 6. Dominant topic assignment ──────────────────────────────────────────────

dom_topic = np.argmax(theta, axis=1)     # (n_segments,) — same length as xy
dom_prob  = theta[np.arange(len(theta)), dom_topic]

# Build cluster centroids in 2D
centroids = {}
for k in range(K):
    mask = dom_topic == k
    if mask.sum() > 0:
        centroids[k] = xy[mask].mean(axis=0)

# ── 7. Plot ───────────────────────────────────────────────────────────────────

PALETTE = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800"]
LABEL_COLORS = ["white", "white", "white", "white"]

fig, ax = plt.subplots(figsize=(9, 7), dpi=200)
fig.patch.set_facecolor("white")
ax.set_facecolor("#F9F9F9")

# scatter all docs, coloured by dominant topic
for k in range(K):
    mask = dom_topic == k
    ax.scatter(xy[mask, 0], xy[mask, 1],
               c=PALETTE[k], s=18, alpha=0.55,
               edgecolors="none", rasterized=True, zorder=2,
               label=f"Topic {k+1}")

# centroid labels
for k, (cx, cy) in centroids.items():
    lbl = topic_labels.get(k, f"T{k+1}")
    ax.scatter(cx, cy, c=PALETTE[k], s=280, marker="D",
               edgecolors="black", linewidths=0.8, zorder=4)
    ax.annotate(lbl,
                xy=(cx, cy), xytext=(0, 14),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=8.5,
                fontweight="bold", color="#222222",
                bbox=dict(boxstyle="round,pad=0.25",
                          fc="white", ec=PALETTE[k],
                          linewidth=1.2, alpha=0.9),
                zorder=5)

ax.set_title("SCPTM — 20 Newsgroups (K = 4)", fontsize=13, fontweight="bold", pad=10)
ax.set_xlabel("UMAP dim 1", fontsize=10)
ax.set_ylabel("UMAP dim 2", fontsize=10)
ax.tick_params(labelsize=8)
ax.grid(True, alpha=0.25, linewidth=0.5)
for spine in ax.spines.values():
    spine.set_linewidth(0.5)
    spine.set_color("#CCCCCC")

handles = [mpatches.Patch(color=PALETTE[k], label=f"Topic {k+1}: {topic_labels.get(k,'')[:30]}")
           for k in range(K)]
ax.legend(handles=handles, loc="lower left", fontsize=7.5,
          framealpha=0.9, edgecolor="#CCCCCC")

plt.tight_layout()
fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")
print(f"Saved → {OUT_PNG}")
print(f"Saved → {OUT_PDF}")
plt.close()
