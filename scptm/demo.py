"""
demo.py
-------
End-to-end demo of SCPTM on a stratified sample of the 20 Newsgroups dataset.
Run with:   python demo.py
"""

# ============================================================
# 0. Imports
# ============================================================
import numpy as np
from sklearn.datasets import fetch_20newsgroups

from scptm import SCPTM, SCPTMConfig, SCPTMEvaluator

# ============================================================
# 1. Load a stratified sample from 20 Newsgroups
# ============================================================
CATEGORIES = [
    "sci.med",
    "sci.space",
    "comp.graphics",
    "rec.sport.hockey",
    "talk.politics.guns",
]
N_PER_CATEGORY = 80   # total corpus: 400 documents

print("Loading 20 Newsgroups dataset...")
raw = fetch_20newsgroups(
    subset="train",
    categories=CATEGORIES,
    remove=("headers", "footers", "quotes"),
    random_state=42,
)

# Stratified sample: N_PER_CATEGORY documents per class
rng = np.random.default_rng(42)
indices = np.concatenate([
    rng.choice(np.where(raw.target == k)[0], size=N_PER_CATEGORY, replace=False)
    for k in range(len(CATEGORIES))
])
indices = rng.permutation(indices)

DOCUMENTS  = [raw.data[i] for i in indices]
TRUE_LABELS = raw.target[indices]

print(f"Corpus: {len(DOCUMENTS)} documents across {len(CATEGORIES)} categories")
print(f"Categories: {', '.join(CATEGORIES)}\n")


# ============================================================
# 2. Base configuration
# ============================================================
BASE_CFG = dict(
    num_topics             = len(CATEGORIES),
    epochs                 = 40,
    graph_mode             = "filtered",
    bow_normalization      = "tf",
    topic_diversity_weight = 0.1,
    kl_strategy            = "linear",
    metrics_every_n_epochs = 10,
    min_df                 = 3,
    # Chunking disabled: newsgroups posts are already short after header/footer
    # removal, and keeping it off ensures len(corpus) == len(TRUE_LABELS).
    apply_chunking         = False,
    use_mixed_precision    = False,   # set True on GPU
)


# ============================================================
# 3. Demo 1 — Basic fit + evaluate (with parse cache)
# ============================================================
print("=" * 60)
print("DEMO 1 — Basic fit + evaluate  (edge_cache_path enabled)")
print("=" * 60)

cfg = SCPTMConfig(**BASE_CFG)
model = SCPTM(config=cfg)

# edge_cache_path: spaCy parsing is saved after the first run.
# Re-running demo.py will skip the two spaCy passes entirely.
theta = model.fit_transform(
    DOCUMENTS,
    source_type    = "list",
    edge_cache_path= "newsgroups_edges.pkl",
)

print("\nTopic overview:")
info = model.get_topic_info(top_k=8)

print("\nEvaluation:")
metrics = model.evaluate(true_labels=TRUE_LABELS)
print(metrics)


# ============================================================
# 4. Demo 2 — Keyword methods: cosine vs c-TF-IDF
# ============================================================
print("\n" + "=" * 60)
print("DEMO 2 — Keyword ranking: cosine vs c-TF-IDF")
print("=" * 60)

print("\n[cosine similarity]")
cosine_info = model.get_topic_info(top_k=8, method="cosine")

print("\n[c-TF-IDF]")
ctfidf_info = model.get_topic_info(top_k=8, method="ctfidf")


# ============================================================
# 5. Demo 3 — Out-of-sample transform
# ============================================================
print("\n" + "=" * 60)
print("DEMO 3 — Out-of-sample transform")
print("=" * 60)

new_docs = [
    "Astronomers discovered a new exoplanet orbiting a distant star using infrared telescopes.",
    "The hockey team secured a playoff spot after a dramatic overtime victory last night.",
    "Patients with chronic pain may benefit from novel analgesic drugs targeting opioid receptors.",
    "Rendering 3D scenes in real time requires efficient GPU-accelerated rasterisation pipelines.",
    "Gun control legislation is being debated in the Senate amid rising concerns over public safety.",
]
new_theta = model.transform(new_docs)
print(f"\n{'Doc':>3}  {'Topic':>7}  {'Prob':>5}  Preview")
print("-" * 70)
for i, doc in enumerate(new_docs):
    dominant = new_theta[i].argmax().item() + 1
    prob     = new_theta[i].max().item()
    print(f"{i+1:>3}  Topic_{dominant:<2}  {prob:.2f}  {doc[:60]}...")


# ============================================================
# 6. Demo 4 — Monte Carlo uncertainty
# ============================================================
print("\n" + "=" * 60)
print("DEMO 4 — MC uncertainty (n_mc_samples=10)")
print("=" * 60)

cfg_mc = SCPTMConfig(**{**BASE_CFG, "n_mc_samples": 10, "epochs": 20})
model_mc = SCPTM(config=cfg_mc)
model_mc.fit(
    DOCUMENTS, source_type="list",
    edge_cache_path="newsgroups_edges.pkl",   # cache already exists → instant skip
)
unc_df = model_mc.get_uncertainty_report()
print(unc_df[["doc_id", "regime", "dominant_topic", "dominant_prob"]].head(12))


# ============================================================
# 7. Demo 5 — Save, reload, and verify edge persistence
# ============================================================
print("\n" + "=" * 60)
print("DEMO 5 — Save and reload (full edge graph preserved)")
print("=" * 60)

model.save("scptm_newsgroups.pkl")
loaded = SCPTM.load("scptm_newsgroups.pkl")

new_theta_loaded = loaded.transform(new_docs)
match = np.allclose(new_theta.numpy(), new_theta_loaded.numpy(), atol=1e-5)
print(f"Transform output matches after reload: {match}")
edge_count = loaded._graph_data["doc", "contains", "word"].edge_index.shape[1]
print(f"Doc-word edges restored in loaded model: {edge_count:,}")


# ============================================================
# 8. Demo 6 — Ablation study (graph modes)
# ============================================================
print("\n" + "=" * 60)
print("DEMO 6 — Ablation study (all graph modes)")
print("=" * 60)

ablation_df = SCPTM.run_ablation_study(
    DOCUMENTS,
    epochs          = 20,
    num_topics      = len(CATEGORIES),
    min_df          = 3,
    use_mixed_precision = False,
)
print(ablation_df)


# ============================================================
# 9. Demo 7 — Iterative refinement
# ============================================================
print("\n" + "=" * 60)
print("DEMO 7 — Iterative refinement")
print("=" * 60)

cfg_ref = SCPTMConfig(**{**BASE_CFG, "epochs": 15})
model_ref = SCPTM(config=cfg_ref)
model_ref.fit(
    DOCUMENTS, source_type="list",
    iterative_refinement = True,
    n_refinement_steps   = 2,
    refinement_blend     = 0.2,
    edge_cache_path      = "newsgroups_edges.pkl",
)
print("\nTopics after iterative refinement (c-TF-IDF):")
model_ref.get_topic_info(top_k=6, method="ctfidf")

print("\nAll demos complete.")
