"""
examples/demo.py
================
End-to-end demo on a synthetic corpus — no download, no model, no network
needed beyond the parser model you already have.

Usage::

    python examples/demo.py            # English, spaCy
    python examples/demo.py ita        # Italian, spaCy
"""

import sys

import numpy as np

from scptm_bow import BowSCPTM

LANG = sys.argv[1] if len(sys.argv) > 1 else "eng"

POOLS = {
    "space": "rocket orbit launch satellite mission astronaut nasa propulsion trajectory payload".split(),
    "hockey": "hockey puck goal playoff team player score ice rink season period".split(),
    "religion": "faith prayer church god belief scripture sacred worship sin soul holy".split(),
    "graphics": "graphics render pixel shader texture polygon image color display resolution".split(),
}

rng = np.random.default_rng(0)
documents = [
    " ".join(rng.choice(pool, size=35))
    for _ in range(50)
    for pool in POOLS.values()
]

print(f"demo corpus: {len(documents)} documents, 4 hidden word pools, lang={LANG}\n")

model = BowSCPTM(
    num_topics=4,
    lang=LANG,
    min_df=2,
    max_features=500,
    nmf_max_iter=100,
    vector_backend="fasttext",
    vector_dim=50,
    fasttext_epochs=5,
)
model.fit(documents, cache_path=f"demo_{LANG}.pkl")

print("\n--- metrics (with graphs) ---")
for key, value in model.evaluate(top_k=6).items():
    print(f"  {key:<12} {value:+.3f}")

print("\n--- ablation: same model without the graph penalties ---")
for key, value in model.evaluate_no_syntax(top_k=6).items():
    print(f"  {key:<12} {value:+.3f}")

print("\n--- topics ---")
for row in model.get_topic_info(top_k=6):
    print(f"  Topic {row['topic_id']}  (n={row['size']})")
    print(f"    words: {row['top_words']}")
    if row["mwes"]:
        print(f"    MWEs : {row['mwes']}")

print("\n--- out-of-sample fold-in ---")
new_docs = [" ".join(rng.choice(POOLS["hockey"], size=30)) for _ in range(3)]
inferred = model.transform(new_docs)
for i, theta in enumerate(inferred):
    print(f"  doc {i}: topic {int(theta.argmax()) + 1} (p={theta.max():.2f})")

print("\nMWEs are extracted from the dependency triples collected during the "
      "single parse pass — no extra parsing, no re-training.")
