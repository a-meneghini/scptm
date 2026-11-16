#!/usr/bin/env bash
# sync_and_rerun.sh
# ------------------
# Push the SCPTM repo to a remote (SSH) host, wipe stale parse caches so
# the graph-fix + syntactic-MWE changes actually take effect, reinstall
# the package + any missing deps, and launch the benchmark(s) in the
# background so they survive an SSH disconnect.
#
# Fill in the two variables below, then run:  bash sync_and_rerun.sh
set -euo pipefail

# ---- fill these in -----------------------------------------------------
REMOTE_HOST="alessandro.meneghini@158.110.146.243"              # SSH alias or user@host
REMOTE_DIR_INPUT="~/scptm"                  # project root on the remote (~ ok here)
REMOTE_VENV_INPUT="~/scptm_env"             # existing venv — sibling dir, NOT ~/scptm/.venv
# Parse caches (*cache*.pkl) hold hours of spaCy syntactic-parsing work per
# (corpus, graph_mode), reused across every K x seed combination — do NOT
# wipe them on routine re-syncs, or you re-pay that cost every single time.
# Only flip this to true for a run right after a change to graph.py's cache
# schema itself (new fields written/read from the pickle, e.g. the
# dep_triples addition earlier this session) — a plain code fix elsewhere
# (benchmark_paper.py, config.py, model.py, ...) does NOT need this.
WIPE_CACHES=false
# --------------------------------------------------------------------------

LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve ~ against the remote's actual $HOME once, up front, into an
# absolute path. Tilde expansion only happens on literal unquoted text at
# shell-parse time — a `~` sitting inside a variable's value never expands
# no matter how it's quoted downstream, and single-quoting it (needed to
# safely pass paths to `find`/etc. over ssh) blocks expansion outright.
# Resolving once here avoids that whole class of bug for every step below.
echo "==> [0/5] Resolving remote paths"
REMOTE_DIR="$(ssh "${REMOTE_HOST}" "echo ${REMOTE_DIR_INPUT}")"
REMOTE_VENV="$(ssh "${REMOTE_HOST}" "echo ${REMOTE_VENV_INPUT}")"
if [ -z "${REMOTE_DIR}" ] || [ -z "${REMOTE_VENV}" ]; then
  echo "Could not resolve REMOTE_DIR_INPUT/REMOTE_VENV_INPUT on ${REMOTE_HOST} — check REMOTE_HOST and the two _INPUT vars."
  exit 1
fi
echo "    project: ${REMOTE_DIR}"
echo "    venv:    ${REMOTE_VENV}"
ssh "${REMOTE_HOST}" "test -f '${REMOTE_VENV}/bin/activate'" || {
  echo "No activate script at ${REMOTE_VENV}/bin/activate — check REMOTE_VENV_INPUT."
  exit 1
}

# Scripts to run remotely, in order. benchmark_paper.py is the main 4-corpus
# sweep; the exp_c3*/exp_c4*/gen_fig_* scripts are downstream analyses that
# read from specific benchmark_paper.py result folders (e.g. "results_v3")
# — uncomment only the ones whose expected input actually exists after this
# run, otherwise they'll fail looking for a results folder that isn't there.
RUN_SCRIPTS=(
  "benchmark_paper.py"
  # "exp_c3a_eu_validation.py"
  # "exp_c3a_vader_validation.py"
  # "exp_c3b_eu_comparison.py"
  # "exp_c3b_pmi_comparison.py"
  # "exp_c4_tfidf_concentration.py"
  # "exp_c4_undebates_only.py"
  # "gen_fig_topics2d.py"
)

echo "==> [1/5] Syncing code to ${REMOTE_HOST}:${REMOTE_DIR}"
rsync -avz --progress \
  --exclude='.venv/' \
  --exclude='__pycache__/' \
  --exclude='.git/' \
  --exclude='*.pyc' \
  --exclude='*.pkl' \
  --exclude='*.sbert.npy' \
  --exclude='benchmark_cache/results*/' \
  "${LOCAL_DIR}/" "${REMOTE_HOST}:${REMOTE_DIR}/"

if [ "${WIPE_CACHES}" = "true" ]; then
  echo "==> [2/5] Wiping stale parse/edge caches on remote (WIPE_CACHES=true)"
  # Scoped to *cache*.pkl only: SBERT embeddings (*.sbert.npy) and results
  # CSVs are untouched.
  ssh "${REMOTE_HOST}" "find '${REMOTE_DIR}' -type f -iname '*cache*.pkl' \
    -not -path '*/results*/*' -print -delete"
else
  echo "==> [2/5] Skipping cache wipe (WIPE_CACHES=false) — reusing existing parse caches"
fi

echo "==> [3/5] Reinstalling package on remote"
# python -m pip (not bare `pip`) — after activation, `python` reliably points
# at the venv's interpreter even if the venv's own pip binary is missing or
# broken, which would otherwise silently fall through PATH to system pip
# (system Python, wrong wheel tags — exactly the thinc/spacy build failure
# this fix avoids).
ssh "${REMOTE_HOST}" "cd '${REMOTE_DIR}' && \
  source '${REMOTE_VENV}/bin/activate' && \
  python -m pip install -e . -q && \
  python -m pip install psutil -q"

echo "==> [4/5] Checking neighbor-sampling deps (pyg-lib / torch-sparse)"
# Large corpora now auto-enable use_neighbor_sampling (see config.py
# neighbor_sampling_edge_threshold); NeighborLoader hard-requires one of
# these two packages or training raises ImportError instead of running.
ssh "${REMOTE_HOST}" "cd '${REMOTE_DIR}' && source '${REMOTE_VENV}/bin/activate' && \
  python -c 'import pyg_lib' 2>/dev/null || python -c 'import torch_sparse' 2>/dev/null || \
  { echo '  Neither pyg-lib nor torch-sparse found — installing pyg-lib.'; \
    TORCH_VER=\$(python -c 'import torch; print(torch.__version__.split(\"+\")[0])'); \
    CUDA_TAG=\$(python -c 'import torch; print(\"cu\"+torch.version.cuda.replace(\".\",\"\") if torch.cuda.is_available() else \"cpu\")'); \
    echo \"  Installing from https://data.pyg.org/whl/torch-\${TORCH_VER}+\${CUDA_TAG}.html\"; \
    python -m pip install pyg-lib -f https://data.pyg.org/whl/torch-\${TORCH_VER}+\${CUDA_TAG}.html || \
    echo '  pyg-lib install failed (see output above) — check https://data.pyg.org/whl/ manually and install torch-sparse instead.'; \
    python -c 'import pyg_lib' 2>/dev/null && echo '  pyg_lib import OK.' || \
    echo '  WARNING: pyg_lib still not importable after install attempt — large-graph SCPTM runs will crash on this remote until fixed.'; \
  }"

echo "==> [5/5] Launching benchmark run(s) in background"
TS="$(date +%Y%m%d_%H%M%S)"
for script in "${RUN_SCRIPTS[@]}"; do
  echo "    -> ${script}"
  # ssh -f: backgrounds the ssh client itself right after auth, so this call
  # returns immediately instead of waiting for the session/channel to close —
  # which, even with the remote command fully nohup'd and redirected, ssh can
  # still block on until the backgrounded process itself exits. -f is the
  # standard fix for "launch a detached remote job without hanging locally".
  # stdin also explicitly redirected (< /dev/null): nohup alone only blocks
  # SIGHUP, it doesn't detach stdin from the session on its own.
  ssh -f "${REMOTE_HOST}" "cd '${REMOTE_DIR}' && source '${REMOTE_VENV}/bin/activate' && \
    nohup python '${script}' < /dev/null > 'run_${script%.py}_${TS}.log' 2>&1 &"
done

echo
echo "Done. Monitor with:"
echo "  ssh ${REMOTE_HOST} 'tail -f ${REMOTE_DIR}/run_*_${TS}.log'"
echo
echo "Pull results back once finished:"
echo "  rsync -avz ${REMOTE_HOST}:${REMOTE_DIR}/benchmark_cache/results*/ ${LOCAL_DIR}/benchmark_cache/"
