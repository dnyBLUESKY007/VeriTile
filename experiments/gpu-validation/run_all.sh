#!/usr/bin/env bash
# Run the whole phase-1 package and pack results/ into one tar.gz to send back.
#   ./run_all.sh            full run (estimated 5-20 min; not yet measured)
#   ./run_all.sh --quick    smoke run (about 1-3 min)
#   CUDA_VARIANT=cu13 ./run_all.sh   use the CUDA 13 wheels (needs driver >= 580)
set -euo pipefail
cd "$(dirname "$0")"

VARIANT="${CUDA_VARIANT:-cu12}"
QUICK=()
[[ "${1:-}" == "--quick" ]] && QUICK=(--quick)

command -v uv >/dev/null || { echo "uv not found; see README.md 'Prerequisites'" >&2; exit 1; }
mkdir -p results
pack() {  # runs on every exit, so even a failed run leaves something to send back
  OUT="veritile-gpu-validation-results-$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
  tar czf "$OUT" results
  echo
  echo "Send back: $(pwd)/$OUT  plus the full terminal output of this script."
}
trap pack EXIT
{
  echo "date_utc: $(date -u +%FT%TZ)"
  echo "variant: $VARIANT  args: $*"
  uv --version
  git rev-parse HEAD 2>/dev/null || echo "git: not a checkout"
  git status --porcelain -- . 2>/dev/null || true
} > results/run_info.txt

uv sync --locked --extra "$VARIANT" 2>&1 | tee results/uv_sync.log
{ uv tree; uv pip list; } > results/uv_tree.txt 2>&1 || true
PY=(uv run --no-sync python)

"${PY[@]}" check_logic.py 2>&1 | tee results/check_logic.log
"${PY[@]}" litmus.py compile-only 2>&1 | tee results/compile.log    # needs no GPU
"${PY[@]}" env_check.py 2>&1 | tee results/env_check.log          # stops here if no usable GPU

# Pass 1: quiet machine.  Pass 2: NVLink stress traffic plus random per-round delays.
# A failing test is recorded in its JSON and does not stop the run.
"${PY[@]}" litmus.py run all ${QUICK[@]+"${QUICK[@]}"} 2>&1 | tee results/run_plain.log || true
"${PY[@]}" litmus.py run all ${QUICK[@]+"${QUICK[@]}"} --stress nvlink --jitter-ns 2000 --tag stress \
  2>&1 | tee results/run_stress.log || true
"${PY[@]}" litmus.py summarize 2>&1 | tee results/summary.txt || true
