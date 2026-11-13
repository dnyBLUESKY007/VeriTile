#!/usr/bin/env bash
# The gate of record. Run it in a checkout of the commit under review, with the
# Lean toolchain and the comparator on PATH (scripts/README.md).
#
#   scripts/local_gate.sh [--no-lean] [--status]
#
# --no-lean  documents and Python only
# --status   post the outcome on the head commit as the status `local-gate`
#            (coordinator; needs push access to dnyBLUESKY007/VeriTile)
#
# Prints one line per command with its exit code; exits 0 only if all are 0.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

lean=1 status=0
for argument in "$@"; do
  case $argument in
    --no-lean) lean=0 ;;
    --status) status=1 ;;
    *) echo "unknown argument: $argument" >&2; exit 2 ;;
  esac
done

# Upstream's dependency-profile test reads TOML with tomllib (Python 3.11 or newer).
profile_python=python3
python3 -c 'import tomllib' 2>/dev/null || profile_python="uv run --no-project --python 3.11 python"
commands=("python3 scripts/check_docs.py"
          "$profile_python -B -m unittest scripts.test_library_dependency_profile")
for suite in scripts/interp scripts/gpurun scripts/census experiments/interp_gpu/deepep_v1 experiments/interp_gpu/internode_logical; do
  commands+=("python3 -B -m unittest discover -s $suite -p test_*.py")
done
if [ "$lean" = 1 ]; then
  commands+=("bash scripts/check-core-examples.sh")
  for target in VeriTileDist VeriTileTileLang VeriTileInterp VeriTileEquiv; do
    commands+=("python3 scripts/check_fork_target.py $target --replay")
  done
  commands+=("python3 scripts/check_fork_target.py VeriTileStaging"
             "python3 scripts/check_frozen_statements.py"
             "python3 scripts/test_fork_targets.py"
             # Fork-owned regression modules that no target gate builds.
             "lake build tests.lean.InterpReflectSmoke")
fi

log=$(mktemp -d)
failed=0
for index in "${!commands[@]}"; do
  # Unquoted on purpose: each entry is a plain command line without shell syntax.
  ${commands[$index]} >"$log/$index.log" 2>&1
  code=$?
  echo "exit $code  ${commands[$index]}"
  if [ "$code" != 0 ]; then
    failed=$((failed + 1))
    tail -n 20 "$log/$index.log" | sed 's/^/    /'
  fi
done
echo "logs: $log"

if [ "$status" = 1 ]; then
  state=success; [ "$failed" = 0 ] || state=failure
  scope="full"; [ "$lean" = 1 ] || scope="documents and Python only"
  gh api -X POST "repos/dnyBLUESKY007/VeriTile/statuses/$(git rev-parse HEAD)" \
    -f state="$state" -f context=local-gate \
    -f description="${#commands[@]} commands, $failed failed ($scope)" -q .state
fi
[ "$failed" = 0 ]
