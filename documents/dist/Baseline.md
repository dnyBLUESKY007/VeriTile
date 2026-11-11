# Build baseline of the fork

Recorded on 2026-11-11 for ticket P1.1. It states what builds on the CPU host
before any fork code is added, how long it takes and how much disk one
checkout needs.

## What was built

| Item | Value |
| --- | --- |
| Commit | fork `main` at `b64043226` |
| Upstream base | `84e769e4c`; the five commits on top change only four files under `documents/` |
| Ancestry | `16abbb7` is an ancestor of `84e769e`, and `84e769e` of fork `main` (`git merge-base --is-ancestor`, both exit 0) |
| Host | `dsp5`: 96 cores (AMD EPYC 9V74), 377 GB memory, no GPU |
| Toolchain | `leanprover/lean4:v4.29.0`, Lake 5.0.0, elan 4.2.4, Mathlib `v4.29.0` |
| Tool location | `~/.elan/bin`; it is not on the default `PATH` |

Fork `main` was already fast-forwarded to `84e769e` on 2026-11-10 and has
since gained documentation commits by merge, so it no longer reads back as
`84e769e` itself. The Lean sources of the two commits are identical.

## Results

Every command ran from the repository root with
`PATH=$HOME/.local/veritile-proof-tools/bin:$HOME/.elan/bin:$PATH` and was
judged by its exit code.

| Gate | Command | Exit | Result |
| --- | --- | --- | --- |
| Build, default target | `lake build` | 0 | 3807 jobs |
| Build, complete shared library | `lake build VeriTile VeriTileMath VeriTileFull` | 0 | 3878 jobs, no `sorry` warning |
| Judge prerequisites | `python3 scripts/check_comparator.py --check-tools` | 0 | comparator, `lean4export`, `landrun` and the systemd user sandbox are available |
| Artifact gate | `bash scripts/check-artifact.sh` | 0 | every check passes |
| Comparator replay | `python3 scripts/check_comparator.py --library` (inside the artifact gate) | 0 | 118 theorems of the proven library manifest accepted |
| Axiom whitelist | `python3 scripts/source_axioms.py VeriTile` against `scripts/artifact-axiom-whitelist.txt` (inside the artifact gate) | 0 | axioms match the whitelist |

No upstream target that was built fails. The baseline did not build
`FPFrontendRegressions`, `TritonBenchSpecExamples`, `BugDetection`,
`PretrainKernels`, `TritonBenchGFinal` or `FinalRegressions`, and it did not
run the corpus audit `scripts/bench/audit_tritonbench_g.sh`.

## Build time

| Build | Wall clock |
| --- | --- |
| `lake build VeriTile VeriTileMath VeriTileFull`, project modules only, Mathlib already built | 11 min 16 s |
| `lake build` after `lake clean`, which also removed the Mathlib build, so 3003 Mathlib modules were compiled from source | 19 min 12 s |
| `lake build VeriTile VeriTileMath VeriTileFull` right after the previous line | 2 min 59 s |
| `bash scripts/check-artifact.sh` on a built tree | 2 min 0 s |

`lake clean` without a package name removes the build directory of every
package, including Mathlib. Use `lake clean veritile` to clean only this
project. A new checkout should fetch the Mathlib build with
`lake exe cache get` instead of compiling it.

## Disk use per checkout

| Part | Size |
| --- | --- |
| Tracked sources | 0.7 GB |
| `.lake/packages` (Mathlib and its dependencies, built) | 5.7 GB |
| `.lake/build` (this project, complete shared library) | 1.4 GB |
| Total | 7.8 GB |

A comparator run copies `.lake` into a temporary judge directory and removes
it afterwards, so it needs about 7 GB more while it runs. The host had 329 GB
free after the baseline.

## Judge tools

The pinned tools were built with
`scripts/setup-comparator.sh $HOME/.local/veritile-proof-tools` (comparator
`2a00b30d`, landrun `5283024a`, Go 1.24.13 from `~/.local/go`). The sandbox
needs a systemd user session; lingering is off on this host, so it works only
while the user is logged in.
