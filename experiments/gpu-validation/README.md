[English](README.md) | [中文](README.zh.md)

# VeriTile GPU validation, phase 1: PTX litmus tests on 2x H800

This package runs small **litmus tests** that check how a Hopper (sm_90) GPU pair connected by
NVLink orders memory operations between two GPUs. VeriTile is a Lean verifier for GPU kernels.
Its first version models hardware behaviour that the PTX and NVSHMEM documentation do not
promise as named **trust-base lemmas**. A lemma may be used in proofs only after the runtime
tests that exercise it pass on this kind of machine. These tests produce that evidence.

You do not need to understand the theory to run the tests. One script does everything and
produces one `.tar.gz` file to send back.

## What you need

- A Linux server with **two NVIDIA H800 (or H100) GPUs connected by NVLink**, idle while the
  tests run (no other jobs, MIG disabled).
- An **NVIDIA driver**. Check with `nvidia-smi`:
  - driver **>= 525** for the default CUDA 12 variant (any newer driver also works);
  - driver **>= 580** only if you choose the CUDA 13 variant.
  You do **not** need a CUDA Toolkit. The CUDA compiler (NVRTC) is installed from NVIDIA's
  Python wheel.
- Internet access to `pypi.org`, `github.com` and `astral.sh` (and to `github.com` releases
  for a Python download if Python 3.10-3.12 is not installed).
- About 1 GB of disk for the Python environment.
- `git`, `bash`, `tar`.

## Step 1: install uv

[uv](https://docs.astral.sh/uv/) installs the right Python and the exact package versions in
`uv.lock`. The lock file was created with uv 0.13.1; use that version or newer:

```bash
curl -LsSf https://astral.sh/uv/0.13.1/install.sh | sh
# then open a new shell, or: source $HOME/.local/bin/env
uv --version
```

## Step 2: get the code

```bash
git clone --branch gpu-validation https://github.com/dnyBLUESKY007/VeriTile.git
cd VeriTile/experiments/gpu-validation
```

## Step 3: run

```bash
./run_all.sh --quick     # smoke run first, about 1-3 minutes (estimate)
./run_all.sh             # full run, about 5-20 minutes (estimate, not yet measured)
```

To use the CUDA 13 wheels instead (driver >= 580): `CUDA_VARIANT=cu13 ./run_all.sh`.

If the two NVLink GPUs are not GPU 0 and GPU 1, select them first, for example
`CUDA_VISIBLE_DEVICES=2,3 ./run_all.sh`.

`run_all.sh` does, in order:

1. `uv sync --locked --extra cu12`: creates `.venv/` with Python 3.12, CuPy 14.2.0, NumPy and
   NVRTC exactly as pinned in `uv.lock`.
2. `check_logic.py`: CPU self-check of the bookkeeping code.
3. `litmus.py compile-only`: compiles all 27 test kernels for `sm_90` (no GPU needed) and
   checks that each expected PTX instruction is in the generated PTX.
4. `env_check.py`: records GPU names, compute capability, driver/runtime/NVRTC versions,
   `nvidia-smi`, `nvidia-smi topo -m`, NVLink status and peer access into `results/env.json`.
   The script stops here if two peer-capable GPUs are not available.
5. `litmus.py run all`: every test on a quiet machine.
6. `litmus.py run all --stress nvlink --jitter-ns 2000 --tag stress`: every test again with
   extra blocks generating NVLink traffic and random per-round delays.
7. `litmus.py summarize`: prints a verdict table and writes `results/summary.json`.
8. Packs `results/` into `veritile-gpu-validation-results-<UTC time>.tar.gz`. This also
   happens when a step fails.

Single commands, if needed:

```bash
uv run --no-sync python litmus.py list                    # all tests and what they serve
uv run --no-sync python litmus.py run full_tut01 --rounds 1000000
uv run --no-sync python litmus.py summarize
```

## Step 4: send back

Send **the `.tar.gz` file** and **the full terminal output** of `run_all.sh` (copy it or run
`./run_all.sh 2>&1 | tee run_all.out` and send `run_all.out`). Nothing else is needed when
the run succeeds. The results contain GPU model, driver and PCI bus IDs but no host name or
user name.

## What the tests do

Every test is the slot protocol of Triton-distributed tutorial 01, repeated many times:

```
PRODUCER (one GPU)                         CONSUMER (same or other GPU)
wait flag == "empty" (2k)                  wait flag == "full" (2k+1)    <- "full" edge
write payload = k+1  (1 KiB)               read payload
fence; flag = "full" (2k+1)                fence; flag = "empty" (2k+2) <- "empty" edge
```

The consumer checks the payload it read in round k:

- `k+1`: ok;
- smaller (**stale**): the "full" edge failed, a message-passing violation;
- larger (**future**): the producer overwrote the slot before the consumer's reads were
  satisfied, a load-buffering violation on the "empty" edge.

By default each test runs **64 producer/consumer pairs x 200,000 rounds = 12.8 million
iterations** (`--quick`: 16 x 5,000). Each test changes exactly the PTX instructions it
studies; `kernels.cu` explains every PTX line. Each result file records the exact PTX used.

| Kind | Meaning | Expected result |
|---|---|---|
| `self-test` | Program order deliberately broken | **Must** show violations, or the harness is broken |
| `sanctioned` | PTX documentation forbids the bad outcome | Zero violations |
| `trust-base` | Candidate trust-base lemma (behaviour PTX does not promise) | Zero violations, with its positive controls firing |
| `positive-control` | One step weaker than a trust-base or sanctioned test | Should show violations; shows the test **can** detect them |
| `probe` | Measurement only | Either outcome is informative |

Test groups (`litmus.py list` shows each one):

- **"full" edge, GPU0 -> GPU1, data in the consumer's memory:** `full_tut01` (tutorial 01:
  `membar.sys; st.relaxed.sys` writer, `ld.global.acquire.gpu` reader), `full_tut01_warm`,
  `full_sys` (`ld.global.acquire.sys` control), `full_nvshmem_wait` (`ld.volatile` +
  `membar.gl`, NVSHMEM's implementation), `full_relaxed_sys_fence_gpu`, `full_nccl_volatile`,
  `full_writer_gpu_fence` (probe), positive controls `full_pos_*`, and
  `selftest_read_before_wait`.
- **Same-GPU controls:** `same_tut01`, `same_pos_relaxed`.
- **Remote device-scope poll of a flag in the peer's memory:** `e2_remote_poll_sys_writer`,
  `e2_remote_poll_gpu_writer` (probe), `e2_remote_poll_sys`, `e2_pos_relaxed`.
- **"empty" edge (load buffering):** `empty_tut01`, `empty_sys`, `empty_pos_relaxed`,
  `selftest_notify_before_read`.
- **Tutorial 01 as written, both directions at once:** `tut01_ring`.
- **DeepEP V1 `ld.global.nc.L1::no_allocate.L2::256B` reads of data a peer writes:**
  `nc_deepep`, `nc_deepep_warm`, positive controls `nc_pos_plain`, `nc_pos_plain_warm`.

## How to read the results

`results/summary.txt` has one line per test and pass (`-` = quiet, `stress`):

| Verdict | Meaning |
|---|---|
| `OK (detects)` | Self-test fired: the harness can count violations |
| `HARNESS-BROKEN` | A self-test did not fire: **no verdict in this run can be trusted** |
| `PASS` | Sanctioned test: zero violations |
| `PASS (evidence)` | Trust-base test: zero violations, zero timeouts, every listed positive control fired in the same pass |
| `INCONCLUSIVE (control silent or missing)` | Zero violations, but a positive control never fired, so the test has not shown it could detect the failure |
| `FAIL (violations)` | Forbidden outcomes observed. `first_bad` in the JSON shows examples |
| `INCOMPLETE (timeouts)` | Some waits timed out (default 2^24 polls). Liveness problem or GPUs not idle |
| `FIRED` / `SILENT` | Positive control did or did not show violations |
| `NONE-SEEN` | Probe: zero violations and its control fired |

Each `results/<test>[__stress].json` holds: test name, kind, the lemma or ticket it serves,
exact PTX per knob, iterations planned and completed, the outcome histogram (`ok`, `stale`,
`future`, `mixed`), `bad_count`, timeouts, the first bad rounds, stress and jitter settings,
elapsed time, GPU and driver information, a UTC timestamp, and the package's git commit.

## Pass/fail rule for admitting a trust-base lemma

A lemma is admitted on the basis of this package only if **all** hold for every test that
serves it:

1. All self-tests are `OK (detects)` in the same run.
2. All sanctioned tests are `PASS`. A violation in a sanctioned test means the harness or the
   machine is broken; investigate before admitting anything.
3. The test is `PASS (evidence)` in **both** the quiet and the stress pass, at the default
   scale or larger.

Zero observed violations is **evidence, not proof**: a later run, another driver, or another
workload can still show a violation. A test whose positive control never fires is
**inconclusive**: it has not shown it can see the failure it looks for, so it does not support
the lemma. Any `FAIL` blocks the lemma; kernels that need it stay reported as racy. Record the
test name, commit, GPU, driver, CUDA/NVRTC versions, date and result file with every admitted
lemma.

## Known limits of phase 1

- One process, unified virtual addressing: the producer and the consumer use the **same
  virtual address** for the shared buffer. NVSHMEM and Triton-distributed use separate
  processes with distinct mappings, so phase 1 does not test the PTX alias/proxy question or
  cross-process `.sys` scope. A pass here covers single-process `cudaMalloc` peer memory only.
- Default memory synchronization domain; `cudaMalloc` memory, not the NVSHMEM symmetric heap.
- The SASS was not inspected. ptxas may legally transform weak loads (it deletes unused
  ones; the warm-up loads therefore feed a sink value).
- Positive controls can fire because of the compiler or the hardware. Both are allowed by the
  PTX model, so either way they show that the harness sees weak outcomes.
- `empty_pos_relaxed` may never fire: load buffering may not be observable on this hardware.
  Then `empty_tut01` and `tut01_ring` stay inconclusive for the "empty" edge.

## If something goes wrong

Send the `.tar.gz` (it is created even on failure) and fill in this template:

```text
### What failed
<one line, e.g. "uv sync failed" or "full_tut01 ERROR">

### Full command
<exactly what you typed, including environment variables>

### Full stderr / terminal output
<paste everything, not only the last lines>

### results/env.json
<attach, or paste>

### nvidia-smi
<output of: nvidia-smi ; nvidia-smi topo -m>

### uv
<output of: uv --version ; uv tree>

### Anything unusual
<other jobs on the GPUs, containers, proxies, CUDA_VISIBLE_DEVICES, ...>
```

Common problems:

- `cudaErrorInsufficientDriver`: the driver is older than the variant needs. Use the default
  `cu12` variant, or report the driver version.
- `cannot access GPU 1 (no P2P)`: the two GPUs are not peer-capable; check
  `nvidia-smi topo -m` and pick an NVLink pair with `CUDA_VISIBLE_DEVICES`.
- `uv sync --locked` fails: report it; do not edit `uv.lock` or `pyproject.toml`.
- Many `INCOMPLETE (timeouts)`: make sure no other job uses the GPUs and rerun.
