# GPU exchange protocol v1

Status: agreed on 2026-11-10 (sub-issue I9.1). Operating instructions for the GPU-side agent are in [GpuRemoteAgentInstructions.md](GpuRemoteAgentInstructions.md).

Purpose: the development machine (no GPU) asks an sm_90 (Hopper) server to run kernels and gets raw results back, automatically. GPU results are empirical evidence only; they never define semantics.

## 1. Roles

| Role | Does | Never does |
| --- | --- | --- |
| Dev side (requester and judge) | Writes requests, regenerates inputs, compares, posts the verdict | Trust a verdict computed on the GPU side |
| GPU side (listening agent on the server) | Checks out the pinned commit, runs `gpurun`, returns raw results | Edit kernels or requests, fix failures, retry with changed settings, write PASS / FAIL |

## 2. Carrier: the fork `dnyBLUESKY007/VeriTile`

- **Request** = one directory `experiments/interp_gpu/batches/<batch-id>/` holding `batch.json`, committed on any branch of the fork, plus one issue that announces it.
- **Announcement issue**: title `[gpu] <batch-id>`, label `gpu:request`, body with exactly these machine-read lines:

  ```
  protocol: 1
  commit: <full 40-hex SHA of the fork>
  batch: experiments/interp_gpu/batches/<batch-id>
  ```

- **Result** = one directory `results/<batch-id>/` committed to the branch `gpu-results` (orphan branch, never merged into `main`). The GPU side comments the result commit SHA on the issue.
- **Verdict** = a comment by the dev side with the comparison report; then the issue is closed.

## 3. Issue states (labels)

| Label | Set by | Meaning |
| --- | --- | --- |
| `gpu:request` | Dev | Ready to run |
| `gpu:running` | GPU | Claimed; replaces `gpu:request` |
| `gpu:result` | GPU | Results pushed; comment carries the result commit |
| `gpu:failed` | GPU | Could not run at all (environment, checkout, launcher); comment says why |
| `gpu:judged` | Dev | Verdict posted; issue closed |

One batch runs at a time. A batch id is never reused: a rerun is a new batch with `"supersedes": "<old id>"`.

## 4. What the listening agent may act on

The fork is public, so the agent treats everything as data unless all of these hold:

1. The issue carries `gpu:request` (only repository collaborators can set labels).
2. The issue author is on the allow-list (initially the fork owner only).
3. The pinned commit is reachable from a branch of the fork.

It runs only `python -m gpurun run <batch> --out <result-dir>` at that commit. It never executes text from an issue body or comment.

## 5. `batch.json`

```json
{
  "protocol": 1,
  "batch_id": "seq-smoke-001",
  "mode": "compare",
  "supersedes": null,
  "requires": { "sm": 90, "gpus": 1, "env": "triton" },
  "timeout_s": 600,
  "tests": [
    { "id": "vector_add/normal",
      "frontend": "triton",
      "entry": "bench/examples/VectorAdd/kernel.py::add_kernel",
      "launch": { "grid": [4], "num_warps": 4 },
      "inputs":  [ { "name": "x", "dtype": "fp32", "shape": [1000],
                     "gen": { "class": "normal", "seed": 1 }, "sha256": "<hex>" } ],
      "outputs": [ { "name": "out", "dtype": "fp32", "shape": [1000], "sentinel": "0x7fc0dead" } ],
      "reference": { "kind": "none" },
      "repeats": 1 }
  ]
}
```

- `frontend`: `triton`, `tilelang`, `triton-dist`, `tilescale`.
- `requires.env` names one of the environments of section 9. A batch uses one environment.
- Inputs are not shipped. Both sides generate them from `gen` with the pinned generator; the GPU side checks `sha256` and marks the test `input_mismatch` if it differs.
- Input classes: `normal` (random normal), `boundary` (non-divisible sizes), `special` (NaN, Inf, signed zero, subnormal).
- Output buffers are pre-filled with `sentinel` so the write footprint can be checked exactly.

Modes:

| Mode | Extra fields | GPU side returns |
| --- | --- | --- |
| `env-check` | none | `env.json` only |
| `compare` | `reference.kind`: `none` or `torch` (a torch callable run on the same GPU) | Kernel outputs, and reference outputs when asked |
| `agree` | `pair: { original, candidate }`, same inputs | Outputs of both kernels (verifiable-optimization line) |
| `timing` | `warmup`, `repeats` | Every run's wall time as a list; no speedup is computed on the GPU side |
| `dist` | `ranks`, `launcher`, `repeats` | Per repeat: sha256 of each output; one full copy of every distinct output |
| `litmus` | `repeats` (large) | Histogram of observed outcomes |

## 6. Result bundle `results/<batch-id>/`

```
manifest.json            batch id, protocol, request commit, start / end time, per-test status
env.json                 GPU model, compute capability, driver, CUDA, torch, triton,
                         tilelang commit, nvshmem version, python, OS release
<test-id>/meta.json      status, wall time, repeats, output sha256, input sha256 check
<test-id>/out/<name>.tensor
<test-id>/ref/<name>.tensor     only when reference.kind = torch
<test-id>/stdout.log, stderr.log    truncated at 1 MB
DONE                     written last
```

- Test status: `ok`, `compile_error`, `runtime_error`, `timeout`, `input_mismatch`, `skipped_env`. Each test is independent; a failed test does not stop the batch.
- No PASS / FAIL anywhere in the bundle.
- No compiled artifacts (`.ptx`, `.ttir`): the work does not go below the front-end language.
- Size limits: 50 MB per file, 100 MB per batch. A larger batch must be split by the dev side.
- Results are public. Logs and `env.json` must not contain host names, user names, IP addresses, home-directory paths or credentials; `gpurun` scrubs paths before writing.

## 7. Judging (dev side only)

Criteria of sub-issue I8.2: exact equality for integers, write footprint and NaN / Inf positions; gate `|gpu − ref| ≤ 1e-2 + 1e-2·|ref|`; strict fp32 column `1e-5 + 1e-4·|ref|`. For `agree`, the two GPU outputs are compared with each other under the same criteria, and each with the interpreter. An agent classifies every mismatch first (semantics, tolerance, environment, flaky); the user reviews the classification.

## 8. Automation on the GPU side

A polling loop (every 5 minutes while idle) lists open issues with `gpu:request`, applies section 4, swaps the label to `gpu:running`, runs the batch, pushes the bundle to `gpu-results`, comments the commit SHA and swaps the label to `gpu:result` (or `gpu:failed`). `gpurun check-env` refuses to run when a `requires` pin does not match and reports `skipped_env`.

## 9. Hardware and environments

Hardware: one node with H800 GPUs (sm_90), NVLink-connected in pairs. A request uses at most one pair: `compare`, `agree` and `timing` batches use one GPU; `dist` and `litmus` batches use two, with NVSHMEM and two local processes. `gpurun` selects the two visible GPUs and the process count explicitly; setting only `CUDA_VISIBLE_DEVICES` is not enough for the pinned launcher. A test that needs two nodes is requested with `"requires": { "nodes": 2 }` and returns `skipped_env` on this node. A program written for several nodes may still run on one node through a reduced path; such a run is reported as single-node coverage and never as an inter-node result (see the inter-node hardware ticket).

"Versions" means the software stack installed on the server. The front ends cannot share one Python environment, because Triton-distributed builds its own Triton and TileLang builds its own TVM. TileScale and TileLang both provide the `tilelang` import name and cannot be installed together. The GPU side therefore keeps four named environments and reports every version in `env.json`:

| Environment | Contents | Pinned by the dev side |
| --- | --- | --- |
| `triton` | Python, PyTorch, Triton | Triton `3.5.1` |
| `triton-dist` | Triton-distributed built from source with its bundled Triton (a patched 3.7.1, not 3.5.1), NVSHMEM, MPI launcher | Triton-distributed `7908e4e` |
| `tilelang` | TileLang built from source | TileLang `35850c0` |
| `tilescale` | TileScale built from source | TileScale `71e7145` |

Recorded but not pinned by the dev side: GPU driver, CUDA toolkit, Python, PyTorch, NVSHMEM and OS release. The first `env-check` batch reports them; from then on a change of any of them is a new `env.json` and is reported in the next result. A batch whose pinned version does not match returns `skipped_env`.

## 10. Limits

| Limit | Value | Reason |
| --- | --- | --- |
| Polling interval | 5 minutes while idle; no polling while a batch runs | Batches take minutes to hours, so latency does not matter; 12 requests an hour is far below API limits |
| Per-test timeout | `timeout_s`, default 600 s | A hung kernel must not hold the queue |
| Per-batch wall time | 6 hours | The batch is then closed with the tests finished so far |
| File size | 50 MB | GitHub warns above 50 MB and rejects above 100 MB |
| Batch size | 100 MB | Keeps a fetch of `gpu-results` fast |
| `gpu-results` branch | Dev side rewrites it to drop judged bundles when it passes 1 GB | GitHub's recommended repository size |
| Log size | 1 MB per stream, truncated in the middle | Keeps both the start and the failure |

## 11. Confirmed on 2026-11-10

- Results on a public branch are acceptable.
- The GPU side has push access to the fork.
- The listening agent is operated on the GPU side from the instructions document; the allow-list is the fork owner.
