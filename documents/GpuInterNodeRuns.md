[English](GpuInterNodeRuns.md) | [中文](GpuInterNodeRuns.zh.md)

# Single-host coverage of inter-node programs

Issue: [#353](https://github.com/dnyBLUESKY007/VeriTile/issues/353). Inspection date: 2026-11-10. Status: source inspection complete; kernel runtime verification pending the GPU exchange runner. No second physical node has been arranged.

## Hardware and scope

The inspected host exposes six H800 PCIe GPUs, all compute capability 9.0. NVLink pairs are 0–1, 2–3 and 4–5. Only two GPUs are intended for each distributed request; the local environment checks used the then-idle 4–5 pair. A device selection is not a resource reservation and occupancy must be rechecked before a run.

All source observations below refer to [Triton-distributed `7908e4ea9010bb2238f9f05b28c904fd4b71c60a`](https://github.com/ByteDance-Seed/Triton-distributed/tree/7908e4ea9010bb2238f9f05b28c904fd4b71c60a). They establish control-flow and configuration constraints, not successful execution or numerical correctness.

Ordinary one-host/two-process launch gives `WORLD_SIZE=2`, `LOCAL_WORLD_SIZE=2`, and `NNODES=1`. A program named `inter_node` can then execute a reduced single-node path while skipping the inter-node communication it was written to exercise.

## Source findings

| Program | One-host/two-GPU source assessment | Missing coverage or blocker |
| --- | --- | --- |
| [Tutorial 03](https://github.com/ByteDance-Seed/Triton-distributed/blob/7908e4ea9010bb2238f9f05b28c904fd4b71c60a/tutorials/03-inter-node-allgather.py) | Its 1D all-gather sends to the other rank through NVSHMEM. Its 2D all-gather accepts `NNODES=1` and uses the intra-node peer branch. Runtime is untested. | The 2D same-local-rank/different-node send and its receive/forward ordering are not covered when `NNODES=1`. |
| [Tutorial 03a](https://github.com/ByteDance-Seed/Triton-distributed/blob/7908e4ea9010bb2238f9f05b28c904fd4b71c60a/tutorials/03a-inter-node-allgather.py) | Not an NVIDIA tutorial at this pin: it imports `pyrocshmem`, uses ROCSHMEM device contexts and ROCSHMEM signal operations. | Cannot run unchanged in the H800/CUDA environment. An AMD/ROCm host or a separately reviewed NVIDIA port is needed. No port was made. |
| [Tutorial 06](https://github.com/ByteDance-Seed/Triton-distributed/blob/7908e4ea9010bb2238f9f05b28c904fd4b71c60a/tutorials/06-inter-node-reduce-scatter.py) | The code explicitly supports `nnodes == 1`, returning the local reduce-scatter buffer. Runtime is untested. | `kernel_inter_node_p2p_for_same_local_rank` is launched only under `nnodes > 1`; a normal two-rank single-host run does not execute it. |
| [All-to-all inter-node kernel](https://github.com/ByteDance-Seed/Triton-distributed/blob/7908e4ea9010bb2238f9f05b28c904fd4b71c60a/python/triton_dist/kernels/nvidia/all_to_all_vdev_2d_offset_inter_node.py) | The core context can describe `world_size=2, local_world_size=2`. The supplied test calls `create_context` without overriding its default `local_world_size=8`. | The supplied two-rank test configuration violates `world_size % local_world_size == 0`. A dev-side adapter must explicitly pass 2. With one logical node, the different-node receive-wait branch is skipped. No test or kernel was edited. |
| [Sequence-parallel all-gather attention inter-node kernel](https://github.com/ByteDance-Seed/Triton-distributed/blob/7908e4ea9010bb2238f9f05b28c904fd4b71c60a/python/triton_dist/kernels/nvidia/sp_ag_attention_inter_node.py) | Its test derives `NNODES` from `WORLD_SIZE / LOCAL_WORLD_SIZE` and passes both into the context. A two-rank single-host configuration follows its intra-node all-gather path. Runtime is untested. | Remote-node puts and cross-node receive/forward synchronization are not covered with `NNODES=1`. |

## Launch constraints

The pinned `scripts/launch.sh` recognizes `--nproc_per_node=2` (with `=`). Its fallback counts physical devices with `nvidia-smi`, so setting only `CUDA_VISIBLE_DEVICES` is insufficient on this six-GPU host. An eventual dev-side request must explicitly select two visible GPUs and two processes.

Do not falsify topology by changing only `LOCAL_WORLD_SIZE`: initialization maps `LOCAL_RANK` to the CUDA device, and symmetric peer-buffer construction also depends on the local grouping. A future two-logical-node experiment on one physical host needs an explicit, reviewed launcher and batch description. Even if it executes remote-node source branches, it does not establish inter-host NIC, RDMA/IBGDA, isolation, latency or failure behavior.

## Affected requests and completion gate

No new batch was submitted or run for this inspection. The following are proposed batch identifiers, not existing manifests or results:

| Proposed batch | Intended check | Present disposition |
| --- | --- | --- |
| `internode-ag03-single-host-001` | Tutorial 03 1D/2D paths with two local ranks | Await `gpurun` and a pinned dev-side manifest; single-node coverage only. |
| `internode-rs06-single-host-001` | Tutorial 06 with two local ranks | Await `gpurun` and manifest; does not cover its inter-node P2P kernel. |
| `internode-a2av-single-host-001` | All-to-all core with explicit `local_world_size=2` | Await `gpurun` and a dev-side adapter/manifest; stock test assumes 8. |
| `internode-sp-attn-single-host-001` | Sequence-parallel attention with two local ranks | Await `gpurun` and manifest; remote-node branches remain unexercised. |
| `internode-ag03a-rocm-001` | Unchanged tutorial 03a | Unavailable on H800; requires an AMD environment or an explicitly reviewed port. |

Requests requiring two physical nodes remain `skipped_env` under protocol v1. A logical-node experiment must not be reported as a two-physical-node result.

Issue #353 is not accepted as runtime-verified by this document. It remains open until the executable requests above have results and the development side has recorded their comparison and coverage limits, or the user explicitly accepts another disposition. The user has explicitly required the first real batch to wait for `gpurun`; this inspection does not bypass that dependency.
