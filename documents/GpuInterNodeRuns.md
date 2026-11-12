[English](GpuInterNodeRuns.md) | [中文](GpuInterNodeRuns.zh.md)

# Single-host coverage of inter-node programs

Issue: [#353](https://github.com/dnyBLUESKY007/VeriTile/issues/353). Inspection date: 2026-11-10; logical-node launcher added on 2026-11-12. Status: source inspection complete; a launcher that starts two logical nodes on the one host and its first request, `internode-logical-env-001`, are written and have not run on a GPU. Of the two outcomes the issue allows, neither holds yet: no second sm_90 node is arranged for the H800 host, and the recorded check that the inter-node code paths run on that host awaits the GPU side. This document states what such a run exercises and what it does not, and names the batches.

**Decision (user, 2026-11-12).** No second physical node will be arranged: the collaborator's cluster is no longer available. Inter-node programs are tested only as logical nodes on the one host. Transport between hosts (network cards, RDMA) is outside the scope of the project: the model has ranks and scopes, not nodes or links, so a proof does not depend on it. If logical nodes cannot be made to run, the inter-node code paths are proved and reported as not run.

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

Do not falsify topology by changing only `LOCAL_WORLD_SIZE`: initialization maps `LOCAL_RANK` to the CUDA device, and symmetric peer-buffer construction also depends on the local grouping. A two-logical-node run on one physical host needs an explicit launcher and batch description; the next section describes them. Even when it executes remote-node source branches, it does not establish inter-host NIC, RDMA/IBGDA, isolation, latency or failure behavior.

## Logical nodes on the one host

Added on 2026-11-12. Nothing in this section has run on a GPU. The launcher is tested on the development host, which has no GPU, against a stand-in for the pinned script. Every statement about torchrun, NVSHMEM or NCCL is read in the cited source or inferred from it, and says which.

### Launcher

`gpurun` has a second `dist` launcher, `triton-dist-logical-nodes`, with the test field `logical_nodes` ([GpuExchangeProtocol.md](GpuExchangeProtocol.md), sections 5, 6 and 9). It starts the pinned `scripts/launch.sh` once per logical node, all on the one host, and drives it through the multi-node variables the script already reads. For `ranks` 2 and `logical_nodes` 2:

| Given to logical node n (0 or 1) | What the pinned script does with it (read in `scripts/launch.sh` at the pin) |
| --- | --- |
| `ARNOLD_WORKER_NUM=2`, `ARNOLD_ID=n` | Passes `--nnodes=2 --node_rank=n` to torchrun. |
| `--nproc_per_node=1` | One rank per node. Without the argument the script counts physical GPUs with `nvidia-smi`. |
| `ARNOLD_WORKER_0_HOST=127.0.0.1` | Passes `--rdzv_endpoint=127.0.0.1:<port>`. The address is a constant of the launcher. The port stays the script's: `23456`, or the operator's `ARNOLD_WORKER_0_PORT`. |
| `CUDA_VISIBLE_DEVICES=` the n-th entry of the operator's list | Not read by the script; inherited by torchrun and by the rank. |

Everything else stays as the script sets it, the NVSHMEM settings among them: `NVSHMEM_BOOTSTRAP=UID`, the bootstrap interface, the symmetric heap size, `NVSHMEM_DISABLE_CUDA_VMM=1`. The launcher needs no interface name: both logical nodes are on the host whose interface the operator already chose for `dist` batches. A request names no host, interface or address.

What each rank is then expected to see, by the torchrun source of the PyTorch version the GPU side reported (2.8.0; `torch/distributed/elastic/agent/server/local_elastic_agent.py`, and `_assign_worker_ranks` in `api.py`): `WORLD_SIZE=2`, `LOCAL_WORLD_SIZE=1`, `LOCAL_RANK=0`, and `RANK` 0 on node 0, 1 on node 1. The pinned `initialize_distributed` then calls `torch.cuda.set_device(0)` in both ranks, and device 0 is a different physical GPU in each: the GPU that rank has under the launcher `triton-dist`. The programs derive `nnodes = WORLD_SIZE // LOCAL_WORLD_SIZE = 2`. A rank whose `LOCAL_WORLD_SIZE` is not `ranks / logical_nodes` is stopped by `gpurun` before the harness is called.

Design choice. The alternative was a launcher of our own that starts every rank itself and sets `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `LOCAL_WORLD_SIZE`, the torch rendezvous variables and the NVSHMEM settings. Starting the pinned script once per logical node was chosen because it keeps the pinned NVSHMEM settings and torchrun, and is about twenty lines of new code. It relies on:

1. the script's multi-node variables keeping the meaning read at the pin;
2. torchrun's static rendezvous accepting two agents on one host, where the agent with node rank 0 hosts the store at the endpoint and the other connects to it (read in `torch/distributed/elastic/rendezvous/static_tcp_rendezvous.py`; not run);
3. torchrun handing its own environment, `CUDA_VISIBLE_DEVICES` included, to each rank (read in `torch/distributed/elastic/multiprocessing/subprocess_handler/subprocess_handler.py`);
4. NCCL and NVSHMEM reaching a GPU that the process cannot see (next subsection).

Each logical node sees only its own GPUs, as a node of a real cluster does. If a GPU run shows that item 4 fails, the fallback is to leave every GPU visible to every logical node and list the node's own GPUs first: NVSHMEM and NCCL then see the peer as they do under `triton-dist`, but a program that counts visible devices sees more than a real node has. The fallback is not implemented.

### Transport expected between the two GPUs

This subsection is an expectation from source, not an observation. Library source is used only to predict whether a run starts; it defines no semantics.

Sources, all read on github.com on 2026-11-12:

- NVSHMEM: [`NVIDIA/nvshmem` at tag `v3.6.5-0`](https://github.com/NVIDIA/nvshmem/tree/6e53a40d9db16df170b3b0057e94c340d72f77eb), commit `6e53a40d9db16df170b3b0057e94c340d72f77eb`. The pinned `python/setup.py` of Triton-distributed requires `nvidia-nvshmem-cu12==3.6.5`.
- NCCL: [`NVIDIA/nccl` at tag `v2.27.3-1`](https://github.com/NVIDIA/nccl/tree/72d2432094d6ae36abd6e511c3a16a2d052dbf94), commit `72d2432094d6ae36abd6e511c3a16a2d052dbf94`. PyTorch 2.8.0 requires `nvidia-nccl-cu12==2.27.3` for CUDA 12.6 (`.github/scripts/generate_binary_build_matrix.py`).
- PyTorch: [`pytorch/pytorch` at tag `v2.8.0`](https://github.com/pytorch/pytorch/tree/ba56102387ef21a3b04b357e5b183d48f0afefc7), commit `ba56102387ef21a3b04b357e5b183d48f0afefc7`.

Not confirmed here: which versions are installed on the GPU side. The one `triton-dist` `env.json` returned so far (`env-check-triton-dist-001`) reports PyTorch `2.8.0+cu126` and the NVSHMEM bindings `0.3.0`; it does not report the NVSHMEM library or NCCL.

Read in source:

- NVSHMEM identifies a host by a hash of `gethostname()` (`src/host/util/util.cpp`, `nvshmemu_getHostHash`). Nothing the launcher sets enters it.
- Its P2P transport (`src/host/transport/p2p/p2p.cpp`, `nvshmemt_p2p_can_reach_peer`) does not reach a peer with another host hash. For a peer with the same hash it looks for the peer's GPU among the devices the process sees. When it finds it, it asks `cudaDeviceCanAccessPeer`. When it does not, which the source attributes to `CUDA_VISIBLE_DEVICES`, and the CUDA driver is version 12 or later or the heap is not the VMM heap, it asks NVML for the P2P status of the two devices and grants the mapping when read and write are allowed. Otherwise it warns that it uses a network transport.
- A peer that no transport reaches ends initialization with the message `Peer GPU … is not accessible` (`src/host/topo/topo.cpp`, `nvshmemi_build_transport_map`).
- NCCL, which `initialize_distributed` uses for its process group, treats two ranks as local when their host hashes agree; where a peer's device is not visible to the process, its P2P check carries the comment `CUDA 10.1 and later can use P2P with invisible devices` (`src/transport/p2p.cc`).

Inferred: both logical nodes have the same host hash, so NVSHMEM is expected to use its P2P transport between the two GPUs, that is, the mapping of the peer's symmetric heap that an ordinary one-node run uses, over the NVLink of the pair. It is reached through the NVML branch, because each logical node hides the other's GPU. No NIC and no remote transport (IBRC, UCX, IBGDA, libfabric) is expected to carry data.

### What a logical-node run exercises

Source reading at the pin for two logical nodes of one local rank each (`WORLD_SIZE=2`, `LOCAL_WORLD_SIZE=1`, `nnodes=2`):

| Program | Reached with two logical nodes of one rank | Still not reached |
| --- | --- | --- |
| Tutorial 03 | `NNODES=2`. In `all_gather_push_2d_kernel` every program has `peer_local_rank == local_rank`: the program of the other rank does the cross-node `putmem_signal_nbi_block`, the program of the own rank waits for the peer's signal. | The intra-node branch, with its wait for a segment from another node before forwarding it. It needs a peer with another local rank: two local ranks on each of two nodes, four GPUs. |
| Tutorial 06 | `nnodes=2`. `kernel_inter_node_p2p_for_same_local_rank` is launched (for `n = 0`), the result is taken from the P2P buffer, and `ring_reduce` runs over two nodes. | `intra_node_scatter` to another local rank: with one local rank it copies into the rank's own buffer. |
| All-to-all inter-node kernel | With `local_world_size=1` passed explicitly, `nnodes=2`, so the pull loop meets `src_node != node_id` and waits for the other node's signal. Read at these lines only; the kernel was not traced further. | Transfer between different local ranks of the receiving node. |
| Sequence-parallel all-gather attention inter-node kernel | Its push kernel has the structure of tutorial 03's 2D kernel: the cross-node put to the same local rank. Read at these lines only. | The intra-node forwarding branch, as for tutorial 03. |

The launcher also starts four ranks as two logical nodes of two (`ranks` 4, `logical_nodes` 2; tested with the stand-in), which would reach the forwarding branches. That takes four GPUs in one request; the rule for this host is two.

### What it does not exercise

- A NIC, or an RDMA or IBGDA transport: data is expected to move by GPU peer-to-peer mapping inside one host.
- NVSHMEM bootstrap, NCCL and the torchrun rendezvous across hosts: all three connect within one host.
- Isolation between hosts: the logical nodes share the operating system, the host name, the file system (the Triton cache directory included) and the CPU memory.
- Latency, bandwidth and timing of a network between nodes.
- Failures: a lost node, a lost link, a partition.

A logical-node result therefore shows the control flow and the rank arithmetic of the `nnodes > 1` branches, and that data addressed to a rank of another node arrives, with the transport of a one-node run. It is evidence about the program text, not about communication between hosts. `gpurun` labels it: `meta.json` says `"topology": "logical-nodes"` with `logical_nodes` and `local_ranks`; `requires.nodes` and `GPURUN_NODES` stay 1; and a logical-node test in a batch that requires more than one physical node returns `skipped_env`.

### What only a run on the GPU side can settle

1. Whether two torchrun agents on one host meet at the loopback endpoint, and whether the fixed port `23456` is free: another job on the shared host, or the store of the previous repeat, may hold it.
2. Whether each rank gets the expected `RANK`, `LOCAL_RANK`, `WORLD_SIZE` and `LOCAL_WORLD_SIZE`.
3. Whether NCCL forms the process group between two ranks that each see one GPU.
4. Whether the NVSHMEM UID bootstrap completes across the two launcher groups. By the source it sees only the unique identifier, the rank and the rank count that `initialize_distributed` passes, not the groups; that is a reading, not a run.
5. Whether NVSHMEM reaches the hidden peer GPU through NVML and maps its heap, or stops with `Peer GPU … is not accessible`; and which NVSHMEM library version is installed.
6. Whether a put to the rank of the other logical node arrives, and whether `nvshmem.core.put` of the installed bindings accepts the call of the harness. The tutorials put from inside kernels. The one host-side put at the pin goes through the low-level binding: `pynvshmem.putmem_on_stream` in `python/triton_dist/kernels/nvidia/reduce_scatter.py` (`reduce_scatter_for_each_node_ring`, under `nnodes > 1`), called with four arguments where the binding at NVSHMEM tag `v3.6.5-0` declares five, the stream last (`nvshmem4py/nvshmem/bindings/nvshmem.pyx`). The harness calls `nvshmem.core.put`, which passes the stream.
7. Whether two logical nodes that compile at the same time into the one Triton cache directory disturb each other. This first matters for the tutorial batches.

### First request

`internode-logical-env-001` (`experiments/interp_gpu/batches/internode-logical-env-001/`, harness `experiments/interp_gpu/smoke/dist_logical_nodes.py`) addresses items 1 to 6 without a tutorial kernel. Each rank initializes through the pinned `initialize_distributed`; reports its rank, the world size, `LOCAL_WORLD_SIZE`, the derived node count, `LOCAL_RANK` and its CUDA device index as a tensor; prints its device name; puts four values into the other rank's symmetric memory between two barriers; and stops. The first barrier is there so that no put lands before the receiver has initialized its buffer.

The batch has two tests with the same harness: `one_node/put_barrier` under `triton-dist` as the control, and `logical_nodes/put_barrier` under the new launcher, so that a failure of the harness itself is told apart from a failure of the logical nodes. The development side expects: under the control, `LOCAL_WORLD_SIZE` 2, one node, device index equal to the rank; under the logical nodes, `LOCAL_WORLD_SIZE` 1, two nodes, device index 0 in both ranks; in both, every rank receives 100 plus the number of the other rank. The batch is committed and not announced: it has no issue and no label.

## Affected requests and completion gate

No batch of this document has run. `internode-logical-env-001` exists as a manifest and is not announced. The other identifiers are proposed names, not existing manifests or results:

| Proposed batch | Intended check | Present disposition |
| --- | --- | --- |
| `internode-ag03-single-host-001` | Tutorial 03 1D/2D paths with two local ranks | Await `gpurun` and a pinned dev-side manifest; single-node coverage only. |
| `internode-rs06-single-host-001` | Tutorial 06 with two local ranks | Await `gpurun` and manifest; does not cover its inter-node P2P kernel. |
| `internode-a2av-single-host-001` | All-to-all core with explicit `local_world_size=2` | Await `gpurun` and a dev-side adapter/manifest; stock test assumes 8. |
| `internode-sp-attn-single-host-001` | Sequence-parallel attention with two local ranks | Await `gpurun` and manifest; remote-node branches remain unexercised. |
| `internode-ag03a-rocm-001` | Unchanged tutorial 03a | Unavailable on H800; requires an AMD environment or an explicitly reviewed port. |
| `internode-logical-env-001` | Two logical nodes come up; one put between two barriers; no kernel | Manifest committed on 2026-11-12; waits for the coordinator to announce it. |
| `internode-logical-ag03-001` | Tutorial 03 under `triton-dist-logical-nodes`, two logical nodes of one rank: the cross-node branch of the 2D all-gather | Name only; after `internode-logical-env-001` has a result. Logical-node coverage only. |
| `internode-logical-rs06-001` | Tutorial 06 under the same launcher: `kernel_inter_node_p2p_for_same_local_rank` and the reduction over two nodes | Name only; same condition. Logical-node coverage only. |

Requests requiring two physical nodes remain `skipped_env` under protocol v1. A logical-node run must not be reported as a two-physical-node result; its `meta.json` carries the label `"topology": "logical-nodes"`.

Issue #353 is not accepted as runtime-verified by this document. It remains open until the executable requests above have results and the development side has recorded their comparison and coverage limits, or the user explicitly accepts another disposition. The user has explicitly required the first real batch to wait for `gpurun`; this inspection does not bypass that dependency.
