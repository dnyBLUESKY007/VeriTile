[English](GpuInterNodeRuns.md) | [中文](GpuInterNodeRuns.zh.md)

# Single-host coverage of inter-node programs

Issue: [#353](https://github.com/dnyBLUESKY007/VeriTile/issues/353). Inspection date: 2026-11-10; logical-node launcher added on 2026-11-12; first logical-node result and the requests for tutorials 03 and 06 recorded on 2026-11-12. Status: source inspection complete. The launcher that starts two logical nodes on the one host has run on the GPU side: `internode-logical-env-002` returned on 2026-11-12 and was judged pass (two logical nodes come up and a put between them lands). The requests for tutorials 03 and 06, `internode-logical-ag03-001` and `internode-logical-rs06-001`, are written and have not run. Of the two outcomes the issue allows, the first will not hold (decision below). The second, a recorded check that the inter-node code paths run on the one host, holds so far for the launch and for one put, and not yet for any tutorial kernel. This document states what such a run exercises and what it does not, and names the batches.

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

Added on 2026-11-12. Of this section, the launcher and one put between two logical nodes have run on a GPU (`internode-logical-env-002`, subsection "First request and its result"); no tutorial kernel has run under the launcher. The launcher is also tested on the development host, which has no GPU, against a stand-in for the pinned script. Every statement about torchrun, NVSHMEM or NCCL is read in the cited source or inferred from it, and says which; the subsection "What only a run on the GPU side can settle" says which of them that run has since confirmed.

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

Installed on the GPU side, by the `env.json` of `internode-logical-env-002`: PyTorch `2.8.0+cu126`, the NVSHMEM library `3.6.5`, the NVSHMEM bindings `0.3.0`, Triton `3.7.1`. Not confirmed: the NCCL version, which `env.json` does not report.

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
| Tutorial 06 | `nnodes=2`. `kernel_inter_node_p2p_for_same_local_rank` is launched (for `n = 0`), the result is taken from the P2P buffer, and `ring_reduce` runs over two nodes. This needs a context that the tutorial's own constructor does not deliver outside node 0 (subsection "The tutorial requests"). | `intra_node_scatter` to another local rank: with one local rank it copies into the rank's own buffer. |
| All-to-all inter-node kernel | With `local_world_size=1` passed explicitly, `nnodes=2`, so the pull loop meets `src_node != node_id` and waits for the other node's signal. Read at these lines only; the kernel was not traced further. | Transfer between different local ranks of the receiving node. |
| Sequence-parallel all-gather attention inter-node kernel | Its push kernel has the structure of tutorial 03's 2D kernel: the cross-node put to the same local rank. Read at these lines only. | The intra-node forwarding branch, as for tutorial 03. |

The launcher also starts four ranks as two logical nodes of two (`ranks` 4, `logical_nodes` 2; tested with the stand-in), which would reach the forwarding branches. That takes four GPUs in one request; the rule for this host is two. The subsection "The tutorial requests" has the branches of tutorials 03 and 06 by line, and what the reading found on the way.

### What it does not exercise

- A NIC, or an RDMA or IBGDA transport: data is expected to move by GPU peer-to-peer mapping inside one host.
- NVSHMEM bootstrap, NCCL and the torchrun rendezvous across hosts: all three connect within one host.
- Isolation between hosts: the logical nodes share the operating system, the host name, the file system (the Triton cache directory included) and the CPU memory.
- A node boundary as NVSHMEM sees it. NVSHMEM puts the ranks whose host-name hashes agree into one node team (`init_team_node`, `src/host/team/team_internal.cpp:1106-1130` at tag `v3.6.5-0`; read on 2026-11-12, not run), so both logical nodes are one NVSHMEM node. Code that asks NVSHMEM which ranks share its node, instead of reading `LOCAL_WORLD_SIZE`, sees one node of all ranks. The intra-node barrier of tutorial 06 is such code (subsection "The tutorial requests").
- Latency, bandwidth and timing of a network between nodes.
- Failures: a lost node, a lost link, a partition.

A logical-node result therefore shows the control flow and the rank arithmetic of the `nnodes > 1` branches, and that data addressed to a rank of another node arrives, with the transport of a one-node run. It is evidence about the program text, not about communication between hosts. `gpurun` labels it: `meta.json` says `"topology": "logical-nodes"` with `logical_nodes` and `local_ranks`; `requires.nodes` and `GPURUN_NODES` stay 1; and a logical-node test in a batch that requires more than one physical node returns `skipped_env`.

### What only a run on the GPU side can settle

Each of the seven points is followed by its state on 2026-11-12, after `internode-logical-env-002` (next subsection). "Settled" means observed in that one batch: two repeats, on one pair of GPUs.

1. Whether two torchrun agents on one host meet at the loopback endpoint, and whether the fixed port `23456` is free: another job on the shared host, or the store of the previous repeat, may hold it. **Settled for that run:** in both repeats the two agents met at port `23456` of the loopback address and every rank started. Whether the port is free stays a question of each run.
2. Whether each rank gets the expected `RANK`, `LOCAL_RANK`, `WORLD_SIZE` and `LOCAL_WORLD_SIZE`. **Settled:** the ranks reported the expected values (table in the next subsection).
3. Whether NCCL forms the process group between two ranks that each see one GPU. **Settled:** `initialize_distributed` returned in both ranks; it creates the NCCL group over all ranks and runs a barrier on it (`python/triton_dist/utils.py:355-356` at the pin).
4. Whether the NVSHMEM UID bootstrap completes across the two launcher groups. By the source it sees only the unique identifier, the rank and the rank count that `initialize_distributed` passes, not the groups. **Settled:** NVSHMEM initialized in both ranks, and the barriers and the put after it completed.
5. Whether NVSHMEM reaches the hidden peer GPU through NVML and maps its heap, or stops with `Peer GPU … is not accessible`; and which NVSHMEM library version is installed. **Settled in its outcome:** initialization did not stop, the put landed, and the logs hold neither that message nor a warning about a network transport. Which branch of the source granted the mapping is not in the logs. The installed library is `3.6.5` (`env.json`).
6. Whether a put to the rank of the other logical node arrives, and whether `nvshmem.core.put` of the installed bindings accepts the call of the harness. **Settled:** every rank received the four values of the other rank. Not settled, and not touched by that batch: the one host-side put at the pin, which goes through the low-level binding, `pynvshmem.putmem_on_stream` in `python/triton_dist/kernels/nvidia/reduce_scatter.py` (`reduce_scatter_for_each_node_ring`, under `nnodes > 1`), called with four arguments where the binding at NVSHMEM tag `v3.6.5-0` declares five, the stream last (`nvshmem4py/nvshmem/bindings/nvshmem.pyx:1615`). Tutorial 06 does not reach that call (subsection "The tutorial requests"), so its request will not settle it either.
7. Whether two logical nodes that compile at the same time into the one Triton cache directory disturb each other. **Open:** the environment batch compiled no kernel. The tutorial requests are the first to do so: in the first repeat of each `logical_nodes` test both logical nodes compile the same kernel at once.

Added with the tutorial requests, all open:

8. Whether any kernel of tutorials 03 and 06 compiles and runs in the installed environment: none has run on the GPU side. The `one_node` control tests answer this apart from the logical nodes.
9. Whether Triton's JIT compiles a kernel of tutorial 03 from a module that holds only the definitions of the file, which is how the harness avoids running the tutorial's program. On the development host this is tested only as far as `inspect.getsourcelines`, which the JIT reads the kernel with, returning the text of the file.
10. What NVSHMEM's node team reports under logical nodes (expected from source: all ranks of the host), and whether the intra-node barrier of tutorial 06, which uses it, completes across the two logical nodes.

### First request and its result

`internode-logical-env-001` (`experiments/interp_gpu/batches/internode-logical-env-001/`, harness `experiments/interp_gpu/smoke/dist_logical_nodes.py`) addresses items 1 to 6 without a tutorial kernel. Each rank initializes through the pinned `initialize_distributed`; reports its rank, the world size, `LOCAL_WORLD_SIZE`, the derived node count, `LOCAL_RANK` and its CUDA device index as a tensor; prints its device name; puts four values into the other rank's symmetric memory between two barriers; and stops. The first barrier is there so that no put lands before the receiver has initialized its buffer.

The batch has two tests with the same harness: `one_node/put_barrier` under `triton-dist` as the control, and `logical_nodes/put_barrier` under the new launcher, so that a failure of the harness itself is told apart from a failure of the logical nodes. The development side expects: under the control, `LOCAL_WORLD_SIZE` 2, one node, device index equal to the rank; under the logical nodes, `LOCAL_WORLD_SIZE` 1, two nodes, device index 0 in both ranks; in both, every rank receives 100 plus the number of the other rank.

Result. `internode-logical-env-001` ran on the GPU side with every test `ok`, and its bundle was not published: the launcher's log holds the loopback address, and the rule for public results did not yet allow an address in a log. `gpurun` now writes that address as `<loopback>`, and the same request ran again as `internode-logical-env-002` on 2026-11-12 ([issue #430](https://github.com/dnyBLUESKY007/VeriTile/issues/430); request commit `ad26d295f4f8737ab82df8312a7eed21bbc4366c`; bundle `results/internode-logical-env-002/` at commit `3ff8914868fa562c7dff6caac9344627732c4ce4` of the branch `gpu-results`). Both tests ended `ok` with both ranks `ok` in both repeats, on the NVLink pair 4–5. Decoded on the development side, equal in both repeats:

| Test | Rank | `topology` (rank, world size, `LOCAL_WORLD_SIZE`, nodes, `LOCAL_RANK`, device index) | `received` |
| --- | --- | --- | --- |
| `one_node/put_barrier` | 0 | 0, 2, 2, 1, 0, 0 | 101 four times |
| `one_node/put_barrier` | 1 | 1, 2, 2, 1, 1, 1 | 100 four times |
| `logical_nodes/put_barrier` | 0 | 0, 2, 1, 2, 0, 0 | 101 four times |
| `logical_nodes/put_barrier` | 1 | 1, 2, 1, 2, 0, 0 | 100 four times |

Every value is the expected one, so the batch was judged pass: two logical nodes of one rank each come up on the one host through the pinned launcher, each rank sees only its own GPU, the runtime derives two nodes, and a put to the rank on the other logical node lands. The previous subsection says what this settles.

### The tutorial requests

Written on 2026-11-12; neither has run. `internode-logical-ag03-001` and `internode-logical-rs06-001` (under `experiments/interp_gpu/batches/`) call the functions of tutorials 03 and 06 on small fixed inputs with 2 ranks. Every test under `logical_nodes/` (launcher `triton-dist-logical-nodes`, two logical nodes of one rank) has a twin under `one_node/` (launcher `triton-dist`) with the same harness, inputs and outputs, so that a failure of the tutorial is told apart from a failure of the logical nodes. Every rank returns its result tensors and the `topology` values of the environment batch. The harness is `experiments/interp_gpu/internode_logical/harness.py`. Beside it, `judge.py` computes the expected tensors from the regenerated inputs with the Python standard library and prints `equal` or `different` per test, rank and output, and `test_internode_logical.py` tests both without a GPU. No tutorial file is edited or copied: the harness reads it from the checkout that the environment was built from. The batches are committed and not announced: they have no issue and no label.

`T03:n`, `T06:n`, `RS:n`, `CO:n` and `UT:n` are lines of `tutorials/03-inter-node-allgather.py`, `tutorials/06-inter-node-reduce-scatter.py`, `python/triton_dist/kernels/nvidia/reduce_scatter.py`, `python/triton_dist/kernels/nvidia/common_ops.py` and `python/triton_dist/utils.py` at the pin, as in [dist/EventControlFlow.md](dist/EventControlFlow.md). Everything in this subsection is read from source on 2026-11-12 or inferred from it; nothing in it has run.

**Tutorial 03, `internode-logical-ag03-001`.** Four tests: the kernels `push_1d` and `push_2d` under each launcher. Input: 64 bytes, the same in every rank; rank r owns bytes 32r to 32r + 31. The buffer of a rank starts with its own segment and the sentinel elsewhere, and comes back after one kernel call as `gathered`. Expected: every test `ok`, and `gathered` equal to the 64 input bytes in every rank.

| Kernel text | `one_node` (one node of two ranks) | `logical_nodes` (two nodes of one rank) | Left for two nodes of two ranks |
| --- | --- | --- | --- |
| `all_gather_push_1d_kernel` (`T03:62-95`); it has no branch on the node count | A put with signal to the other rank (`T03:85-95`) and a wait for its signal (`T03:72-76`). | The same. | Nothing. |
| `all_gather_push_2d_kernel`, a program of the same local rank and another rank (`T03:135-147`): the cross-node `putmem_signal_nbi_block` | Not reached. | Reached: every program has `peer_local_rank == local_rank`, and the program of the other rank puts the rank's segment to the other node. | Nothing. |
| The same kernel, the program of the rank itself (`T03:148-155`): the wait for every peer's signal | Reached. | Reached. | Nothing. |
| The same kernel, a program of another local rank (`T03:156-177`): the intra-node `putmem_signal_block` | Reached, without the wait of `T03:160-167`, because `peer_node_id == node_id`. | Not reached. | The wait for a segment from the other node before it is forwarded inside the node (`T03:160-167`). |

Read on the way:

- **The tutorial runs its whole program at import.** The file has no `__main__` guard: `T03:256-299` start the runtime, draw random data, time both kernels and stop the runtime. The harness therefore executes only the top-level imports, classes and functions of the file. It filters the syntax tree and compiles it under the name of the file, so that Triton's JIT still reads the unchanged kernel text at its lines. The harness is itself the program around these definitions: one buffer of 64 bytes and one kernel call per test, where the tutorial has a double buffer and a timing loop.
- **Signal tensors: the reading of observation 5 in section 8 of `dist/EventControlFlow.md` is confirmed from source.** `T03:276-278` create each signal tensor with shape `(1, )`, one 8-byte word (`UT:572`); `T03:103` and `T03:185` pass one of them as `symm_flag`. The 1D kernel waits on `symm_flag + peer` for every `peer` below `WORLD_SIZE` other than the rank (`T03:68-76`) and signals `symm_flag + segment` with `segment = rank` (`T03:81`, `T03:91`). The 2D kernel does the same at `T03:138-143`, `T03:149-154`, `T03:158-163` and `T03:173`. With two ranks, rank 1 sets word 1 of the tensor on rank 0 and rank 0 waits on that word: one word past what the tutorial allocated. So the tutorial's program accesses memory outside the tensor whenever `WORLD_SIZE > 1`, under one node as under two; whether the allocator leaves that word unused is not known. The batch does not run that program. The harness allocates `ranks` words per signal tensor and states it. Running the tutorial as written was not chosen because no outcome would show the defect: a pass does not refute it, and a hang, a wrong byte or an allocator error could each have another cause.

**Tutorial 06, `internode-logical-rs06-001`.** The file has a `__main__` guard (`T06:448`) and is imported whole. Input: per rank a 512 by 128 matrix of integers between -100 and 100, converted to bfloat16, the dtype of the tutorial (`T06:458`). `reduce_scatter_2d_op` (`T06:410-433`) is called as the tutorial's `__main__` calls it, between the same two barriers (`T06:474-479`), and what it returns comes back as `reduced`. Expected in rank r: rows 256r to 256r + 255 of the sum of the two matrices. Every sum is an integer of magnitude at most 200, which bfloat16 holds exactly, so the comparison has no tolerance. The shape is the smallest for which every tile of `kernel_ring_reduce_tma` (256 rows by 64 or 128 columns, `RS:749-775`) lies inside its tensors, as with the tutorial's own 8192 by 16384; a smaller matrix would also test how a tile behaves across the border of a tensor, which is not the subject.

| Text | `one_node` (one node of two ranks) | `logical_nodes` (two nodes of one rank) | Left for two nodes of two ranks |
| --- | --- | --- | --- |
| The loop over nodes (`T06:314`) | One pass: the own node. | Two passes: the other node, then the own. | Nothing. |
| `intra_node_scatter` (`T06:264-294`), host copies | Into the buffer of the other rank. | Into the rank's own buffer (`remote_local_rank == local_rank`). | A copy to another local rank in each of two passes. |
| `barrier_all_on_stream` (`T06:344`) | The intra-node barrier kernel over two ranks. | The same kernel, expected over two ranks, where a real node of one rank would run it over one (below). | Nothing. |
| `ring_reduce` per node (`T06:349-354`) | A sum of two blocks. | A copy of one block (`num_splits` is 1). | A sum of two blocks per pass. |
| Under `nnodes > 1` (`T06:357`), last pass (`T06:358-363`): the rank's own result is copied to the P2P buffer | Not reached. | Reached, second pass. | Nothing. |
| Under `nnodes > 1`, other passes (`T06:364-374`): `kernel_inter_node_p2p_for_same_local_rank`, one `putmem_block` to the same local rank of the other node (`T06:256-261`) | Not reached. | Reached, first pass. | Nothing. |
| The value returned (`T06:377-379`) | The per-node buffer. | The P2P buffer. | Nothing. |
| `ring_reduce` over nodes (`T06:406`) | A copy of one block (`num_splits = nnodes = 1`). | A sum of two blocks, one of them received from the other node. | Nothing. |

Four tests: `tutorial_context` and `harness_context` under each launcher. Read on the way:

- **The context constructor of the tutorial fails outside node 0: a predicted failure.** `create_reduce_scater_2d_ctx` gets its signal tensors from `nvshmem_create_tensors(..., rank, local_world_size)`, a list with one entry per rank of the caller's node (`UT:257-272`), and then takes `signal_bufs[rank]` with the global rank (`T06:199-202`). In a rank of node 0 that is the rank's own tensor. In any other rank the index is past the end of the list: `IndexError: list index out of range`, before any kernel is launched. So tutorial 06 as written cannot run with more than one node, logical or physical. The request exposes this and also goes around it, in separate tests:
  - `*/tutorial_context` calls the tutorial's constructor as its `__main__` does (`T06:460-466`). Under `one_node` it is expected to work. Under `logical_nodes` rank 1 is expected to raise the `IndexError`, and rank 0 is then expected to wait for it in the barrier of `T06:205` until the timeout of that test, 240 s. The expected status of `logical_nodes/tutorial_context` is `timeout` or `runtime_error`, with `runtime_error` for rank 1 and the `IndexError` in `stderr.log`. The judge calls that `failed as predicted`, and anything else `prediction not met`. This failure is the wanted result of the test; the notes of the batch say so to the GPU side.
  - `*/harness_context` builds the same `ReduceScatter2DContext` in the harness, with the allocations and values of `T06:188-229` and that one index taken within the node (`rank % local_world_size`). Everything after it is the tutorial's. `logical_nodes/harness_context` is the test that reaches the `nnodes > 1` branches of the table.
- **`putmem_on_stream` is not reached.** Tutorial 06 defines its own `reduce_scatter_for_each_node` and `reduce_scatter_multi_node` (`T06:297`, `T06:382`) and imports only `ring_reduce` from the library file (`T06:49`). The four-argument call of `pynvshmem.putmem_on_stream` is in `reduce_scatter_for_each_node_ring` (`RS:538-543`). Its one caller is the library's `reduce_scatter_multi_node`, when `has_fullmesh_nvlink()` is false (`RS:807-808`); tutorial 08 reaches that function, tutorial 06 does not. So this batch is not expected to show a `runtime_error` from that call, and the call stays unexercised by every batch of this document.
- **The intra-node barrier does not see the logical nodes.** The context holds `BarrierAllContext(True)` (`T06:223`), which takes its local rank and its node size from NVSHMEM's node team (`CO:238-240`), not from `LOCAL_WORLD_SIZE`. Both logical nodes are one NVSHMEM node (subsection "What it does not exercise"). Under `logical_nodes` the barrier of `T06:344` is therefore expected to run over both ranks. Both ranks call it equally often, once per pass, so it is expected to complete: it synchronizes more than the text intends for a node of one rank, not less. Every rank returns the local rank and the node size of the barrier in `barrier`; expected r and 2 under both launchers.
- **Native atomics are assumed under logical nodes.** `barrier_all_on_stream` launches the compare-and-swap kernel when `supports_p2p_native_atomic()` holds (`CO:257-258`). That function returns true without asking the driver when the process sees one GPU (`UT:536-538`), as every rank of a logical node of one does; under `one_node` it asks the driver about the pair (`UT:544-547`). The docstring of the kernel forbids it where the link between the GPUs has no atomics (`CO:155-157`). The third value of `barrier` is the answer of that function: expected 1 under both launchers on an NVLink pair (inferred from that docstring, which names PCIe as the case without atomics). A 0 under `one_node` would mean that the `logical_nodes` test used atomic operations on a pair that reports none. The request belongs on an NVLink pair, as the environment batch ran.

**Branches left for two local ranks per node.** The wait of `T03:160-167` and, in tutorial 06, a scatter to another local rank followed by the P2P kernel need two logical nodes of two ranks: four GPUs in one request. The launcher supports it (`ranks` 4, `logical_nodes` 2). Proposed names, without manifests: `internode-logical-ag03-2x2-001` and `internode-logical-rs06-2x2-001`. Four GPUs exceed the rule of two per request for this host, so such a request needs the user's permission first. In it NVSHMEM's node team would hold all four ranks, so the barrier of tutorial 06 would again not follow the logical nodes.

## Affected requests and completion gate

Of the batches of this document, `internode-logical-env-001` and `internode-logical-env-002` have run, and the second has a published result. `internode-logical-ag03-001` and `internode-logical-rs06-001` exist as manifests and are not announced. The other identifiers are proposed names, not existing manifests or results:

| Proposed batch | Intended check | Present disposition |
| --- | --- | --- |
| `internode-ag03-single-host-001` | Tutorial 03 1D/2D paths with two local ranks | No manifest of its own: the `one_node` tests of `internode-logical-ag03-001` are this check. |
| `internode-rs06-single-host-001` | Tutorial 06 with two local ranks | No manifest of its own: the `one_node` tests of `internode-logical-rs06-001` are this check. |
| `internode-a2av-single-host-001` | All-to-all core with explicit `local_world_size=2` | Await `gpurun` and a dev-side adapter/manifest; stock test assumes 8. |
| `internode-sp-attn-single-host-001` | Sequence-parallel attention with two local ranks | Await `gpurun` and manifest; remote-node branches remain unexercised. |
| `internode-ag03a-rocm-001` | Unchanged tutorial 03a | Unavailable on H800; requires an AMD environment or an explicitly reviewed port. |
| `internode-logical-env-001` | Two logical nodes come up; one put between two barriers; no kernel | Ran with every test `ok`; bundle not published. Superseded by `internode-logical-env-002`. |
| `internode-logical-env-002` | The same request under a new identifier | Ran on 2026-11-12; judged pass. |
| `internode-logical-ag03-001` | Tutorial 03 under `triton-dist-logical-nodes`, two logical nodes of one rank: the cross-node branch of the 2D all-gather; with the same tests under `triton-dist` as the control | Manifest committed on 2026-11-12; waits for the coordinator to announce it. Expected: every test `ok` and equal. Logical-node coverage only. |
| `internode-logical-rs06-001` | Tutorial 06 under the same launcher: `kernel_inter_node_p2p_for_same_local_rank` and the reduction over two nodes; with the same tests under `triton-dist` as the control | Manifest committed on 2026-11-12; waits for the coordinator to announce it. Expected: three tests `ok` and equal, and `logical_nodes/tutorial_context` failing as predicted. Logical-node coverage only. |
| `internode-logical-ag03-2x2-001` | Tutorial 03, two logical nodes of two ranks: the wait before the intra-node forwarding | Name only. Four GPUs in one request: needs the user's permission. |
| `internode-logical-rs06-2x2-001` | Tutorial 06, two logical nodes of two ranks: scatter to another local rank, then the P2P kernel | Name only. Same condition. |

Requests requiring two physical nodes remain `skipped_env` under protocol v1. A logical-node run must not be reported as a two-physical-node result; its `meta.json` carries the label `"topology": "logical-nodes"`.

Issue #353 is not accepted as runtime-verified by this document. It remains open until the executable requests above have results and the development side has recorded their comparison and coverage limits, or the user explicitly accepts another disposition. The user has explicitly required the first real batch to wait for `gpurun`; this inspection does not bypass that dependency.
