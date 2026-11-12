# DeepEP inventory and pins

Status: delivered for ticket K-D.1 on 2026-11-12, not yet reviewed by the user.
It applies three user decisions of 2026-11-12: the project proves the normal
(non low-latency) dispatch and combine of DeepEP version 1; GPU runs use one
H800 node (sm_90, one NVLink pair, two GPUs) and proofs stay independent of the
GPU generation; the examples end up in `VeriInfra-Lab/VeriTileBench`, so this
document records sources and pins only.

Nothing here is a proved result, and nothing was executed: this host has no
GPU and none of the packages. Every source was fetched read-only at the commit
named in section 2 and read as text.

Each claim carries one of three marks:

- **R** (read): stated by the source at the file and line given.
- **I** (inferred): a conclusion drawn from lines that were read; the
  conclusion itself is not written in the source.
- **U** (unchecked): not established. Section 9 lists every such item.

## 1. Answers

1. **What "DeepEP v1, normal version" is.** DeepEP's repository holds version 1
   at its tree root up to commit `567632dd` (2026-02-03); the next commit,
   `b306af06` (2026-04-30), introduces EPv2 and moves version 1 to `legacy`
   directories; `def86514` (2026-09-29, "DeepEP V2.5") removes version 1. The
   normal mode is the pair of "normal kernels" of version 1 (`dispatch`,
   `combine` and their layout and notify kernels); the mode this project proves
   is its intranode case, where all ranks are on one NVLink domain.
2. **Which port implements it.** TileScale branch `wt/deepep-internode` at
   `9ba8a605`, directory `examples/distributed/deepseek_deepep/intranode/`, is
   a kernel-by-kernel port of the version 1 intranode normal mode. Despite the
   branch name it holds no inter-node dispatch or combine. The other pinned
   branch (`deepep-v2-intranode`) ports EPv2. No Triton-distributed kernel is a
   port of version 1: its expert-parallel kernels solve the same problem with a
   different layout and protocol.
3. **Recommended source.** The TileScale port at `9ba8a605` for the proof, with
   DeepEP at `b57e5e21` as the reference it was written and tested against
   (section 6).
4. **GPU run.** The recommended source cannot run in an installed environment:
   the `tilescale` environment is built from `main` at `71e71457`, where two
   names the port needs do not exist. No run request was written. Section 7
   says exactly what is missing.
5. **Specification.** The frozen MoE declarations hold for version 1 normal
   mode as far as they go, but they leave out or weaken six things that
   version 1 fixes (section 8).

## 2. Sources and how they were read

| Name | Repository | Commit | Date of the commit | License |
| --- | --- | --- | --- | --- |
| DeepEP, last version 1 commit | <https://github.com/deepseek-ai/DeepEP> | `567632dd59810d77b3cc05553df953cc0f779799` | 2026-02-03 | MIT (`LICENSE`); the files that reference NVSHMEM are under the NVSHMEM SLA (`README.md:326`) |
| DeepEP, reference of the port | same | `b57e5e212ab75350f53c72064333e4fe1076b1da` | 2025-12-05 | same |
| DeepEP, only tag | same | `9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee` (tag `v1.2.1`) | 2025-09-15 | same |
| TileScale, version 1 port | <https://github.com/tile-ai/tilescale> | `9ba8a605d347f600a95d0275e2639cb00122c084` (branch `wt/deepep-internode`) | authored 2025-12-08, committed 2026-02-06 | MIT (`LICENSE`), no per-file header |
| TileScale, EPv2 port | same | `c0f80cc5a5e386ae77075af9574701d3e0a99b00` (branch `deepep-v2-intranode`) | 2026-08-17 | same |
| TileScale, `main` (installed environment) | same | `71e71457d40ed6e6a2dad297936bfdbff21d104b` | 2026-08-03 | same |
| Triton-distributed | <https://github.com/ByteDance-Seed/Triton-distributed> | `7908e4ea9010bb2238f9f05b28c904fd4b71c60a` | 2026-09-18 | MIT terms (`LICENSE`; the text does not use the name); the same notice heads `ep_a2a_intra_node.py` |

Branch heads were confirmed with `git ls-remote` on 2026-11-12: both TileScale
branches and `main` still point at the commits above. The TileScale and
Triton-distributed trees were fetched with `git fetch --depth 1 <url> <hash>`;
DeepEP was fetched with its full history of `main` (258 commits) to answer the
version question. Submodules were not fetched. The census of section 5 was
produced by `scripts/census/scan.py`, which parses files and never imports
them.

## 3. The version question

### 3.1 History of DeepEP

| Commit | Date | What it is | Evidence (R) |
| --- | --- | --- | --- |
| `9af0e0d0` | 2025-09-15 | Tag `v1.2.1`, the only tag of the repository | `git ls-remote --tags` |
| `b57e5e21` | 2025-12-05 | The commit that TileScale `9ba8a605` records as submodule `3rdparty/DeepEP` | `git ls-tree 9ba8a605 3rdparty/` |
| `567632dd` | 2026-02-03 | Last commit with version 1 at the tree root; `setup.py:117` gives version `'1.2.1' + revision` | parent of `b306af06` |
| `b306af06` | 2026-04-30 | "[Public release 26/04] Introducing EPv2"; version 1 moves to `csrc/kernels/legacy/`, `csrc/legacy/`, `deep_ep/buffers/legacy.py`, `docs/legacy.md` | file list of the commit |
| `a56d6156` | 2026-09-16 | Last commit that still carries the `legacy` copy | parent of `def86514` |
| `def86514` | 2026-09-29 | "DeepEP V2.5"; the README of `main` says "Fully remove V1, including its APIs, NVSHMEM backend, and legacy documentation" | `README.md:18` at `f6aa38fe` |

Three facts fix the pin:

- **R.** Between `b57e5e21` and `567632dd` only five files change
  (`.github/workflows/format.yml`, `README.md`, `csrc/kernels/configs.cuh`,
  `csrc/kernels/ibgda_device.cuh`, `third-party/README.md`); the kernels
  `csrc/kernels/intranode.cu` and `csrc/kernels/layout.cu` are identical.
- **R.** Between the tag `v1.2.1` and `567632dd`, `intranode.cu` changes in 622
  lines and `layout.cu` in 66. The tag is therefore not the same kernel text.
- **R.** The `legacy` copy of `intranode.cu` at `b306af06` and at `a56d6156`
  differs from `567632dd` in 30 lines: the namespace, the includes, a
  stand-alone `barrier` kernel and the name of the timeout constant.

"DeepEP version 1" in this project therefore means the tree at `567632dd`, and
its intranode normal kernels are equally the ones at `b57e5e21`.

### 3.2 What version 1 normal mode does, intranode

All line numbers are in DeepEP at `567632dd`. `R` ranks hold `E` experts,
`E / R` consecutive experts per rank (`intranode.cu:153`). Rank `s` holds
`num_tokens` tokens of `hidden` elements (the count is an argument of each
process and may differ between ranks) and, per token, `num_topk` expert indices
`topk_idx`, where `-1` means no selection (`deep_ep/buffer.py:301-302`), and
`num_topk` weights.

**Layout** (`csrc/kernels/layout.cu:10-121`, one kernel `get_dispatch_layout`,
runs on one GPU without communication).

- **R.** `is_token_in_rank[t, r]` is true exactly when some selected expert of
  token `t` lies in the expert range of rank `r` (`layout.cu:81-93`). An index
  of `-1` matches no range.
- **R.** `num_tokens_per_rank[r]` counts the tokens with
  `is_token_in_rank[t, r]` (`layout.cu:94`, `105-110`): a token is counted once
  per rank, also when it selects several experts there.
- **R.** `num_tokens_per_expert[e]` counts the selections of expert `e`
  (`layout.cu:31-49`).

**Dispatch** (`intranode.cu:11-159` `notify_dispatch`, `197-532` `dispatch`).

- **R.** `notify_dispatch` writes this rank's per-destination counts into the
  buffer of every destination, waits at a barrier, turns the column of the
  local buffer into a prefix sum over source ranks (`rank_prefix_matrix`,
  `intranode.cu:41-64`), returns the total row count and the per-expert counts
  to the host (`63`, `66-75`), and counts per channel how many tokens go to
  each rank (`channel_prefix_matrix`, `90-111`).
- **R.** `dispatch` runs `num_sms` blocks of 768 threads; even blocks send, odd
  blocks receive, and block pair `c` is channel `c` (`223-233`, `562`). A
  channel owns a contiguous token range (`utils.cuh:434-438`).
- **R.** A sender walks its token range in increasing token order and, for
  each destination rank, copies every token with `is_token_in_rank` into the
  next slot of a ring buffer that lives on the destination
  (`intranode.cu:320-391`): the token row, the token index (`364`), the
  expert indices rewritten to indices local to the destination, `-1` for
  experts of other ranks (`369-373`), and the weights, `0` where the index is
  `-1` (`376-378`). It publishes the new tail with a release store at system
  scope (`397`).
- **R.** A receiver reads the tail with an acquire load at system scope
  (`437`) and copies the new slots to rows `total_offset + i` of the output,
  where `total_offset` starts at the prefix of the source rank plus the prefix
  of the channel (`411`, `416-426`, `464`, `486`, `494-497`) and advances by
  the number of rows copied (`510-511`).
- **I.** The row that a token gets is therefore fixed by the routing table:
  the rows of a destination are ordered by source rank, then by token index,
  and row `p` holding token `t` of rank `s` satisfies
  `p = (number of tokens that ranks below s send to d) + (number of tokens t' < t of s sent to d)`.
  It does not depend on the schedule. The test of the repository checks the
  grouping by source rank (`tests/test_intranode.py:80-86`).
- **R.** Outputs: `recv_x`, `recv_topk_idx`, `recv_topk_weights`,
  `recv_src_idx` (token index per row), `send_head[t, r]` (the position of
  token `t` in the stream of its channel to rank `r`, or `-1`;
  `intranode.cu:344-346`), the two prefix matrices and the per-expert counts
  rounded up to `expert_alignment` (`73`). With `num_worst_tokens > 0` the
  output has that many rows and the unused rows of `recv_topk_idx` are `-1`
  (`521-531`).
- **R.** The payload may also be FP8 with per-token scales, which travel in a
  buffer of their own (`276-277`, `381-386`, `500-507`).

**Combine** (`intranode.cu:612-689` `cached_notify_combine`, `691-1032`
`combine`).

- **R.** It needs the handle of the dispatch: both prefix matrices,
  `recv_src_idx` and `send_head` (`deep_ep/buffer.py:442-449`).
- **R.** A sender on rank `d` sends its rows back in row order, per source
  rank and channel, into a ring buffer on the source rank
  (`intranode.cu:766-824`).
- **R.** The receiver of the source rank computes, for each of its tokens `t`
  and each element, `bias_0 + bias_1` in `float`, then adds the returned rows
  of the ranks `r` with `send_head[t, r] >= 0` in increasing `r`, each
  converted to `float`, and converts the sum back to the element type
  (`925-933`, `960-981`). A missing bias is zero (`948-951`). A token that was
  sent nowhere gets the bias alone.
- **R.** No weight is applied: the interface says "addition **without**
  weights" (`deep_ep/buffer.py:413`). The weights travel separately:
  `recv_topk_weights[t, k]` is the sum over the same ranks of the weight
  vectors that the caller passed to `combine` (`intranode.cu:1012-1019`).
- **R.** The repository's test passes the dispatched rows back unchanged and
  expects each token times the number of ranks it was sent to
  (`tests/test_intranode.py:175-177`).
- **R.** The combine payload is BF16 (`deep_ep/buffer.py:420`).

**Launch.** **R.** One process per rank, started by
`torch.multiprocessing.spawn` (`tests/test_intranode.py:311`), a
`torch.distributed` process group, and one buffer per rank shared with the
other ranks through CUDA inter-process handles (`deep_ep/buffer.py:100-101`,
`135`; `csrc/deep_ep.cpp:98`, `113`). NVSHMEM is used only for inter-node and
low-latency modes (`deep_ep/buffer.py:105`; `README.md:82`). Rank counts 2, 4
and 8 are supported (`csrc/kernels/launch.cuh:54-65`).

### 3.3 Which sources implement it

| Source | Relation to version 1 normal mode, intranode |
| --- | --- |
| TileScale `9ba8a605`, `examples/distributed/deepseek_deepep/intranode/` | **R.** A port of it. `deepep.md:5-8` lists "Intranode Normal Mode" as done and "Internode Normal Mode" and "Low-latency Mode" as not done. The kernels correspond one to one (section 4.2), keep the names, the block and thread counts and the comments of the original, and the example asserts bit-exact equality of every output with `deep_ep` (`intranode/example_intranode.py:48-133`). |
| TileScale `c0f80cc5`, `examples/distributed/deepep_v2/` | **R.** A port of EPv2: "following DeepEP EPv2's `impls/dispatch.cuh` and `impls/combine.cuh`" (`README.md:3-4`). Out of scope by the decision of 2026-11-12. |
| Triton-distributed `7908e4ea`, `kernels/nvidia/ep_a2a_intra_node.py` | **I.** Not a port. Same task, different design: one receive row per (token, expert), grouped by expert; a token is transferred once per rank and duplicated locally; combine reads the peers' buffers and sums over the top-k slots; no ring buffer, no channel, no head and tail (section 4.4). |
| Triton-distributed `7908e4ea`, `tutorials/04-deepseek-infer-all2all.py` and `kernels/nvidia/low_latency_all_to_all*.py` | **R.** The tutorial's own title is "Low Latency All-to-All Communication" (`tutorials/04-deepseek-infer-all2all.py:26-27`), and the README compares it with DeepEP for inference (`README.md:109-110`). They belong with the low-latency mode, which the decision leaves out. |

## 4. Candidate sources

### 4.1 DeepEP version 1 (the reference; not a front end this project supports)

- **Files (R):** `csrc/kernels/layout.cu` (153 lines), `csrc/kernels/intranode.cu`
  (1103 lines), helpers in `csrc/kernels/utils.cuh`, `launch.cuh`,
  `configs.cuh`, `buffer.cuh`; host side `csrc/deep_ep.cpp`,
  `deep_ep/buffer.py`; test `tests/test_intranode.py`.
- **Kernels (R):** `layout::get_dispatch_layout`; `intranode::notify_dispatch`,
  `cached_notify_dispatch`, `dispatch`, `cached_notify_combine`, `combine`.
  Section 3.2 says what each does.
- **Language:** CUDA C++ with inline PTX. The hard constraints of the map put
  it outside what the project may prove. Its role is to define "version 1" and
  to serve as the comparison base of the port.
- **Ties to a GPU generation (R):**
  - Requirements: "Ampere (SM80), Hopper (SM90) GPUs, or other architectures
    with SM90 PTX ISA support"; CUDA 12.3 or newer for SM90 (`README.md:45-49`).
    Measurements are reported on H800 (`README.md:15`).
  - Without `DISABLE_SM90_FEATURES` the two main kernels use TMA copies and
    `mbarrier` (`intranode.cu:280-293`, `465-479`, `983-1009`) and are launched
    with `cudaLaunchKernelEx` and a cluster dimension (`launch.cuh:7-29`).
  - The loads named `ld_nc_global` read data that a peer writes through the
    non-coherent cache; the README calls this "an undefined-behavior PTX usage"
    whose correctness "is tested to be guaranteed with `.L1::no_allocate` on
    Hopper architectures" (`README.md:316`). `setup.py:80-82` disables it by
    default (`DISABLE_AGGRESSIVE_PTX_INSTRS`).
- **Ties to package versions (R):** PyTorch 2.1 or newer; NVSHMEM only for the
  modes that are out of scope here.

### 4.2 TileScale `wt/deepep-internode` at `9ba8a605` (the version 1 port)

Directory `examples/distributed/deepseek_deepep/`. Line numbers are in that
directory.

| File | Lines | Content |
| --- | --- | --- |
| `intranode/get_dispatch_layout.py` | 154 | Kernel `get_dispatch_layout_main` and its host function |
| `intranode/dispatch.py` | 911 | Kernels `notify_dispatch_main`, `cached_notify_dispatch_main`, `dispatch_main`, `cached_dispatch_main`; host function `intranode_dispatch` |
| `intranode/combine.py` | 415 | Kernels `cached_notify_combine_main`, `combine_main`; host function `intranode_combine` |
| `buffer.py` | 310 | Class `EPBuffer`: allocator, symmetric buffers, counters, methods `get_dispatch_layout`, `dispatch`, `combine` |
| `deepep_utils.py` | 321 | Tuning table `Config`, input generators, benchmark helper, an inline C++ extension that waits for the counters |
| `intranode/example_intranode.py` | 262 | Launcher and comparison with `deep_ep` |
| `intranode/test_intranode.py` | 14 | Pytest wrapper |
| `internode/get_dispatch_layout.py` | 270 | Layout kernel with the extra per-node count; the only inter-node file |
| `deepep.md` | 191 | Description |

**Kernels and their counterparts (R for the lines, I for "corresponds").**

| Kernel (file, line of the function) | Launch shape | Counterpart in DeepEP `567632dd` | What it does | Differences read in the source |
| --- | --- | --- | --- | --- |
| `get_dispatch_layout_main` (`intranode/get_dispatch_layout.py:78`) | `ceil(E/4) + ceil(R/8)` blocks, 256 threads (`69-72`, `84`) | `layout.cu:10` | Routing layout of section 3.2 | No barrier between the per-thread counts and the sum over threads (`96-108`, `127-152`); the original has `__syncthreads()` (`layout.cu:40`, `101`). Whether the compiler inserts one is **U**. No per-node count. |
| `notify_dispatch_main` (`intranode/dispatch.py:37`) | `R + 1` blocks, 128 threads (`30`, `55`) | `intranode.cu:12` | Exchanges counts, builds both prefix matrices, returns the counts to the host, zeroes four metadata buffers | The counts go to the peers with `T.st(..., dst_pe=tx)` (`66-68`); barriers are `T.sync_blocks` and `T.barrier_blocks` (`61`, `70`, `99`). |
| `cached_notify_dispatch_main` (`intranode/dispatch.py:192`) | 1 block, 128 threads (`200`) | `intranode.cu:162` | Zeroes the metadata buffers between two barriers | None found |
| `dispatch_main` (`intranode/dispatch.py:270`) | `num_sms` blocks (20 by default), 768 threads (`255`, `300`) | `intranode.cu:198` | Dispatch of section 3.2, BF16 only | No FP8 scales (`298`, `403`, `504`); no `num_worst_tokens`; no timeout; the row copy is `T.put_warp` (`363-370`, `471-478`) and TMA lowering is disabled (`239-244`); the wait for a free slot is `T.wait_ge` on the peer's head (`334-338`). |
| `cached_dispatch_main` (`intranode/dispatch.py:547`) | as above | `intranode.cu:198` with a handle | Dispatch of the rows only, reusing the layout | The thread that publishes the tail and the head is chosen by `T.shuffle_elect(96)` (`653`, `739`): 96 is the thread count per rank for 8 ranks. **I.** With 2 ranks (384 threads per rank) four threads of a rank store the same value. |
| `cached_notify_combine_main` (`intranode/combine.py:25`) | `num_sms/2 + 1` blocks, `max(128, 32 R)` threads (`19-20`, `32`) | `intranode.cu:613` | Zeroes head and tail between two barriers; rewrites the `-1` entries of `send_head` to the next expected position | None found |
| `combine_main` (`intranode/combine.py:120`) | `num_sms` blocks, 768 threads (`109`, `141`) | `intranode.cu:692` | Combine of section 3.2 | No bias (`126`, `318`); needs `hidden % 8 == 0` (`117`); the sum starts from zero and adds the returned rows in increasing rank in `float32` (`306-325`); the tail is published under `T.shuffle_elect(96)` (`217`), with the same consequence for 2 ranks; the receivers wait with `T.warp_any` on a shared tail read with acquire semantics at block scope (`284-287`). |
| `get_dispatch_layout_main` (`internode/get_dispatch_layout.py:83`) | as the intranode one (`73`, `90`) | `layout.cu:10` | The layout with the per-node count | Not used by the intranode path |

**Launch requirements (R).**

- One process per rank: `torch.multiprocessing.spawn(run, args=(num_ranks, args), nprocs=num_ranks)`
  (`intranode/example_intranode.py:258`); each calls
  `init_dist(local_rank, num_local_ranks)`, which creates an NCCL process
  group over TCP and binds the process to GPU `local_rank`
  (`tilelang/distributed/utils.py:42-63` at `9ba8a605`).
- One GPU per rank; "all ranks must be within one node and NVLink-visible"
  (`deepep.md:31`); at most 8 ranks (`buffer.py:57`); `Config` has entries for
  2, 4 and 8 ranks (`deepep_utils.py:46-48`, `76-78`).
- Communication: a symmetric heap per rank, allocated on the device and
  mapped into the peers through CUDA inter-process handles that travel over
  the process group (`tilelang/utils/allocator.py:147-154` at `9ba8a605`);
  the kernels address a peer with `dst_pe` / `peer`. NVSHMEM is linked only
  when `TILELANG_USE_NVSHMEM=1` (`tilelang/engine/lower.py:90-99`); the
  documented command sets `TILELANG_USE_DISTRIBUTED=1` alone (`deepep.md:139`).
- Inputs: `x` in BF16, `topk_idx` in `int64`, contiguous and two-dimensional,
  `topk_weights` in `float32`; `num_experts % num_ranks == 0`
  (`deepep.md:33`, `171-175`).

**Ties to a GPU generation.**

- **R.** No file of the directory names an architecture-specific feature; the
  only hardware statement is the measurement on "8xH100 on NVL"
  (`deepep.md:14`). The search of ticket A0.2 for SM100, Blackwell, `tcgen05`,
  TMEM and cluster launch found no match.
- **R.** TMA lowering and warp specialization are switched off in every kernel
  (`pass_configs` at `intranode/dispatch.py:23`, `189`, `239-244`, `519-524`;
  `intranode/combine.py:17`, `90-95`; `intranode/get_dispatch_layout.py:63`).
- **R.** The kernels read peer-written buffers with `T.ld(..., nc=True)`:
  `channel_src_idx_buffers` (`intranode/dispatch.py:485-489`),
  `channel_x_buffers` (`intranode/combine.py:312-316`) and
  `channel_topk_weights_buffers` (`intranode/combine.py:333-337`). This is the
  usage that DeepEP documents as tested on Hopper only (section 4.1), and the
  documentation of `T.ld` on TileScale `main` says that `nc=True` "must not be
  used on memory a peer may write during the kernel"
  (`tilelang/language/distributed/comm.py:82-85` at `71e71457`).
- **U.** What the compiler emits for these kernels. A missing mention is not
  evidence that a file runs on sm_90; only a run is.

**Ties to package versions other than the installed environments.**

- **R.** The branch is not based on the `main` that the `tilescale`
  environment was built from. GitHub's comparison of `71e71457` with `9ba8a605`
  reports "diverged", 3 commits ahead and 4 behind, with merge base `7771e606`;
  the 4 commits on `main` include "[Release] TileScale 0.0.2 on upstream
  TileLang 550e25d" (`4e42b835`). The branch has `VERSION` `0.1.7.post1`,
  package name `tilelang` and TVM submodule `23bce012`; `main` has `VERSION`
  `0.0.2`, package name `tilescale` and TVM submodule `ec7f7bd9`.
- **R.** Names the port uses that do not exist on `main` at `71e71457`:

  | Use in the port | State on `main` at `71e71457` |
  | --- | --- |
  | `from tilelang.distributed.utils import create_mapped_tensor` (`buffer.py:9`, used at `152-153`) | No module `tilelang/distributed/utils.py`; no function of that name or of the name it wraps, `create_host_device_tensor`, anywhere under `tilelang/` |
  | `from tilelang.distributed.utils import init_dist` (`intranode/example_intranode.py:10`) | `init_dist` is in `tilelang/distributed/host.py:35`, exported by `tilelang.distributed` |
  | `T.warp_any(...)` (`intranode/combine.py:285`) | No `warp_any`; the warp vote is `any_sync(predicate, mask)` (`tilelang/language/builtin.py:1044`) |
  | `tilelang.testing.requires_distributed` (`intranode/test_intranode.py:6`) | Absent |
  | Environment variable `TILELANG_USE_DISTRIBUTED` | Not read anywhere under `tilelang/` |

- **R.** Present on both sides, and the port's calls fit the signatures of
  `main` (which adds optional parameters to some): `T.st`, `T.ld`,
  `T.put_warp`, `T.wait_ge`, `T.wait_ne`, `T.sync_blocks`, `T.barrier_blocks`,
  `T.sync_threads(barrier_id, arrive_count)`, `T.sync_warp`,
  `T.shuffle_elect`, `T.warp_reduce_sum`, `T.tvm_warp_shuffle`,
  `tilelang.get_allocator`, `tilelang.tensor`. `main` adds checks: `T.ld` with
  `nc=True` asserts the default scope and semantic
  (`tilelang/language/distributed/comm.py:90-98`), which the port's calls
  satisfy.
- **R.** `deepep_utils.py:319-321` compiles a C++ extension with
  `torch.utils.cpp_extension.load_inline` when the module is imported, so a
  host compiler is needed at run time.
- **R.** The example imports `deep_ep` and fails without it
  (`intranode/example_intranode.py:30-33`); `EPBuffer` itself does not.
- **R.** Further imports: `tilescale_ext` (a native extension built with the
  package: `tilescale_ext/__init__.py`), `cuda-python` and `packaging`
  (`tilelang/distributed/utils.py:14-24` at `9ba8a605`), `tvm_ffi`
  (`intranode/dispatch.py:14`, unused).

### 4.3 TileScale `deepep-v2-intranode` at `c0f80cc5` (the EPv2 port)

- **Files (R):** `examples/distributed/deepep_v2/kernels/dispatch.py` (583
  lines, kernel `main` at line 236 and helper `_dedup_leader` at 154),
  `kernels/combine.py` (243 lines, kernel `main` at 89), `buffer.py`,
  `reference.py`, two examples, a test and `README.md`.
- **What the kernels do (R, from `README.md:1-30`):** dispatch sends each token
  to the ranks that own its experts and lands it at its final row; combine
  sends the contributions back and reduces per source token. Launch shape:
  `num_sms` blocks (`kernels/dispatch.py:285`, `kernels/combine.py:108`, `130`)
  and, for the reduction, one block per token (`kernels/combine.py:217`).
- **Ties (R):** the README scopes it to "intranode NVLink on Blackwell
  (SM100)" (`README.md:4`) and measures on B200 (`README.md:203`, `502`); no
  source file mentions an SM100-only feature (ticket A0.2). The branch is 29
  commits ahead of `main` at `71e71457` and 0 behind, so it is the only DeepEP
  source written against the API of the installed environment. Upstream EPv2
  itself lists SM90 results (DeepEP `README.md` at `a56d6156`, performance
  table).
- **Status:** out of the proof scope by the decision of 2026-11-12. The frozen
  MoE specification was written against it.

### 4.4 Triton-distributed at `7908e4ea`

- **Files (R):** `python/triton_dist/kernels/nvidia/ep_a2a_intra_node.py` (480
  lines), `ep_a2a.py` (1035), `ep_all2all_fused.py` (2071),
  `low_latency_all_to_all.py` (279), `low_latency_all_to_all_v2.py` (696),
  `low_latency_allgather.py` (987), `tutorials/04-deepseek-infer-all2all.py`
  (655); host layer `python/triton_dist/layers/nvidia/ep_a2a_layer.py`; test
  `python/triton_dist/test/nvidia/test_ep_a2a.py`.
- **The intra-node kernels (R),** `ep_a2a_intra_node.py`:

  | Kernel (line) | What it does |
  | --- | --- |
  | `kernel_get_ag_splits_and_recv_offset_intra_node` (293) | Sends this rank's per-expert counts to every rank, waits at a barrier, and builds `recv_buf_offset_per_expert[rank, expert, source rank]`: the first row of each (expert, source rank) group (`319-372`, `441-446`) |
  | `kernel_dispatch_token_intra_node` (39) | For each token and top-k slot: takes a row in the group of the expert, by an atomic increment or from precomputed indices (`81-85`); transfers the token once per destination rank with `putmem_signal_warp` (`108-112`) and, for a further slot on the same rank, stores only the index of the first row (`113-116`); sends one weight per row (`123-125`) |
  | `kernel_skipped_token_local_dispatch_intra_node` (129) | On the destination, copies the first row into the rows whose transfer was skipped (`151-162`) |
  | `kernel_skipped_token_inplace_local_combine_intra_node` (174) | Optional: adds the rows of one token on one rank in place before the combine (`192-215`) |
  | `kernel_combine_token_intra_node` (219) | On the source rank, for each token: reads the rows from the peers' buffers with `dl.symm_at`, adds them over the top-k slots in `float32` from zero and stores the sum (`248-289`) |

  The layer launches each with `grid = (num_sm,)` and `num_warps=32`
  (`layers/nvidia/ep_a2a_layer.py:373`, `400-442`, `518`, `547-572`).
- **Relation to version 1 (I).** The task is the same and the combine is also
  an unweighted sum. The layout differs (a row per (token, expert), grouped by
  expert and source rank, instead of a row per (token, rank) ordered by source
  rank and token); with the atomic increment the row inside a group depends on
  the schedule; and the protocol differs (signals and one barrier instead of
  channels with ring buffers, head and tail). A proof of these kernels is not
  a proof of DeepEP version 1.
- **Launch requirements (R):** `bash scripts/launch.sh <script>`, which starts
  one process per GPU; NVSHMEM symmetric memory (`nvshmem_create_tensor`,
  `layers/nvidia/ep_a2a_layer.py:126-161`); the documented run sets
  `NVSHMEM_SYMMETRIC_SIZE` (`docs/kernels/nvidia/ep_a2a_intra_node.rst`, "Run
  Example").
- **Ties:** it is the pinned commit of the installed `triton-dist`
  environment. No sm_100 mention (ticket A0.2). The remaining files of the
  list hold the inter-node kernels, the fused kernels and the low-latency
  kernels; section 5 lists their functions, and their bodies were not read for
  this ticket.

## 5. Kernel census

Every function decorated with `prim_func`, `triton.jit` or `triton_dist.jit`
in the 13 files below has a row: 56 kernels, plus 4 helpers that a kernel
reaches. The count of decorators found by a text search in the same files is
also 56. The rows come from `scripts/census/scan.py`; "Calls" lists the
resolved names a function calls, without Python builtins, methods of local
values and `torch`. Prefixes: `T.` is `tilelang.language.`, `tl.` is
`triton.language.`, `dl.` is `triton_dist.language.`, `libshmem_device.` is
`triton_dist.language.extra.libshmem_device.`. A name without a prefix is
imported from `triton_dist.language.extra.language_extra`,
`triton_dist.language.extra.cuda.language_extra` or
`triton_dist.kernels.nvidia.common_ops`, or is a function of the same file.

Launch shapes are in sections 4.2 to 4.4 for the files read there. For
`ep_a2a.py`, `ep_all2all_fused.py`, the three `low_latency_*` files and
tutorial 04 the launch shape was not collected.

**TileScale `9ba8a605`, `examples/distributed/deepseek_deepep/intranode/get_dispatch_layout.py`**

| Line | Function | Decorators | Calls |
| --- | --- | --- | --- |
| 78 | `get_dispatch_layout_kernel.get_dispatch_layout_main` | `T.prim_func` | `T.Kernel`, `T.Tensor`, `T.alloc_local`, `T.alloc_shared`, `T.alloc_var`, `T.ceildiv`, `T.clear`, `T.get_thread_binding`, `T.min`, `T.serial` |

**TileScale `9ba8a605`, `examples/distributed/deepseek_deepep/intranode/dispatch.py`**

| Line | Function | Decorators | Calls |
| --- | --- | --- | --- |
| 37 | `notify_dispatch_kernel.notify_dispatch_main` | `T.prim_func` | `T.Kernel`, `T.Tensor`, `T.alloc_local`, `T.alloc_var`, `T.barrier_blocks`, `T.ceildiv`, `T.clear`, `T.copy`, `T.get_thread_binding`, `T.min`, `T.serial`, `T.shuffle_elect`, `T.st`, `T.sync_blocks`, `T.sync_threads`, `T.truncdiv`, `T.warp_reduce_sum` |
| 192 | `cached_notify_dispatch_kernel.cached_notify_dispatch_main` | `T.prim_func` | `T.Kernel`, `T.Tensor`, `T.barrier_blocks`, `T.clear`, `T.sync_blocks` |
| 270 | `dispatch_kernel.dispatch_main` | `T.prim_func` | `T.Kernel`, `T.Tensor`, `T.address_of`, `T.alloc_shared`, `T.alloc_var`, `T.cast`, `T.get_thread_binding`, `T.if_then_else`, `T.ld`, `T.min`, `T.put_warp`, `T.serial`, `T.shuffle_elect`, `T.st`, `T.sync_threads`, `T.sync_warp`, `T.truncdiv`, `T.tvm_warp_shuffle`, `T.wait_ge`, `T.wait_ne` |
| 547 | `cached_dispatch_kernel.cached_dispatch_main` | `T.prim_func` | `T.Kernel`, `T.Tensor`, `T.address_of`, `T.alloc_shared`, `T.alloc_var`, `T.ceildiv`, `T.get_thread_binding`, `T.if_then_else`, `T.ld`, `T.min`, `T.put_warp`, `T.serial`, `T.shuffle_elect`, `T.st`, `T.sync_threads`, `T.sync_warp`, `T.tvm_warp_shuffle`, `T.wait_ge`, `T.wait_ne` |

**TileScale `9ba8a605`, `examples/distributed/deepseek_deepep/intranode/combine.py`**

| Line | Function | Decorators | Calls |
| --- | --- | --- | --- |
| 25 | `cached_notify_combine_kernel.cached_notify_combine_main` | `T.prim_func` | `T.Kernel`, `T.Tensor`, `T.alloc_var`, `T.barrier_blocks`, `T.ceildiv`, `T.clear`, `T.get_thread_binding`, `T.ld`, `T.min`, `T.serial`, `T.sync_blocks`, `T.thread_return`, `T.tvm_warp_shuffle` |
| 120 | `combine_kernel.combine_main` | `T.prim_func` | `T.Kernel`, `T.Tensor`, `T.address_of`, `T.alloc_local`, `T.alloc_shared`, `T.alloc_var`, `T.clear`, `T.get_thread_binding`, `T.if_then_else`, `T.ld`, `T.min`, `T.put_warp`, `T.serial`, `T.shuffle_elect`, `T.st`, `T.sync_threads`, `T.sync_warp`, `T.truncdiv`, `T.tvm_warp_shuffle`, `T.vectorized`, `T.wait_ge`, `T.warp_any` |

**TileScale `9ba8a605`, `examples/distributed/deepseek_deepep/internode/get_dispatch_layout.py`**

| Line | Function | Decorators | Calls |
| --- | --- | --- | --- |
| 83 | `get_dispatch_layout_kernel.get_dispatch_layout_main` | `T.prim_func` | `T.Kernel`, `T.Tensor`, `T.alloc_local`, `T.alloc_shared`, `T.alloc_var`, `T.ceildiv`, `T.clear`, `T.get_thread_binding`, `T.min`, `T.serial` |

**TileScale `c0f80cc5`, `examples/distributed/deepep_v2/kernels/dispatch.py`**

| Line | Function | Decorators | Calls |
| --- | --- | --- | --- |
| 154 | `_dedup_leader` (helper) | none | `T.Cast`, `T.if_then_else`, `T.match_any_sync` |
| 236 | `dispatch_kernel.main` | `T.prim_func` | `T.Cast`, `T.Kernel`, `T.Tensor`, `T.address_of`, `T.alloc_shared`, `T.alloc_var`, `T.atom_add`, `T.barrier_blocks`, `T.ceildiv`, `T.fence_sys`, `T.get_rank`, `T.get_thread_binding`, `T.put_warp`, `T.serial`, `T.shfl_sync`, `T.st`, `T.sync_grid`, `T.sync_threads`, `T.sync_warp`, `T.wait_ge`, `_dedup_leader` |

**TileScale `c0f80cc5`, `examples/distributed/deepep_v2/kernels/combine.py`**

| Line | Function | Decorators | Calls |
| --- | --- | --- | --- |
| 89 | `combine_kernel.main` | `T.prim_func` | `T.Cast`, `T.Kernel`, `T.Parallel`, `T.Tensor`, `T.address_of`, `T.alloc_fragment`, `T.alloc_var`, `T.atom_add`, `T.barrier_blocks`, `T.ceildiv`, `T.clear`, `T.copy`, `T.get_rank`, `T.get_thread_binding`, `T.min`, `T.put_warp`, `T.serial`, `T.sync_grid`, `T.sync_warp` |

**Triton-distributed `7908e4ea`, `python/triton_dist/kernels/nvidia/ep_a2a_intra_node.py`**

| Line | Function | Decorators | Calls |
| --- | --- | --- | --- |
| 39 | `kernel_dispatch_token_intra_node` | `triton_dist.jit` | `atomic_add_per_warp`, `dl.num_ranks`, `dl.rank`, `dl.symm_at`, `ld`, `libshmem_device.putmem_signal_warp`, `libshmem_device.putmem_warp`, `st`, `threads_per_warp`, `tid`, `tl.constexpr`, `tl.load`, `tl.num_programs`, `tl.program_id` |
| 129 | `kernel_skipped_token_local_dispatch_intra_node` | `triton_dist.jit` | `dl.ld_vector`, `dl.rank`, `dl.st_vector`, `ld`, `libshmem_device.putmem_warp`, `st`, `threads_per_warp`, `tid`, `tl.num_programs`, `tl.program_id` |
| 174 | `kernel_skipped_token_inplace_local_combine_intra_node` | `triton_dist.jit` | `dl.ld_vector`, `dl.st_vector`, `dl.zeros_vector`, `ld`, `threads_per_warp`, `tid`, `tl.num_programs`, `tl.program_id`, `tl.static_assert` |
| 219 | `kernel_combine_token_intra_node` | `triton_dist.jit` | `dl.ld_vector`, `dl.rank`, `dl.st_vector`, `dl.symm_at`, `dl.zeros_vector`, `ld`, `threads_per_warp`, `tid`, `tl.load`, `tl.multiple_of`, `tl.num_programs`, `tl.program_id`, `tl.static_assert` |
| 293 | `kernel_get_ag_splits_and_recv_offset_intra_node` | `triton_dist.jit` | `__syncthreads`, `barrier_on_this_grid`, `dl.consume_token`, `dl.num_ranks`, `dl.rank`, `dl.wait`, `ld`, `libshmem_device.barrier_all_block`, `libshmem_device.putmem_signal_nbi_block`, `st`, `tid`, `tl.arange`, `tl.constexpr`, `tl.cumsum`, `tl.extra.cuda.num_warps`, `tl.load`, `tl.num_programs`, `tl.program_id`, `tl.static_assert`, `tl.store`, `tl.sum` |

**Triton-distributed `7908e4ea`, `python/triton_dist/kernels/nvidia/ep_a2a.py`**

| Line | Function | Decorators | Calls |
| --- | --- | --- | --- |
| 79 | `kernel_dispatch_token` | `triton_dist.tools.aot_compile_spaces`, `triton_dist.jit` | `__syncthreads`, `atomic_add_per_warp`, `dl.num_ranks`, `dl.rank`, `ld`, `ld_b32`, `libshmem_device.fence`, `libshmem_device.putmem_nbi_block`, `libshmem_device.putmem_nbi_warp`, `libshmem_device.putmem_warp`, `libshmem_device.signal_op`, `libshmem_device.signal_wait_until`, `membar`, `num_warps`, `st`, `tid`, `tl.constexpr`, `tl.load`, `tl.num_programs`, `tl.program_id` |
| 214 | `kernel_combine_token` | `triton_dist.tools.aot_compile_spaces`, `triton_dist.jit` | `__syncthreads`, `dl.ld_vector`, `dl.num_ranks`, `dl.rank`, `dl.st_vector`, `dl.symm_at`, `dl.zeros_vector`, `ld`, `ld_acquire`, `ld_b32`, `libshmem_device.putmem_nbi_warp`, `num_warps`, `threads_per_warp`, `tid`, `tl.arange`, `tl.atomic_add`, `tl.load`, `tl.multiple_of`, `tl.num_programs`, `tl.program_id`, `tl.store`, `tl.zeros` |
| 329 | `copy_1d_tilewise_kernel` | `triton_dist.jit` | `tl.arange`, `tl.load`, `tl.num_programs`, `tl.program_id`, `tl.store` |
| 382 | `kernel_get_ag_splits_and_recv_offset` | `triton_dist.tools.aot_compile_spaces`, `triton_dist.jit` | `__syncthreads`, `barrier_on_this_grid`, `copy_1d_tilewise_kernel`, `dl.consume_token`, `dl.num_ranks`, `dl.rank`, `dl.wait`, `ld`, `libshmem_device.barrier_all_block`, `libshmem_device.fence`, `libshmem_device.putmem_nbi_block`, `libshmem_device.putmem_signal_nbi_block`, `st`, `tid`, `tl.arange`, `tl.constexpr`, `tl.cumsum`, `tl.extra.cuda.num_warps`, `tl.load`, `tl.num_programs`, `tl.program_id`, `tl.static_assert`, `tl.store`, `tl.sum` |
| 582 | `kernel_bincount` | `triton_dist.tools.aot_compile_spaces`, `triton_dist.jit` | `atomic_add`, `dl.simt_exec_region`, `ld`, `tl.num_programs`, `tl.program_id` |
| 605 | `kernel_get_dispatch_send_reqs` | `triton_dist.tools.aot_compile_spaces`, `triton_dist.jit` | `__syncthreads`, `dl.num_ranks`, `dl.rank`, `ld`, `st`, `tid`, `tl.arange`, `tl.cdiv`, `tl.load`, `tl.num_programs`, `tl.program_id`, `tl.store`, `tl.sum`, `tl.where`, `triton.next_power_of_2` |

**Triton-distributed `7908e4ea`, `python/triton_dist/kernels/nvidia/ep_all2all_fused.py`**

| Line | Function | Decorators | Calls |
| --- | --- | --- | --- |
| 41 | `sync_warp` (helper) | `tl.core.extern` | `tl.inline_asm_elementwise` |
| 57 | `consume_token` (helper) | `tl.core.extern` | `tl.cast`, `tl.inline_asm_elementwise`, `tl.pointer_type` |
| 73 | `tile_kernel_dispatch_token_intra_node` | `triton_dist.jit` | `atomic_add`, `atomic_add_per_warp`, `dl.num_ranks`, `dl.rank`, `ld`, `libshmem_device.fence`, `libshmem_device.putmem_warp`, `libshmem_device.signal_op`, `st`, `sync_warp`, `tid`, `tl.load` |
| 168 | `tile_kernel_dispatch_token_intra_node_two_stage` | `triton_dist.jit` | `__syncthreads`, `atomic_add_per_warp`, `dl.num_ranks`, `dl.rank`, `dl.symm_at`, `ld`, `ld_acquire`, `libshmem_device.fence`, `libshmem_device.putmem_warp`, `st`, `sync_warp`, `tid`, `tl.load` |
| 308 | `tile_kernel_gather_combine_token_intra_node` | `triton_dist.jit` | `consume_token`, `dl.num_ranks`, `dl.rank`, `dl.symm_at`, `ld_acquire`, `ld_b32`, `st`, `tid`, `tl.cast`, `tl.load`, `tl.pointer_type`, `tl.static_assert` |
| 406 | `tile_kernel_scatter_token_intra_node` | `triton_dist.jit` | `dl.rank`, `dl.symm_at`, `ld_acquire`, `ld_b32`, `libshmem_device.fence`, `st`, `sync_warp`, `tid`, `tl.cast`, `tl.load`, `tl.pointer_type`, `tl.static_assert` |
| 485 | `tile_kernel_topk_reduce_token_intra_node` | `triton_dist.jit` | `dl.rank`, `ld_acquire`, `ld_b32`, `sync_warp`, `tid`, `tl.load`, `tl.static_assert` |
| 561 | `dot_k_const` | `triton_dist.jit` | `tl.arange`, `tl.cdiv`, `tl.dot`, `tl.load`, `tl.store`, `tl.zeros` |
| 599 | `tile_kernel_moe_grouped_gemm_nk_const` | `triton_dist.jit` | `__syncthreads`, `dl.num_ranks`, `dot_k_const`, `ld_acquire`, `st`, `tid`, `tl.arange`, `tl.cdiv`, `tl.load`, `tl.swizzle2d` |
| 722 | `transposed_dot` | `triton_dist.jit` | `tl.arange`, `tl.dot`, `tl.load`, `tl.store`, `tl.zeros` |
| 750 | `tile_kernel_transposed_moe_grouped_gemm_nk_const` | `triton_dist.jit` | `tl.arange`, `tl.cdiv`, `tl.load`, `tl.store`, `tl.swizzle2d`, `tl.zeros`, `transposed_dot` |
| 839 | `mega_kernel_dispatch_token_moe_grouped_gemm` | `triton_dist.jit` | `tid`, `tile_kernel_dispatch_token_intra_node`, `tile_kernel_dispatch_token_intra_node_two_stage`, `tile_kernel_moe_grouped_gemm_nk_const`, `tl.atomic_add`, `tl.cdiv`, `tl.load`, `triton_dist.tools.profiler.Profiler.create` |
| 1020 | `mega_kernel_moe_grouped_gemm_combine_token` | `triton_dist.jit` | `dl.num_ranks`, `dl.rank`, `tid`, `tile_kernel_gather_combine_token_intra_node`, `tile_kernel_moe_grouped_gemm_nk_const`, `tile_kernel_scatter_token_intra_node`, `tile_kernel_topk_reduce_token_intra_node`, `tl.atomic_add`, `tl.cdiv`, `tl.load`, `tl.num_programs`, `tl.program_id`, `triton_dist.tools.profiler.Profiler.create` |
| 1288 | `mega_kernel_moe_grouped_gemm_combine_token_transposed_grouped_gemm` | `triton_dist.jit` | `dl.num_ranks`, `dl.rank`, `tid`, `tile_kernel_gather_combine_token_intra_node`, `tile_kernel_moe_grouped_gemm_nk_const`, `tile_kernel_scatter_token_intra_node`, `tile_kernel_topk_reduce_token_intra_node`, `tile_kernel_transposed_moe_grouped_gemm_nk_const`, `tl.atomic_add`, `tl.cdiv`, `tl.load`, `tl.num_programs`, `tl.program_id`, `triton_dist.tools.profiler.Profiler.create` |
| 1641 | `kernel_get_ag_splits_and_recv_offset` | `triton_dist.jit` | `__syncthreads`, `atomic_add`, `dl.consume_token`, `dl.num_ranks`, `dl.rank`, `dl.symm_at`, `dl.wait`, `ld`, `libshmem_device.barrier_all_block`, `libshmem_device.fence`, `libshmem_device.putmem_nbi_block`, `libshmem_device.putmem_signal_nbi_block`, `st`, `tid`, `tl.arange`, `tl.constexpr`, `tl.cumsum`, `tl.extra.cuda.num_warps`, `tl.load`, `tl.num_programs`, `tl.program_id`, `tl.static_assert`, `tl.store`, `tl.sum` |

**Triton-distributed `7908e4ea`, `python/triton_dist/kernels/nvidia/low_latency_all_to_all.py`**

| Line | Function | Decorators | Calls |
| --- | --- | --- | --- |
| 36 | `all_to_all_kernel` | `triton_dist.jit` | `libshmem_device.fence`, `libshmem_device.putmem_nbi_block`, `libshmem_device.putmem_signal_nbi_block`, `libshmem_device.signal_op`, `libshmem_device.signal_wait_until`, `tid`, `tl.arange`, `tl.load`, `tl.program_id`, `tl.store` |

**Triton-distributed `7908e4ea`, `python/triton_dist/kernels/nvidia/low_latency_all_to_all_v2.py`**

| Line | Function | Decorators | Calls |
| --- | --- | --- | --- |
| 43 | `unpack_b32_v2` (helper) | `tl.core.extern` | `tl.inline_asm_elementwise` |
| 60 | `dispatch_postprocess_kernel_v2_for_expert` | `triton_dist.jit` | `dl.num_ranks`, `pack_b32_v2`, `tl.arange`, `tl.atomic_add`, `tl.cdiv`, `tl.constexpr`, `tl.load`, `tl.pointer_type`, `tl.static_assert`, `tl.store` |
| 156 | `dispatch_kernel_v2` | `triton_dist.jit` | `__syncthreads`, `atomic_add_per_warp`, `barrier_on_this_grid`, `dispatch_postprocess_kernel_v2_for_expert`, `dl.num_ranks`, `dl.rank`, `libshmem_device.fence`, `libshmem_device.putmem_nbi_warp`, `libshmem_device.putmem_signal_nbi_block`, `libshmem_device.signal_wait_until`, `membar`, `st`, `tid`, `tl.abs`, `tl.arange`, `tl.cast`, `tl.constexpr`, `tl.extra.cuda.num_threads`, `tl.load`, `tl.max`, `tl.num_programs`, `tl.pointer_type`, `tl.program_id`, `tl.reshape`, `tl.static_assert`, `tl.store`, `triton_dist.tools.profiler.Profiler.create` |
| 360 | `combine_kernel_v2` | `triton_dist.jit` | `__syncthreads`, `barrier_on_this_grid`, `dl.ld_vector`, `dl.num_ranks`, `dl.rank`, `dl.st_vector`, `dl.symm_at`, `ld`, `libshmem_device.fence`, `libshmem_device.putmem_nbi_warp`, `libshmem_device.signal_op`, `libshmem_device.signal_wait_until`, `membar`, `tid`, `tl.arange`, `tl.cdiv`, `tl.constexpr`, `tl.extra.cuda.utils.num_warps`, `tl.load`, `tl.num_programs`, `tl.program_id`, `tl.static_assert`, `tl.store`, `triton.next_power_of_2`, `triton_dist.tools.profiler.Profiler.create`, `unpack_b32_v2` |

**Triton-distributed `7908e4ea`, `python/triton_dist/kernels/nvidia/low_latency_allgather.py`**

| Line | Function | Decorators | Calls |
| --- | --- | --- | --- |
| 48 | `_forward_pull_kernel` | `triton_dist.jit` | `__syncthreads`, `libshmem_device.getmem_block`, `libshmem_device.signal_op`, `libshmem_device.signal_wait_until`, `tid`, `tl.cast`, `tl.pointer_type`, `tl.program_id` |
| 74 | `_forward_push_numa_2d_kernel` | `triton_dist.jit` | `__syncthreads`, `libshmem_device.putmem_signal_block`, `libshmem_device.signal_wait_until`, `tid`, `tl.cast`, `tl.pointer_type`, `tl.program_id`, `tl.static_assert` |
| 135 | `_forward_push_numa_2d_ll_kernel` | `triton_dist.jit` | `__syncthreads`, `_pack_ll_block`, `_recv_ll_block`, `libshmem_device.putmem_block`, `libshmem_device.putmem_signal_block`, `libshmem_device.signal_wait_until`, `st`, `tid`, `tl.cast`, `tl.pointer_type`, `tl.program_id`, `tl.static_assert` |
| 217 | `_forward_push_numa_2d_ll_multinode_kernel` | `triton_dist.jit` | `__syncthreads`, `_pack_ll_block`, `_recv_ll_block`, `libshmem_device.putmem_nbi_warp`, `libshmem_device.putmem_signal_block`, `libshmem_device.signal_wait_until`, `st`, `tid`, `tl.cast`, `tl.pointer_type`, `tl.program_id`, `tl.static_assert` |
| 345 | `_forward_push_2d_kernel` | `triton_dist.jit` | `__syncthreads`, `libshmem_device.putmem_signal_block`, `libshmem_device.putmem_signal_nbi_block`, `libshmem_device.signal_wait_until`, `tid`, `tl.cast`, `tl.pointer_type`, `tl.program_id` |
| 400 | `_forward_push_3d_kernel` | `triton_dist.jit` | `__syncthreads`, `_pack_ll_block`, `_recv_ll_block`, `libshmem_device.putmem_nbi_warp`, `libshmem_device.putmem_signal_block`, `libshmem_device.putmem_signal_nbi_warp`, `libshmem_device.signal_wait_until`, `st`, `tid`, `tl.cast`, `tl.pointer_type`, `tl.program_id` |
| 531 | `_recv_ll_block` | `triton_dist.jit` | `load_v4_u32`, `ntid`, `st_v2_u32`, `tid`, `tl.cast`, `tl.pointer_type` |
| 549 | `_pack_ll_block` | `triton_dist.jit` | `tl.arange`, `tl.cast`, `tl.cdiv`, `tl.full`, `tl.interleave`, `tl.load`, `tl.store` |
| 570 | `_recv_ll_and_multimem_st_block` | `triton_dist.jit` | `libshmem_device.remote_mc_ptr`, `load_v4_u32`, `multimem_st_b64`, `ntid`, `pack_b32_v2`, `tid`, `tl.cast`, `tl.pointer_type` |
| 589 | `_recv_ll_and_multimem_st_ll_block` | `triton_dist.jit` | `libshmem_device.remote_mc_ptr`, `load_v4_u32`, `multimem_st_b64`, `ntid`, `pack_b32_v2`, `tid`, `tl.cast`, `tl.pointer_type` |
| 609 | `broadcast_naive_block` | `triton_dist.jit` | `libshmem_device.remote_mc_ptr`, `load_v2_b64`, `multimem_st_b64`, `ntid`, `tid`, `tl.cast`, `tl.pointer_type` |
| 623 | `_forward_push_2d_ll_multimem_kernel` | `triton_dist.jit` | `__syncthreads`, `_pack_ll_block`, `_recv_ll_and_multimem_st_ll_block`, `_recv_ll_block`, `broadcast_naive_block`, `libshmem_device.putmem_nbi_warp`, `tid`, `tl.cast`, `tl.pointer_type`, `tl.program_id` |
| 700 | `_forward_push_2d_ll_kernel` | `triton_dist.jit` | `__syncthreads`, `_pack_ll_block`, `_recv_ll_block`, `libshmem_device.putmem_nbi_warp`, `libshmem_device.putmem_signal_block`, `libshmem_device.signal_wait_until`, `st`, `tid`, `tl.cast`, `tl.pointer_type`, `tl.program_id` |

**Triton-distributed `7908e4ea`, `tutorials/04-deepseek-infer-all2all.py`**

| Line | Function | Decorators | Calls |
| --- | --- | --- | --- |
| 78 | `ceil_div` | `triton.jit` | none |
| 94 | `all_to_all_kernel` | `triton.autotune`, `triton_dist.jit` | `ceil_div`, `libshmem_device.fence`, `libshmem_device.putmem_nbi_block`, `libshmem_device.putmem_signal_nbi_block`, `libshmem_device.signal_op`, `libshmem_device.signal_wait_until`, `tid`, `tl.abs`, `tl.arange`, `tl.cast`, `tl.load`, `tl.max`, `tl.pointer_type`, `tl.program_id`, `tl.range`, `tl.reshape`, `tl.static_range`, `tl.store` |
| 296 | `calc_gather_index._kernel` | `triton.jit` | `tl.arange`, `tl.load`, `tl.program_id`, `tl.store` |
| 336 | `_quant_kernel` | `triton.jit` | `tl.abs`, `tl.arange`, `tl.cast`, `tl.constexpr`, `tl.load`, `tl.max`, `tl.pointer_type`, `tl.program_id`, `tl.reshape`, `tl.static_range`, `tl.store` |
| 375 | `_dequant_kernel` | `triton.jit` | `tl.arange`, `tl.cast`, `tl.load`, `tl.pointer_type`, `tl.program_id`, `tl.reshape`, `tl.static_range`, `tl.store` |

## 6. Recommendation and pins

**Source for the proof:** TileScale, commit
`9ba8a605d347f600a95d0275e2639cb00122c084` (branch `wt/deepep-internode`),
files `examples/distributed/deepseek_deepep/intranode/get_dispatch_layout.py`,
`dispatch.py` and `combine.py`. The basic path is five kernels:
`get_dispatch_layout_main`, `notify_dispatch_main`, `dispatch_main`,
`cached_notify_combine_main`, `combine_main`. The cached dispatch
(`cached_notify_dispatch_main`, `cached_dispatch_main`) is a second step.

**Reference:** DeepEP, commit `b57e5e212ab75350f53c72064333e4fe1076b1da`, files
`csrc/kernels/layout.cu` and `csrc/kernels/intranode.cu`. It is the commit the
port records as its submodule and compares against, and these two files are
identical at the last version 1 commit `567632dd59810d77b3cc05553df953cc0f779799`.

**Text proposed for the pins of the map** (this ticket does not edit the map):

> DeepEP version 1, normal mode, intranode: TileScale branch
> `wt/deepep-internode` at `9ba8a605`, directory
> `examples/distributed/deepseek_deepep/intranode/`; reference
> `deepseek-ai/DeepEP` at `b57e5e21` (version 1; the intranode kernels are
> unchanged up to the last version 1 commit `567632dd`). Branch
> `deepep-v2-intranode` at `c0f80cc5` ports EPv2 and is outside the proof
> scope.

**Why this one.**

- It is the only source in a supported front end that implements the
  algorithm of version 1 (section 3.3).
- It is unmodified upstream text at a fixed commit, with a bit-exact
  comparison against the original in the same directory.
- It was measured on Hopper (H100), the generation of the H800 node.

**Costs that come with it.**

- It does not run in an installed environment (section 7).
- It is written against an older TileScale. The primitive table
  (`documents/dist/PrimitiveSources.md`) cites TileScale at `71e7145`. The
  port's primitives that have a row there (`st`, `ld`, `put_warp`, `wait_ge`,
  `wait_ne`, `barrier_blocks`, `sync_blocks`, `copy`) exist at both commits
  with the same parameters, but their docstrings at `9ba8a605` say less
  (`barrier_blocks`: "have an extra system-level fence effect",
  `tilelang/language/builtin.py:981-988` at `9ba8a605`). Using the
  documentation of `71e7145` for a kernel written at `9ba8a605` is a named
  assumption to register. `warp_any`, `shuffle_elect`, `warp_reduce_sum`,
  `tvm_warp_shuffle`, `sync_threads`, `sync_warp` and `thread_return` have no
  row in the table: it leaves base TileLang out "until a kernel in scope
  uses them" (its section on coverage), and these kernels do.
- Its non-coherent loads of peer-written buffers (section 4.2) fall under the
  rule that documentation forbids, and under the weakest semantics of the
  table.
- The combine has no bias and the dispatch no FP8 payload, so the proof
  covers the BF16 path of version 1 without bias.

**Alternatives and why not.**

| Alternative | Why not |
| --- | --- |
| DeepEP version 1 itself (`567632dd`) | CUDA C++ with inline PTX: outside every supported front end and below the line the map draws. It stays the reference. |
| A new port of version 1 to TileScale `main` (`71e71457`), made by this project from the files of `9ba8a605` | It would run in the installed environment and match the cited documentation, but it is this project's own program, not an upstream one: the claim "DeepEP version 1 as ported by TileScale" would become "our port". It needs a replacement for the host-mapped counters (`create_mapped_tensor`), `any_sync` for `warp_any`, and a comparison run against the original. It is the fallback if an environment for `9ba8a605` cannot be built. |
| TileScale `deepep-v2-intranode` (`c0f80cc5`) | EPv2, excluded by the decision; scoped to Blackwell by its README. |
| Triton-distributed `ep_a2a_intra_node` (`7908e4ea`) | Runs in an installed environment, but it is not DeepEP version 1 (section 4.4). Tickets K-D.3 to K-D.5 cover it as a kernel of its own. |
| Triton-distributed tutorial 04 and `low_latency_all_to_all*` | Low-latency mode, excluded by the decision. |

**Consequences for the other tickets of the epic** (observations; no issue was
changed). K-D.14 describes `wt/deepep-internode` as "the inter-node branch"
and depends on K-D.13 and K-D.7; the branch holds the intranode normal mode
and no inter-node dispatch or combine, so K-D.14 is where the version 1 proof
lives and it does not depend on those two. K-D.12 and K-D.13 concern the EPv2
port. K-D.2, K-D.10, K-D.11, K-D.15 and K-D.16 concern low-latency kernels.

## 7. Running it on the H800 node through `gpurun`

Updated for ticket K-D.17 on 2026-11-12. The request is written and committed;
it is not announced, and nothing of the port has run. This section keeps the
nine needs that ticket K-D.1 listed, says how the request meets each, and
where the source made it differ from that list. Line numbers are in TileScale
at `9ba8a605`; a path that starts with `intranode/`, `buffer.py` or
`deepep_utils.py` is under `examples/distributed/deepseek_deepep/`.

The request is four things:

| What | Where |
| --- | --- |
| Batch, two tests, two ranks, two repeats each | `experiments/interp_gpu/batches/deepep-v1-intranode-001/batch.json` |
| Per-rank harness | `experiments/interp_gpu/deepep_v1/harness.py` |
| Launcher `tilescale-deepep` | `scripts/gpurun/dist.py` |
| Judge and its self-test | `experiments/interp_gpu/deepep_v1/judge.py`, `test_deepep_v1.py` |

### 7.1 The nine needs

1. **An environment for the port.** Met by the GPU side on 2026-11-12: the
   environment `tilescale-deepep` reports `tilelang.__version__`
   `0.1.7.post1+cuda.git9ba8a605`, Python 3.11 and PyTorch 2.9.1+cu126, and
   imports `tilelang`, `tilescale_ext._C` and `tilelang.distributed`. No kernel
   has run in it.
2. **A pin in `gpurun`.** Met: `scripts/gpurun/env.py` pins `tilescale-deepep`
   to a `tilelang` version containing `git9ba8a60`.
3. **The kernel files.** Differs from the list: nothing is copied. The `entry`
   is the harness, a file of this repository, and it imports the port from
   the checkout the environment was built from
   (`gpurun.dist.deepep_port()`: `examples/distributed/deepseek_deepep` beside
   the `tilelang` module). **R:** the install that the pinned TileScale
   documents is editable (`docs/get_started/Installation.md:36`,
   `pip install -e . --no-build-isolation`), and a wheel holds only the
   packages `tilelang` and `tilescale_ext` (`pyproject.toml:118-121`), so the
   import needs the editable install. **U:** whether the environment was
   installed that way. If it was not, the launcher refuses the test and it
   returns `skipped_env` with that reason; the remedy is then an editable
   install on the GPU side, or vendoring the five files with the MIT notice.
4. **A harness that starts the ranks.** Differs from the list, which was
   written before `dist` mode existed: `gpurun` now starts one process per
   rank and calls the harness in each, so the starting is done by a launcher.
   The launcher `tilescale-deepep` starts one process per rank on the host
   with `RANK`, `LOCAL_RANK`, `WORLD_SIZE` and `LOCAL_WORLD_SIZE`, as the
   port's test spawns one process per GPU
   (`intranode/example_intranode.py:258`), and sets three more variables:
   - `MASTER_ADDR=127.0.0.1` and `MASTER_PORT`, a port that is free when the
     launch starts. **R:** `init_dist` reads both
     (`tilelang/distributed/utils.py:43-44`).
   - `TILELANG_USE_DISTRIBUTED=1`, the setting of the documented command
     (`deepep.md:139`). **R:** the code generator and the kernel wrapper read
     it (`src/target/codegen_cuda.h:24`, `tilelang/jit/adapter/wrapper.py:220`).
     `TILELANG_USE_NVSHMEM` stays unset.

   The harness then follows the port's test: `init_dist`, `EPBuffer`,
   `get_dispatch_layout`, `dispatch` without a handle, `combine`, with a
   barrier of the process group after each step, and
   `destroy_process_group`. Two things in the source go beyond the list:
   - **R:** `init_dist` reads `WORLD_SIZE` and `RANK` as the number of nodes
     and the rank of this node and multiplies by its second argument
     (`tilelang/distributed/utils.py:45-53`). `gpurun` needs both variables
     set per rank, so the harness removes them for that one call and puts
     them back.
   - **R:** `intranode_dispatch` launches its kernel on a stream of its own
     and returns without waiting for it (`intranode/dispatch.py:856-881`);
     `intranode_combine` waits (`intranode/combine.py:413-414`). The harness
     therefore calls `torch.cuda.synchronize()` before it reads a result and
     before it hands rows to the combine. It changes no kernel and no file of
     the port.

   `gpurun` kills every process below the launcher when `timeout_s` passes
   (1800 s for the two repeats of a test). That is still the only guard
   against a hang. Outputs are lost with a killed rank, so the harness prints
   one line per finished step, and the layout counts, to the log.
5. **Outputs with known shapes.** Met with the nine candidate outputs. The
   four dispatch outputs have room for `ranks * tokens` rows, the largest
   count possible; rows a rank does not receive keep the sentinel, so the
   footprint check counts the rows. `is_token_in_rank` is returned as `uint8`
   because a `bool` tensor has no sentinel.
6. **Inputs from the pinned generator.** Met. Every rank gets the same input
   tensors in `dist` mode, so each input has a leading dimension of size
   `ranks` and rank r takes slice r: `x` is `bf16` `[2, 40, 64]`, `topk_idx`
   is `int64` `[2, 40, 2]` in `-1 .. 7`, `topk_weights` is `fp32`
   `[2, 40, 2]`, with 8 experts. The seed of `topk_idx` (34630) is the first
   for which no token selects an expert twice and each rank has a token for
   no rank, for each rank alone and for both. The self-test checks these
   properties.
7. **Parameters for two ranks.** Met: 2 ranks, 4 experts per rank,
   `hidden = 64`, `num_topk = 2`, the non-cached dispatch. Section 7.3 says
   what the thread election does with two ranks.
8. **An NVLink pair.** Unchanged: the operator makes one pair visible with
   `CUDA_VISIBLE_DEVICES`; rank r runs on visible GPU r.
9. **The original as a second reference.** Not done. `deep_ep` is not asked
   for; the reference is the judge's.

### 7.2 What is compared

The judge computes every output of every rank from the regenerated inputs
with the standard library, following section 3.2, and says `equal` only when
all bytes agree in every repeat. Between dispatch and combine the harness
lets rank d return each received row times a factor `scale[d]`. That is the
expert step of the frozen `moeCombine_of_moeDispatch` with the weight of a
token on rank d set to `scale[d]`; the gate weights travel beside the rows
and are compared on their own.

| Test | `scale` | Combined row of a token sent to rank 0, rank 1, both, none |
| --- | --- | --- |
| `intranode/identity` | 1, 1 | x, x, 2x, 0 |
| `intranode/rank_weighted` | 1, -0.5 | x, -x/2, x/2, 0 |

`identity` is what the port's own test and DeepEP's do. It cannot tell the
sum of the two ranks' rows from one rank's row taken twice; `rank_weighted`
can, because the two ranks return different rows.

**Order and tolerance.** **R:** the port starts from zero and adds the
returned rows in increasing rank in `float32`, then stores `bf16`
(`intranode/combine.py:306-325`). The comparison has no tolerance. With these
factors every partial sum is x, -x/2, x/2, 2x or 0, each exact in `float32`
and in `bf16`, so neither the order nor a rounding mode can change a bit, and
with two ranks a sum has at most two terms. The judge refuses a reference
value that the output type cannot hold exactly, so a later batch with other
factors cannot pass this claim on silently.

### 7.3 What only a run can settle

1. **The thread election with two ranks.** **R:** the combine sender publishes
   a tail under `T.shuffle_elect(96)` (`intranode/combine.py:217`), which
   elects "exactly one lane" in each group of 96 threads
   (`tilelang/language/builtin.py:438-459`); a block has 768 threads, so
   eight lanes are elected. Warp w of a block serves rank
   `(channel + w) % num_ranks` (`intranode/combine.py:148`), and with two
   ranks a rank owns 384 threads (`112`). **I:** a group of 96 threads is
   three consecutive warps. If the elected lane sits in the same warp of
   every group, four elected lanes serve each rank, and each stores the same
   value after the barrier of its rank (`216`). The documentation does not
   say which lane of a group is elected. If the lanes were spread otherwise,
   a rank could get none, its tail would never be published and the receiver
   would wait until the timeout. **U:** which of these happens, and whether
   four release stores of one value to one location are harmless. The request
   does not work around this. Its inputs give at most 4 rows per channel and
   rank, below the 10 rows of one round (`deepep_utils.py:76`), so a tail is
   published in one round and takes one value. The non-cached dispatch elects
   one lane per warp (`intranode/dispatch.py:311`, `411`, `510`) and does not
   use the constant.
2. **The build for two ranks.** The kernels are compiled for `num_ranks = 2`
   (384 threads and 12 warps per rank, `intranode/dispatch.py:259-264`,
   `intranode/combine.py:111-112`). The source has tuning entries for 2 ranks
   (`deepep_utils.py:46`, `76`); its only reported run uses 8.
3. **The layout kernel without a barrier** (section 4.2): the counts per rank
   and per expert are summed over threads with no barrier after the
   per-thread counting. The dispatch takes its sizes from these counts. Both
   are outputs of the request.
4. **Loads with `nc=True` from buffers a peer writes** (section 4.2). A stale
   value would show in `recv_src_idx`, `combined_x` or
   `combined_topk_weights`.
5. **Paths the port's own test does not reach.** **I:** its inputs come from
   a top-k of 8 among 32 experts for 4096 tokens (`deepep_utils.py:148-152`,
   `intranode/example_intranode.py:244-248`), so no selection is `-1`, no
   token goes to no rank, and every channel has tokens for every rank. The
   request has all three cases; **R:** the kernels encode a count of zero
   (`intranode/dispatch.py:309-316`).
6. **The editable install** (need 3).
7. **The host side with two ranks**: the process group over TCP on a fresh
   port for each launch; the 1 GiB symmetric heap of each rank mapped into
   the other through inter-process handles; the C++ helper that
   `deepep_utils.py:319` compiles at import through
   `torch.utils.cpp_extension`, in both ranks at the same time;
   the mapped counters of `buffer.py:152-153`. `gpurun` moves the inputs to
   the GPU before the harness calls `init_dist`; in the port's test
   `init_dist` comes first.
8. **Time.** **R:** `intranode/combine.py:13` disables the kernel cache at
   import, so every launch compiles the five kernels in each rank. Whether
   two launches fit in 1800 s is not known.
9. **The harness's own arithmetic.** `recv_x * scale[d]` is computed by
   PyTorch in `bf16`. **I:** the products by 1 and by -0.5 are exact.
10. **The row order** of section 3.2, which is an inference from the source.

Not covered by this request: the cached dispatch, more than two ranks, a
receive buffer that fills (256 slots per channel and rank against at most 4
rows), the FP8 payload and a comparison with `deep_ep`.

## 8. The frozen MoE specification against version 1 normal mode

The declarations are in `VeriTile/Staging/Dist/SpecLibSpec.lean` (lines
217-290): `MoERouting`, `MoERows`, `MoERouting.Assigns`, `MoEDispatch`,
`MoEStoreBack`, `moeCombine`, `moeCombine_of_moeDispatch`. The file header
(lines 27-30) names the EPv2 port as their source. This ticket does not edit
the file.

**What agrees (I, from section 3.2).**

- `MoERouting.dest s t` is the set of ranks `r` with `is_token_in_rank[t, r]`
  on rank `s`; it may be empty and may contain `s` itself.
- A token has exactly one row on each destination and none elsewhere
  (`MoERouting.Assigns`), also when it selects several experts there (decision
  11 of the header). Version 1 has no expanded layout.
- A row that holds a token holds the token's values (`MoEDispatch`).
- The combine applies no weight and adds what the destinations returned
  (`moeCombine`, decision 13). With no bias, the order of version 1 is the
  left fold from zero over the destinations in increasing rank, which is
  `ReduceOrder.ofList` of that list.
- With rows returned unchanged the combine gives the token times the number
  of destinations, which is `moeCombine_of_moeDispatch` with every weight 1
  and matches the repository's test.

**Mismatches.**

1. **Source.** The header says the declarations follow the EPv2 port at
   `c0f80cc5` and its `do_expand=False` layout. For the version 1 proof the
   source is section 6, and `do_expand` does not exist there.
2. **The row assignment is not schedule-dependent.** Decision 12 and the
   docstrings of `MoERows` and `MoEDispatch` say that the row of a token
   "depends on the schedule" and that a dispatch guarantees the relation "for
   some `rows`, not for a given one". In version 1 and in its port the row is
   a function of the routing table: destination rows are ordered by source
   rank, then token index, they are rows `0` to `N - 1` with no gap, and the
   dispatch returns the assignment (`recv_src_idx` and `rank_prefix_matrix`).
   The specification is weaker than what holds, so nothing proved from it
   becomes false, but a version 1 theorem can and should state the assignment,
   and the text "which rows of a buffer are used, and their grouping by source
   rank, are not specified" does not describe version 1.
3. **Top-k data delivered next to a row.** Decision 13 leaves "the gate
   weights that a dispatch delivers next to a row" unspecified. Version 1
   delivers, per row, the expert indices local to the destination (`-1` for
   experts elsewhere) and the weights (`0` where the index is `-1`), and the
   combine returns per token the sum over the destinations of the weight
   vectors it was given. The specification has no statement for either.
   `MoERouting.weight s t d`, "the sum of the gate weights of the experts on
   rank `d` that the token selects", is not a quantity the kernels compute; it
   equals the sum of the delivered weight vector of the row, a fact that the
   specification does not state.
4. **Bias.** Version 1 combine adds up to two bias tensors before the returned
   rows, so the order is `((bias_0 + bias_1) + r_1) + r_2 ...`. The
   specification has no bias (decision 13 says so). The port has none either,
   so the mismatch concerns the original only.
5. **Token counts per rank.** `tokens : RankTensor numRanks (Fin numTokens × ι)`
   gives every rank the same `numTokens`. In version 1 the count
   is per rank. Padding with tokens whose destination set is empty encodes
   this, but the encoding has to be stated.
6. **The intermediate state of the combine.** `MoEStoreBack` relates the
   results to a tensor `back s d (t, j)` and `moeCombine` sums it, as two
   steps. Version 1 has one kernel in which senders and receivers run
   concurrently; a returned row sits in a ring-buffer slot that is reused
   modulo the buffer capacity once the head has passed it. `back` is never a
   memory state that can be observed after the kernel, so a proof has to
   introduce it as a logical value, and only the composition
   (`moeCombine_of_moeDispatch`) can be compared with a run.

**Outside the specification by its own design, recorded for the proof.** The
specification is over real numbers. Version 1 converts each returned BF16
element to `float`, accumulates in `float` and converts the sum to BF16; the
dispatch copies bits. The FP8 payload with scales, the per-expert counts
rounded to `expert_alignment`, `num_worst_tokens`, the cached dispatch and the
capacity of the ring buffers have no counterpart in the specification.

## 9. Unchecked

- Nothing was compiled or run. Whether the port builds and runs on H800, with
  2 ranks in particular, is not known. The statement about `T.shuffle_elect(96)`
  with 2 ranks is an inference from the thread arithmetic.
- The lowering of the port's kernels was not inspected: whether a barrier is
  inserted in the layout kernel, what `T.put_warp` and `T.ld(nc=True)` emit,
  and whether any emitted instruction is specific to a GPU generation.
- `T.thread_return` (`intranode/combine.py:45`) is not defined in a Python
  file under `tilelang/` at either commit; it presumably comes from the TVM
  script import, and the TVM submodule was not fetched.
- The equivalence of the port with the original was read kernel by kernel, not
  proved and not line by line; the differences listed in section 4.2 are the
  ones noticed, and the list may be incomplete. The port's own claim of
  bit-exact agreement was not reproduced.
- `csrc/deep_ep.cpp` of DeepEP was read only for the inter-process handles and
  the mapped counters; the host-side argument checks were not read.
- The bodies of `ep_a2a.py`, `ep_all2all_fused.py`, the `low_latency_*` files
  and tutorial 04 beyond the kernel named in section 3.3 were not read; their
  rows in section 5 are the scanner's output.
- The version string that an environment built from `9ba8a605` would report.
- The commit dates in section 2 are the dates git reports; the DeepEP history
  was not checked against release notes.
