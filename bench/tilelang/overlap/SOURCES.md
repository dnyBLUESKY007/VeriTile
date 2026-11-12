# TileScale overlap and Triton-distributed point-to-point sources

Inventory for [K-S.1, issue #290](https://github.com/dnyBLUESKY007/VeriTile/issues/290),
using the pins in [map #1](https://github.com/dnyBLUESKY007/VeriTile/issues/1).
All sources were read on **2026-11-12**, in scratch checkouts outside VeriTile;
none was imported, executed or copied into this repository. This file inventories
source syntax, not documented memory semantics or verified kernel correctness.

## Pins and licenses

| Key | Repository | Full commit | License note |
| --- | --- | --- | --- |
| T | [TileScale](https://github.com/tile-ai/tilescale/tree/71e71457d40ed6e6a2dad297936bfdbff21d104b) | `71e71457d40ed6e6a2dad297936bfdbff21d104b` | [LICENSE](https://github.com/tile-ai/tilescale/blob/71e71457d40ed6e6a2dad297936bfdbff21d104b/LICENSE) names MIT, copyright Tile-AI, and requires preservation of the copyright and permission notice. It also states additional Microsoft collaboration terms for 2024-12-01 through 2025-03-14; those terms are not detailed here. None of the seven inventoried Python files has its own license header. |
| D | [Triton-distributed](https://github.com/ByteDance-Seed/Triton-distributed/tree/7908e4ea9010bb2238f9f05b28c904fd4b71c60a) | `7908e4ea9010bb2238f9f05b28c904fd4b71c60a` | [LICENSE](https://github.com/ByteDance-Seed/Triton-distributed/blob/7908e4ea9010bb2238f9f05b28c904fd4b71c60a/LICENSE) contains MIT terms, copyright 2025 ByteDance Ltd. and/or its affiliates, including notice preservation; the same terms head `p2p.py`. |

Each file below inherits the full pin and license note of its key. Paths are
relative to that external repository, not paths of vendored files. The two
TileScale directories contain exactly these seven files at T; both host-only
test files are included so the directory inventory is complete.

| ID | Source file at the pin | Pin / license | Census kernels / reachable helpers |
| --- | --- | --- | --- |
| T1 | [examples/distributed/allgather_gemm/example_allgather_gemm_overlapped.py](https://github.com/tile-ai/tilescale/blob/71e71457d40ed6e6a2dad297936bfdbff21d104b/examples/distributed/allgather_gemm/example_allgather_gemm_overlapped.py) | T / MIT with dated collaboration note above | 3 / 0 |
| T2 | [examples/distributed/allgather_gemm/example_allgather_gemm_specialized.py](https://github.com/tile-ai/tilescale/blob/71e71457d40ed6e6a2dad297936bfdbff21d104b/examples/distributed/allgather_gemm/example_allgather_gemm_specialized.py) | T / same | 2 / 3 |
| T3 | [examples/distributed/allgather_gemm/test_example_allgather_gemm.py](https://github.com/tile-ai/tilescale/blob/71e71457d40ed6e6a2dad297936bfdbff21d104b/examples/distributed/allgather_gemm/test_example_allgather_gemm.py) | T / same | 0 / 0 (host tests) |
| T4 | [examples/distributed/gemm_reducescatter/example_gemm_rs_overlapped.py](https://github.com/tile-ai/tilescale/blob/71e71457d40ed6e6a2dad297936bfdbff21d104b/examples/distributed/gemm_reducescatter/example_gemm_rs_overlapped.py) | T / same | 1 / 1 |
| T5 | [examples/distributed/gemm_reducescatter/example_gemm_rs_specialized.py](https://github.com/tile-ai/tilescale/blob/71e71457d40ed6e6a2dad297936bfdbff21d104b/examples/distributed/gemm_reducescatter/example_gemm_rs_specialized.py) | T / same | 1 / 2 |
| T6 | [examples/distributed/gemm_reducescatter/reduce_scatter_utils.py](https://github.com/tile-ai/tilescale/blob/71e71457d40ed6e6a2dad297936bfdbff21d104b/examples/distributed/gemm_reducescatter/reduce_scatter_utils.py) | T / same | 1 / 0 |
| T7 | [examples/distributed/gemm_reducescatter/test_example_gemm_rs.py](https://github.com/tile-ai/tilescale/blob/71e71457d40ed6e6a2dad297936bfdbff21d104b/examples/distributed/gemm_reducescatter/test_example_gemm_rs.py) | T / same | 0 / 0 (host tests) |
| D1 | [python/triton_dist/kernels/nvidia/p2p.py](https://github.com/ByteDance-Seed/Triton-distributed/blob/7908e4ea9010bb2238f9f05b28c904fd4b71c60a/python/triton_dist/kernels/nvidia/p2p.py) | D / MIT terms, repeated in file header | 5 / 0 |

## Source observations and limits

These are observations of the pinned files above, not semantic guarantees.

- **T1:** `gemm_kernel.main` and `main_persistent` wait on remote-shard signals
  before GEMM; the separate reset kernel writes zero. Outside the kernel census,
  `cp_engine_producer_all_gather_full_mesh_pull` calls tensor `copy_` and
  `cuda.cuStreamWriteValue32`, and `ag_gemm_op` coordinates CUDA streams. A
  kernel-only construct list does not cover this host producer protocol.
- **T2:** the main kernel separates communication and computation with
  `T.sm_specialize_scope`; communication calls `T.tma_copy` and
  `T.multimem_signal_add`, while computation calls `T.wait_ge`. Reset uses
  `T.sync_blocks`, `T.fence_sys` and `T.barrier_blocks`. Host allocation uses
  `_allocate_mcast_tensor`. These names alone establish no multicast ordering
  guarantee.
- **T3:** no kernel definitions. Host constructs are `distributed_test`,
  `argparse.Namespace`, the two imported example `main` calls and
  `tilelang.testing.main`. Both tests request four processes; the specialized
  test requests multicast and sets `use_tma_store=False`.
- **T4:** GEMM publication calls `T.atom_add` and `T.st` with
  `scope="gpu", sem="release"`. The host calls the T6 utility on a separate
  stream. The `persistent` argument exists but this file defines only one
  GEMM kernel body.
- **T5:** the fused epilogue selects between
  `T.atomic_add(..., use_tma=True, tma_wait_complete=True, dst_pe=dst_rank)`
  and `T.atomic_addx2(..., dst_pe=dst_rank)`. Its host `main` asserts exactly
  eight local ranks; the source therefore cannot be assumed runnable on the
  map's two-H800 node unchanged. This is a source-level constraint, not a GPU
  test result.
- **T6:** the census finds a local ring-reduction kernel. Outside that census,
  `_wait_eq_cuda` uses `cuStreamWaitValue32/64` and `intra_node_scatter` uses
  tensor `copy_`. `ring_reduce` checks bulk-copy support and otherwise raises;
  `ring_reduce_tma` rejects `num_sms != -1`; `reduce_scatter_for_each_node`
  raises `NotImplementedError` on its inter-node path. The host also asserts
  full-mesh NVLink. Its multi-node names are not evidence of an implemented
  inter-node transfer.
- **T7:** no kernel definitions. Host constructs are `distributed_test`,
  `argparse.Namespace`, the two imported example `main` calls and
  `tilelang.testing.main`. The overlapped test requests four processes; the
  specialized test requests eight and sets `tma_epilogue=True`.
- **D1:** there are five decorated entrypoints, not just the copy entrypoint:
  signal set, signal wait, get, put-with-signal and remote-to-local copy. The
  last performs a `getmem_block` into the local symmetric buffer before masked
  loads and stores to `dst_ptr`. Constants such as `NVSHMEM_SIGNAL_SET`,
  `NVSHMEM_CMP_EQ`, `tl.constexpr` and `tl.int8` are attribute uses, not calls,
  and therefore do not appear in the census call lists.

**Inference:** a later whole-operation model must account for the host stream
protocols in T1/T4/T6 as well as the device kernels, and for both epilogue branches
in T5. No ordering, atomicity, progress or completion guarantee is inferred from
implementation names. Front-end documentation and the assumption registry must
settle those questions in the downstream tickets.

**Not checked:** GPU execution, performance, numerical agreement, Lean proofs,
compiler lowering, imported helper implementations, and legal interpretation of
the dated collaboration clause. Nothing at or below PTX is cited or modelled.

## Census method and reproduction

The tables below are a projection of the actual CLI output of
[`scripts/census/scan.py`](../../../scripts/census/scan.py) at VeriTile commit
`de73640aa55bba1826f5c3c3ad0b4bd44491ce96`. It parses source with `ast` without
importing or running it. The aggregate is **13 kernels and 6 reachable helpers**.
T3 and T7 each yield no records, as expected for host-only tests.

Call notation is `name(positional-count; keyword-names) x occurrence-count`;
`-` means no keywords. This preserves every distinct call signature reported by
the scanner. `T`, `tl`, `tdl`, `shmem` and `extra` abbreviate respectively
`tilelang.language`, `triton.language`, `triton_dist.language`,
`triton_dist.language.extra.libshmem_device` and
`triton_dist.language.extra.language_extra`. Other names are kept as emitted.
Statement counts are Python AST classes, not dynamic execution counts. All
same-file helpers reached by direct calls have their own rows; imported helpers,
attribute-only uses, operators, host launchers and outer factory configuration
are outside the scanner's coverage. Counts do not establish front-end support.

To reproduce from the VeriTile root, set `TILESCALE_SRC` and `TRITON_DIST_SRC`
to scratch checkouts at T and D above, with unchanged source files. Verify their
heads and compare the files against the pinned blobs before scanning:

```sh
VERITILE_ROOT="$PWD"
git -C "$TILESCALE_SRC" rev-parse HEAD
git -C "$TILESCALE_SRC" diff --exit-code 71e71457d40ed6e6a2dad297936bfdbff21d104b -- LICENSE examples/distributed/allgather_gemm examples/distributed/gemm_reducescatter
git -C "$TRITON_DIST_SRC" rev-parse HEAD
git -C "$TRITON_DIST_SRC" diff --exit-code 7908e4ea9010bb2238f9f05b28c904fd4b71c60a -- LICENSE python/triton_dist/kernels/nvidia/p2p.py
(
  cd "$TILESCALE_SRC"
  python3 "$VERITILE_ROOT/scripts/census/scan.py" examples/distributed/allgather_gemm/*.py examples/distributed/gemm_reducescatter/*.py
)
(
  cd "$TRITON_DIST_SRC"
  python3 "$VERITILE_ROOT/scripts/census/scan.py" python/triton_dist/kernels/nvidia/p2p.py
)
python3 -m unittest discover -s scripts/census -p 'test_*.py'
python3 scripts/check_docs.py
```

Both scanner invocations returned exit **0**; the census suite passed **16 tests**
(exit **0**). The documentation check passed before the addition (106 tracked
Markdown documents, 13 synchronized pages); its post-addition result is recorded
in the pull request. The scratch files and licenses were also compared byte for
byte with their pinned Git blobs: all ten matched. Sources, pins, licenses and
scanner results in this record were accessed on 2026-11-12.

### T1: `example_allgather_gemm_overlapped.py`

| Function (definition line) | Kind | Statement counts | Calls |
| --- | --- | --- | --- |
| `reset_signal_kernel._reset_signal_kernel` (L37) | kernel | Assign=2, If=1, With=1 | `T.Kernel(1; threads) x 1`<br>`T.Tensor(2; -) x 1`<br>`T.get_thread_binding(1; -) x 1` |
| `gemm_kernel.main` (L71) | kernel | Assign=18, Expr=8, For=1, If=1, With=1 | `T.Kernel(1; threads) x 1`<br>`T.Pipelined(1; num_stages) x 1`<br>`T.Tensor(2; -) x 4`<br>`T.alloc_fragment(2; -) x 1`<br>`T.alloc_shared(2; -) x 3`<br>`T.ceildiv(2; -) x 6`<br>`T.clear(1; -) x 1`<br>`T.copy(2; -) x 4`<br>`T.gemm(3; -) x 1`<br>`T.get_rank(0; -) x 1`<br>`T.min(2; -) x 1`<br>`T.sync_threads(0; -) x 1`<br>`T.wait_eq(2; -) x 1` |
| `gemm_kernel.main_persistent` (L113) | kernel | Assign=19, Expr=8, For=2, If=2, With=1 | `T.Kernel(1; threads) x 1`<br>`T.Pipelined(1; num_stages) x 1`<br>`T.Tensor(2; -) x 4`<br>`T.alloc_fragment(2; -) x 1`<br>`T.alloc_shared(2; -) x 3`<br>`T.ceildiv(2; -) x 4`<br>`T.clear(1; -) x 1`<br>`T.copy(2; -) x 4`<br>`T.gemm(3; -) x 1`<br>`T.get_rank(0; -) x 1`<br>`T.min(2; -) x 1`<br>`T.serial(1; -) x 1`<br>`T.sync_threads(0; -) x 1`<br>`T.wait_eq(2; -) x 1` |

### T2: `example_allgather_gemm_specialized.py`

| Function (definition line) | Kind | Statement counts | Calls |
| --- | --- | --- | --- |
| `ag_gemm_sm_specialized_kernel.tile_coords` (L59) | helper | Assign=18, Expr=1, Return=1 | `T.if_then_else(3; -) x 5`<br>`T.max(2; -) x 1` |
| `ag_gemm_sm_specialized_kernel.materialized_tile_coords` (L97) | helper | Assign=7, Return=1 | `ag_gemm_sm_specialized_kernel.tile_coords(2; -) x 1`<br>`T.alloc_var(1; -) x 3` |
| `ag_gemm_sm_specialized_kernel.materialized_global_tile_coords` (L108) | helper | Assign=5, Return=1 | `ag_gemm_sm_specialized_kernel.tile_coords(2; -) x 1`<br>`T.alloc_var(1; -) x 2` |
| `ag_gemm_sm_specialized_kernel.main` (L117) | kernel | Assign=26, Expr=18, For=5, If=10, With=3 | `ag_gemm_sm_specialized_kernel.materialized_global_tile_coords(2; -) x 1`<br>`ag_gemm_sm_specialized_kernel.materialized_tile_coords(2; -) x 1`<br>`ag_gemm_sm_specialized_kernel.tile_coords(2; -) x 1`<br>`T.Kernel(1; threads) x 1`<br>`T.Parallel(2; -) x 1`<br>`T.Pipelined(1; num_stages) x 1`<br>`T.Tensor(2; -) x 8`<br>`T.alloc_barrier(1; -) x 1`<br>`T.alloc_fragment(2; -) x 2`<br>`T.alloc_shared(2; -) x 4`<br>`T.barrier_arrive(1; -) x 1`<br>`T.barrier_blocks(1; -) x 1`<br>`T.ceildiv(2; -) x 2`<br>`T.clear(1; -) x 1`<br>`T.copy(2; -) x 6`<br>`T.gemm(3; -) x 1`<br>`T.get_lane_idx(0; -) x 1`<br>`T.get_rank(0; -) x 1`<br>`T.get_warp_idx_sync(0; -) x 1`<br>`T.mbarrier_wait_parity(2; -) x 1`<br>`T.min(2; -) x 1`<br>`T.multimem_signal_add(2; -) x 1`<br>`T.serial(1; -) x 2`<br>`T.serial(3; -) x 1`<br>`T.sm_specialize_scope(0; auto_ws) x 2`<br>`T.sync_threads(2; -) x 2`<br>`T.tma_copy(2; barrier, leader_thread_extent) x 1`<br>`T.tma_copy(2; leader_thread_extent) x 1`<br>`T.tma_store_wait(2; -) x 1`<br>`T.wait_ge(2; -) x 1` |
| `reset_ag_gemm_specialized_state_kernel.main` (L259) | kernel | Assign=4, Expr=3, For=1, If=4, With=1 | `T.Kernel(1; threads) x 1`<br>`T.Tensor(2; -) x 2`<br>`T.barrier_blocks(1; -) x 1`<br>`T.ceildiv(2; -) x 1`<br>`T.fence_sys(0; -) x 1`<br>`T.get_thread_binding(1; -) x 1`<br>`T.serial(1; -) x 1`<br>`T.sync_blocks(1; -) x 1` |

### T4: `example_gemm_rs_overlapped.py`

| Function (definition line) | Kind | Statement counts | Calls |
| --- | --- | --- | --- |
| `gemm_kernel.swizzle_2d` (L19) | helper | Assign=6, Return=1 | `T.min(2; -) x 1` |
| `gemm_kernel.main` (L29) | kernel | Assign=21, Expr=7, For=1, If=2, With=1 | `gemm_kernel.swizzle_2d(3; -) x 1`<br>`T.Cast(2; -) x 1`<br>`T.Kernel(1; threads) x 1`<br>`T.Pipelined(1; num_stages) x 1`<br>`T.Tensor(2; -) x 5`<br>`T.alloc_fragment(2; -) x 1`<br>`T.alloc_local(2; -) x 1`<br>`T.alloc_shared(2; -) x 3`<br>`T.atom_add(2; scope, sem) x 1`<br>`T.ceildiv(2; -) x 5`<br>`T.clear(1; -) x 1`<br>`T.copy(2; -) x 4`<br>`T.gemm(3; -) x 1`<br>`T.get_rank(0; -) x 1`<br>`T.get_thread_binding(1; -) x 1`<br>`T.min(2; -) x 1`<br>`T.st(2; scope, sem) x 1` |

### T5: `example_gemm_rs_specialized.py`

| Function (definition line) | Kind | Statement counts | Calls |
| --- | --- | --- | --- |
| `gemm_rs_specialized_kernel.tile_coords` (L47) | helper | Assign=10, Return=1 | `T.if_then_else(3; -) x 2`<br>`T.max(2; -) x 1` |
| `gemm_rs_specialized_kernel.materialized_tile_coords` (L70) | helper | Assign=5, Return=1 | `gemm_rs_specialized_kernel.tile_coords(2; -) x 1`<br>`T.alloc_var(1; -) x 2` |
| `gemm_rs_specialized_kernel.main` (L79) | kernel | Assign=15, Expr=17, For=4, If=3, With=1 | `gemm_rs_specialized_kernel.materialized_tile_coords(2; -) x 1`<br>`T.Kernel(1; threads) x 1`<br>`T.Pipelined(1; num_stages) x 1`<br>`T.Tensor(2; -) x 3`<br>`T.alloc_fragment(2; -) x 2`<br>`T.alloc_shared(2; -) x 4`<br>`T.atomic_add(2; dst_pe, tma_wait_complete, use_tma) x 2`<br>`T.atomic_addx2(2; dst_pe) x 2`<br>`T.clear(1; -) x 2`<br>`T.copy(2; -) x 5`<br>`T.gemm(3; -) x 2`<br>`T.get_rank(0; -) x 1`<br>`T.get_thread_binding(0; -) x 2`<br>`T.serial(1; -) x 1`<br>`T.serial(3; -) x 2`<br>`T.sync_threads(2; -) x 4` |

### T6: `reduce_scatter_utils.py`

| Function (definition line) | Kind | Statement counts | Calls |
| --- | --- | --- | --- |
| `kernel_ring_reduce_tma._kernel_ring_reduce_tma` (L177) | kernel | Assign=6, AugAssign=1, Expr=5, For=2, With=1 | `T.Kernel(2; threads) x 1`<br>`T.Parallel(2; -) x 1`<br>`T.Pipelined(1; num_stages) x 1`<br>`T.Tensor(2; -) x 2`<br>`T.alloc_fragment(2; -) x 2`<br>`T.alloc_shared(2; -) x 2`<br>`T.ceildiv(2; -) x 2`<br>`T.copy(2; -) x 5` |

### D1: `p2p.py`

| Function (definition line) | Kind | Statement counts | Calls |
| --- | --- | --- | --- |
| `p2p_set_signal` (L33) | kernel | Assign=2, Expr=2, If=1 | `tl.program_id(1; -) x 1`<br>`extra.__syncthreads(0; -) x 1`<br>`extra.tid(1; -) x 1`<br>`shmem.signal_op(4; -) x 1` |
| `p2p_wait_signal` (L47) | kernel | Assign=3, Expr=2, If=1 | `tl.program_id(1; -) x 1`<br>`extra.__syncthreads(0; -) x 1`<br>`extra.tid(1; -) x 1`<br>`shmem.signal_wait_until(3; -) x 1`<br>`tdl.symm_at(2; -) x 1` |
| `p2p_copy_kernel` (L66) | kernel | Assign=7, Expr=1 | `.to(1; -) x 1`<br>`min(2; -) x 1`<br>`tl.cdiv(2; -) x 1`<br>`tl.num_programs(1; -) x 1`<br>`tl.pointer_type(1; -) x 1`<br>`tl.program_id(1; -) x 1`<br>`shmem.getmem_block(4; -) x 1` |
| `p2p_put_kernel` (L89) | kernel | Assign=8, Expr=1 | `.to(1; -) x 2`<br>`min(2; -) x 1`<br>`tl.cdiv(2; -) x 1`<br>`tl.num_programs(1; -) x 1`<br>`tl.pointer_type(1; -) x 2`<br>`tl.program_id(1; -) x 1`<br>`shmem.putmem_signal_nbi_block(7; -) x 1` |
| `p2p_copy_remote_to_local_kernel` (L119) | kernel | Assign=10, Expr=2, For=1 | `.to(1; -) x 2`<br>`min(2; -) x 1`<br>`range(2; -) x 1`<br>`tl.arange(2; -) x 1`<br>`tl.cdiv(2; -) x 2`<br>`tl.load(1; mask) x 1`<br>`tl.pointer_type(1; -) x 2`<br>`tl.program_id(1; -) x 1`<br>`tl.store(2; mask) x 1`<br>`shmem.getmem_block(4; -) x 1` |
