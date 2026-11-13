# Documentation sources of the distributed primitives

Status: revision of ticket S2.2 that applies the user's decisions of
2026-11-12 (section 9). The user has not reviewed this revision. Sources were
read on 2026-11-11; NVSHMEM archive passages and the S6.8 additions were
rechecked on 2026-11-13. Nothing here is a proved result; the table fixes what the
semantics of `VeriTile/Dist/` may assume about each primitive.

The table has one row per primitive of Triton-distributed (`dl.*`,
`libshmem_device.*`, `language_extra`) and of TileScale, and one row for each
further name that the decision on the extent adds (section 5.7). A row gives
the pattern, the documented guarantee, the citation and the treatment. The
Lean instance table of ticket S2.6 refers to a row by section and primitive
name.

## 1. Rules

**What counts as a source** (hard constraints of the map):

- Front-end documentation: Triton, TileLang, Triton-distributed, TileScale.
- The CUDA C++ memory model: `cuda::thread_scope_*` and the libcu++ atomics.
- The NVSHMEM API documentation.

Nothing at or below PTX is cited. A guarantee that the documentation does not
state does not hold. A source file of a front end is used for three things
only: that a name exists, with its parameters and their default values; which
documented routine a binding names; and the docstrings it carries.

**Docstrings** (decision of 2026-11-12). A docstring in the pinned source
counts as front-end documentation of lower rank. A row may rest on one. Such a
row cites it as "docstring at the pinned commit" and gets only what the
docstring states. Where a docstring and a documentation page disagree, the
page governs. A docstring that explains a primitive by an instruction at or
below PTX gives the row nothing.

**Pins.** Triton-distributed `7908e4e` (full hash
`7908e4ea9010bb2238f9f05b28c904fd4b71c60a`); TileScale `71e7145` (full hash
`71e71457d40ed6e6a2dad297936bfdbff21d104b`); Triton `v3.5.1` (commit
`0add68262ab0a2e33b84524346cb27cbb2787356`). Triton-distributed at its pin
bundles Triton at another commit, `f797708`; the one Triton passage that is
cited here has the same text at both (E-TR1).

**NVSHMEM version of record** (decision of 2026-11-12). The version of record
is the one on the GPU node: 3.6.5, in the `triton-dist` environment. The
`triton` environment has 3.3.20, only as a dependency of PyTorch; it is not
the version of record. Both numbers are taken from the decision; this revision
did not check them again. The first reading used `latest` (3.8.0); S6.8 reads
the frozen 3.6.5 archive and compares it with `latest` (sections 3 and 4.2).
Rule: a proof may rely only on guarantees that
already hold in 3.6.5. An item that the release notes date after 3.6.5 is
flagged in its row as "not relied on"; an item absent from the frozen archive
is likewise not relied on. Decision D7 lists the excluded items.

**Patterns.**

| Pattern | Meaning |
| --- | --- |
| a | A memory access, an atomic or a fence, with a scope and an ordering semantics |
| b | An event on a monotone counter, or a wait on one |
| c | An asynchronous transfer: an initiation event plus a completion event bound to a counter |
| none | No memory event: a query, an address computation or a pure value |

**Treatments.**

| Treatment | Meaning |
| --- | --- |
| `documented` | The cited documentation states the guarantee that is used |
| `weakest (Wn)` | The documentation states no guarantee, or omits the aspect named in the row; rule Wn of section 6 applies |
| `named assumption (An)` | The guarantee comes from assumption An of section 7 |
| `out of scope` | The name has no definition for the CUDA back end at the pin. It is not modelled; a kernel that calls it is rejected |

`out of scope` is a fourth label, outside the three treatments of the ticket.
The user accepted it on 2026-11-12. It follows the map, which excludes AMD
back ends.

**Rows with scope and semantic strings.** Such a row has one treatment. It is
`documented` if the documentation of its front end defines strings for it
(decision D5) and states everything else that the row uses. Its note then
names the strings that fall under A1 or get nothing. It is
`named assumption (A1)` if it cannot be used without A1: its return value is
not documented, or the documentation of its front end defines none of its
strings. Its note names the clause of A1.

**Evidence, inference, decision.** Section 4 holds evidence: what a source
says, quoted. In the tables the columns `Evidence` and `Source` are evidence,
`Note` is inference and `Treatment` is the decision. Sections 6 and 7 hold the
decisions that apply to many rows. Section 9 lists the user's decisions of
2026-11-12, and section 10 lists what no decision covers yet.

## 2. Enumeration

The names were read from the pinned source with Python `ast`; no source file
was imported or executed. Each table was then compared with the enumerated
names by a script: every name has exactly one row and no row is extra.

| Module | Rule | Names |
| --- | --- | --- |
| `dl.*` | `__all__` of `python/triton_dist/language/__init__.py` | 25 |
| `dl.tma.*` | `__all__` of `python/triton_dist/language/tma/__init__.py`, without its 6 encoding constants | 5 |
| `libshmem_device.*` | top-level functions of `extra/libshmem_device.py` (85), plus `sync`, which only the CUDA back end defines | 86 |
| `language_extra` | top-level functions of `extra/language_extra.py` (13) united with those of `extra/cuda/language_extra.py` (50); 10 names are in both. A helper whose name starts with one underscore is private and not listed | 53 |
| TileScale, kernel side | top-level functions of `tilelang/language/distributed/{common,comm,sync,multimem}.py` (27), plus the functions of `copy_op.py`, `atomic.py` and `builtin.py` that take `src_pe` or `dst_pe` (4), and `fence_gpu`/`fence_sys` from `builtin.py` (2; S6.8) | 33 |
| TileScale, host side | the signal utilities of `tilelang/distributed/host.py` | 2 |
| Further names of the tutorials in scope | the three names that the decision of 2026-11-12 adds: `set_signal` and `wait_eq` of `triton_dist.kernels.common_ops`, and base Triton `tl.atomic_add` with `sem` and `scope` | 3 |
| Total | | 207 |

The three names of section 5.7 were not enumerated. They come from the
decision and were located in tutorials 06, 07 and 08 by search.

Of the 85 generic `libshmem_device` functions, 60 have a definition in
`extra/cuda/libnvshmem_device.py` and 25 do not.

Not listed, because they are constants or types and not operations: the
comparison, signal, atomic-operation and team constants of
`libshmem_device.py`; the 6 constants of `dl.tma`; the TileScale enums
`BinaryRelation`, `WaitScope`, `WaitSemantics` and `MultimemReduceOp`.

Not covered, because they lie outside the modules the ticket names:
`extra/cuda/tma_language.py` (40 functions), the host-side functions of
`triton_dist.utils`, the other helpers under `triton_dist/kernels/`, base
Triton other than `tl.atomic_add`, and base TileLang (`T.tma_store_wait`,
`T.sync_threads`). The user decided on 2026-11-12 that they stay out until a
kernel in scope uses them. The host code of tutorials 01 to 08 does call
host-side utilities, for example `nvshmem_barrier_all_on_stream` and
`cudaMemcpyAsync`. These calls are host code, not kernel code, and have no
row.

## 3. Sources

Paths with prefix `TD:` are in Triton-distributed at `7908e4e`; paths with
prefix `TS:` are in TileScale at `71e7145`; paths with prefix `TR:` are in
Triton at `v3.5.1`. Original URLs were accessed on 2026-11-11; the NVSHMEM
archive/latest comparison, CU-MM atomicity condition and pinned S6.8 additions
were read on 2026-11-13.

| ID | Source |
| --- | --- |
| TD-PRIM | `TD:docs/primitives.md` |
| TD-API | `TD:docs/python-api/triton-dist.language.rst` |
| TD-SEM | `TD:docs/python-api/triton-dist.semantics.rst` |
| TD-T01 | `TD:tutorials/01-distributed-notify-wait.py` |
| TD-T07 | `TD:tutorials/07-overlapping-allgather-gemm.py` |
| TD-T08 | `TD:tutorials/08-overlapping-gemm-reduce-scatter.py` |
| TD-INIT | `TD:python/triton_dist/language/__init__.py` (names only) |
| TD-SMEM | `TD:python/triton_dist/language/smem_ops.py` (docstrings) |
| TD-TMA | `TD:python/triton_dist/language/tma/pipeline.py`, `TD:python/triton_dist/language/tma/tensormap.py` (docstrings) |
| TD-SH | `TD:python/triton_dist/language/extra/libshmem_device.py` (names; seven docstrings) |
| TD-SHC | `TD:python/triton_dist/language/extra/cuda/libnvshmem_device.py` (names and bound routine names only) |
| TD-LEC | `TD:python/triton_dist/language/extra/cuda/language_extra.py` (names; two docstrings) |
| TD-KCO | `TD:python/triton_dist/kernels/common_ops.py`, `TD:python/triton_dist/kernels/nvidia/common_ops.py` (names, bound routines and the atomic-support docstring at lines 155–157) |
| TR-CORE | `TR:python/triton/language/core.py` (docstring of `atomic_add`) |
| TR-API | `TR:docs/python-api/triton.language.rst`, which lists `atomic_add`; its page is generated from the docstring. Rendered for `main`: <https://triton-lang.org/main/python-api/generated/triton.language.atomic_add.html> |
| TS-API | `TS:docs/distributed_api_reference.md` |
| TS-COMM | `TS:tilelang/language/distributed/comm.py` (docstrings) |
| TS-SYNC | `TS:tilelang/language/distributed/sync.py` (docstrings) |
| TS-MM | `TS:tilelang/language/distributed/multimem.py` (docstrings) |
| TS-COPY | `TS:tilelang/language/copy_op.py` (docstrings) |
| TS-BUILTIN | `TS:tilelang/language/builtin.py`, docstrings at lines 1194–1201 |
| TS-ATOM | `TS:tilelang/language/atomic.py` (docstrings) |
| NV-MM | NVSHMEM 3.6.5, Memory Model: <https://archive.docs.nvidia.com/nvshmem/api/3.6.5/gen/mem-model.html> |
| NV-ORD | NVSHMEM 3.6.5, Memory Ordering: <https://archive.docs.nvidia.com/nvshmem/api/3.6.5/gen/api/ordering.html> |
| NV-RMA | NVSHMEM 3.6.5, Remote Memory Access: <https://archive.docs.nvidia.com/nvshmem/api/3.6.5/gen/api/rma.html> |
| NV-SIG | NVSHMEM 3.6.5, Signaling Operations: <https://archive.docs.nvidia.com/nvshmem/api/3.6.5/gen/api/signal.html> |
| NV-SYNC | NVSHMEM 3.6.5, Point-To-Point Synchronization: <https://archive.docs.nvidia.com/nvshmem/api/3.6.5/gen/api/sync.html> |
| NV-COLL | NVSHMEM 3.6.5, Collective Communication: <https://archive.docs.nvidia.com/nvshmem/api/3.6.5/gen/api/collectives.html> |
| NV-SETUP | NVSHMEM 3.6.5, Library Setup, Exit, and Query: <https://archive.docs.nvidia.com/nvshmem/api/3.6.5/gen/api/setup.html> |
| NV-TEAM | NVSHMEM 3.6.5, Team Management: <https://archive.docs.nvidia.com/nvshmem/api/3.6.5/gen/api/teams.html> |
| NV-OVER | NVSHMEM 3.6.5, API overview: <https://archive.docs.nvidia.com/nvshmem/api/3.6.5/api/overview.html> |
| NV-FAQ | NVSHMEM 3.6.5, FAQ: <https://archive.docs.nvidia.com/nvshmem/api/3.6.5/faq.html> |
| NV-USE | NVSHMEM 3.6.5, Using NVSHMEM: <https://archive.docs.nvidia.com/nvshmem/api/3.6.5/using.html> |
| NV-LATEST | Comparison only: the same paths under <https://docs.nvidia.com/nvshmem/api/latest/> (3.8.0); E-NV22–E-NV24 are historical observations, not 3.6.5 guarantees |
| NV-REL | NVSHMEM release notes 3.6.5, 3.7.0, 3.7.1, 3.7.2 and 3.8.0: <https://github.com/NVIDIA/nvshmem/releases>. Used to date items, not as a source of guarantees |
| CU-MM | libcu++, Memory model: <https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory_model.html> |
| CU-AT | libcu++, `cuda::atomic`: <https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/synchronization_primitives/atomic.html> |
| CU-FE | libcu++, `cuda::atomic_thread_fence`: <https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/synchronization_primitives/atomic/atomic_thread_fence.html> |

The frozen 3.6.5 manual is linked by the [official version index](https://docs.nvidia.com/nvshmem/api/versions1.json).
All eleven archive pages in the table returned HTTP 200 and report
`docsearch:version=3.6.5`; the same eleven `latest` paths report 3.8.0.
The earlier claim that only `latest` was available was incorrect. Section 4.2's
comparison checks every E-NV item against 3.6.5; no absence of a release-note
entry is used as proof of unchanged documentation.

The first draft of this document said that the text had changed since the
research ticket #11 read it on 2026-10-20. This revision could not confirm
that. Ticket #11 also read version 3.8.0, and its dossier already cites the
flush section, the counted signals and the `__syncthreads` note on
`nvshmem_fence`. What a proof may use is decided against 3.6.5, not against
that reading (decision D7).

The libcu++ pages are served under `unstable`; the unversioned path returned
404.

## 4. Evidence

### 4.1 Triton-distributed

- **E-TD1** (TD-SEM, "Memory Scope and Ordering"). `"gpu"`: "Synchronization
  visible only within the same GPU". `"sys"`: "Synchronization visible across
  the entire system (multiple GPUs/nodes)". `"acquire"`: "Ensures all previous
  writes by other threads are visible after the wait". `"release"`: "Ensures
  all previous writes by this thread are visible to other threads". No other
  scope or semantic value is defined.
- **E-TD2** (TD-SEM, "Signal-Based Synchronization"). `wait`: "Wait until n
  signals at ptr reach expected values". `notify`: "Send a signal to a specific
  rank". `consume_token`: "Establish data dependency between wait and memory
  access".
- **E-TD3** (TD-API). `wait(barrierPtrs, numBarriers, scope, semantic,
  waitValue=1)`: "Wait for distributed barriers"; scope is `"gpu"` or `"sys"`;
  semantic is `"acquire"`, `"release"`, "etc."; it returns a "Token for
  consume_token". `notify(ptr, rank, signal=1, sig_op="set",
  comm_scope="inter_node")`: "Send a signal to a remote rank"; `sig_op` is
  `"set"` or `"add"`; `comm_scope` is `"intra_node"` or `"inter_node"`.
- **E-TD4** (TD-SEM, "Token-Based Data Dependency"). With `consume_token` the
  load is "Guaranteed to execute after wait".
- **E-TD5** (TD-SEM, "Symmetric Memory Model"). `symm_at(ptr, rank)`: "Maps a
  local pointer to the corresponding address on another rank"; it "Enables
  direct load/store across ranks without explicit send/receive".
- **E-TD6** (TD-API). `rank`: "Get the current rank ID". `num_ranks`: "Get the
  total number of ranks".
- **E-TD7** (TD-PRIM; TD-API, "libshmem_device Module"). "Triton-distributed
  also expose all the NVSHMEM primitives to Python". "NVSHMEM/ROCSHMEM device
  API bindings. The API automatically dispatches to the appropriate backend."
  Each listed function has a one-line description and no ordering statement.
- **E-TD8** (TD-API, "language_extra Module"). `ld`: "Load with memory
  ordering". `st`: "Store with memory ordering". `atomic_add`: "Atomic add
  operation". `atomic_cas`: "Atomic compare-and-swap". `ld_vector`: "Load a
  vector". `st_vector`: "Store a vector". Each takes `scope` ("Memory scope")
  and `semantic` ("Memory semantic"). `__syncthreads`: "Block-level
  synchronization barrier". `tid`: "Get the thread index within a CTA".
  `pack`, `unpack`, `threads_per_warp`, `num_threads` and `num_warps` have
  one-line descriptions. No return value is described for `atomic_add` or
  `atomic_cas`.
- **E-TD9** (TD-API, "SIMT Operations", "Core Operations"). One-line
  descriptions of `simt_exec_region`, `vector`, `make_vector`, `zeros_vector`,
  `extract`, `insert`. `extern_call`: "Call an external library function".
- **E-TD10** (TD-SEM, "Producer-Consumer Model", "Best Practices"). The example
  stores a tile into a peer rank's buffer, calls `dl.notify` for that rank, and
  the consumer calls `dl.wait(..., "gpu", "acquire")` and then loads, with the
  comment "Now safe to compute on the tile". Best practice 5: "Use `"gpu"`
  scope for intra-node, `"sys"` for inter-node". Both contradict E-TD1, where
  `"gpu"` reaches "only within the same GPU".
- **E-TD11** (TD-T01, comments). Before the `notify`: "need a syncthreads to
  make sure all the data has been sent". "`notify` is also single thread
  scope".
- **E-TD12** (TD-SMEM, docstrings). `allocate_smem`: "Allocate shared memory
  with a swizzled layout". `smem_load`: "Load a scalar element from shared
  memory at given indices". `smem_store`: "Store a scalar value to shared
  memory at given indices". The others describe an index, a pointer or the end
  of the allocation.
- **E-TD13** (TD-TMA, docstrings). `TmaPipeline`: an "object owning barriers
  and buffers" with `producer_*` and `consumer_*` methods. `store_and_wait`:
  "TMA 2D store + commit_group + wait_group". `create_tmap_2d`: "Create and
  publish a 2D tensormap". No visibility or ordering statement.
- **E-TD14** (TD-LEC, docstring of the CUDA `atomic_add`). `scope`: "one of
  "gpu", "sys". default to "gpu"". `semantic`: "one of "release", "acquire",
  "relaxed", "acq_rel". default to "relaxed"". "returns: the result of
  atomic_add". It does not say whether the result is the value before or after
  the addition, and it defines no string. TD-LEC has one other docstring, on
  `multimem_ld_reduce_v4`; it names an instruction and states no guarantee.
  The generic `language_extra.py` has no docstring.
- **E-TD15** (TD-SH, docstrings). `set_rocshmem_ctx`: "ROCSHMEM only".
  `barrier_all_vec`: "ACLSHMEM only". `barrier_vec`: "ACLSHMEM Only". `my_pe`,
  `n_pes`, `int_p` and `remote_ptr`: "Both NVSHMEM and ROCSHMEM". They name
  back ends and state no guarantee. TD-SHC has no docstring.
- **E-TD16** (TD-LEC, signatures). The CUDA definitions of `ld`, `st` and
  `atomic_add` have the defaults `gpu` for the scope and `relaxed` for the
  semantic. Those of `ld_vector` and `st_vector` have the empty string for
  both. `atomic_cas` has no default.
- **E-TD17** (TD-KCO; TD-T07). For the CUDA back end
  `triton_dist.kernels.common_ops` binds `set_signal` to `_set_signal_cuda`
  and `wait_eq` to `_wait_eq_cuda` of `kernels/nvidia/common_ops.py`. Both are
  host functions that enqueue a CUDA driver routine on a stream:
  `cuStreamWriteValue32` or `cuStreamWriteValue64`, and `cuStreamWaitValue32`
  or `cuStreamWaitValue64` with the flag for equality. Neither has a
  docstring, and no page under `TD:docs` describes them; they occur only in
  the tutorial listings. Tutorial 07 comments the calls: "Notify the peer that
  the transmission is done" and "Waiting for the internode data ready". The
  kernel of tutorial 07 waits with `dl.wait(..., "gpu", "acquire")` on the
  words that `set_signal` and `putmem_signal_block` write. Tutorial 06 imports
  `wait_eq` too. The definitions take a tensor: `(signal_tensor, signal,
  stream)` and `(signal_tensor, signal, stream, require_i64)`. Tutorials 06
  and 07 pass an address (`.data_ptr()`), and tutorial 07 passes a fourth
  positional argument to `set_signal`. Nothing was executed; whether these
  tutorials run at the pin was not tested.
- **E-TD18** (TD-T08). `val = tl.atomic_add(counter_ptr + counter_id, 1,
  sem="release", scope="gpu")`, then `dl.notify(barrier_ptr + counter_id,
  rank, signal=1, comm_scope="gpu")` if `val` equals the number of tiles of
  the rank minus one. Comment: "it means the current tile is the last one to
  be completed for this rank".

### 4.2 NVSHMEM

The quotations below are from 3.6.5 unless marked `latest only`. The
comparison table identifies the archive page for every item and quotes changed
wording beside the version-of-record evidence.

- **E-NV1** (NV-ORD, `nvshmem_fence`). "All operations on symmetric data
  objects issued to a particular PE prior to the call to `nvshmem_fence` are
  guaranteed to be delivered before any subsequent operations on symmetric data
  objects to the same PE. `nvshmem_fence` guarantees order of delivery, not
  completion." It "can enforce memory store ordering only for the calling
  thread" and "does not have an effect on the ordering between memory accesses
  issued by the target PE". The Notes also say: "nvshmem_quiet guarantees completion of all operations on symmetric data objects which makes the updates visible to all other PEs." This unconditional wording is part of the D2 conflict (section 10 item 4).
- **E-NV2** (NV-ORD, `nvshmem_quiet`, 3.6.5). It "ensures completion of all operations on symmetric data objects issued by the calling PE". "On systems with only NVLink, all operations on symmetric data objects are guaranteed to be complete and visible to all PEs when nvshmem_quiet returns. On systems with both NVLink and InfiniBand, visibility is only guaranteed at the destination PE." "A call to nvshmem_quiet by a thread completes the operations posted prior to calling nvshmem_quiet." The local/non-collective and non-notification wording is absent from 3.6.5; the comparison below quotes the newer wording.
- **E-NV3** (NV-ORD, table; NV-MM, "Ordering of Operations"). The table marks
  memory store, blocking put and blocking AMO under both Fence and Quiet,
  non-blocking put under both, and blocking get under neither. NV-MM says:
  "Non-blocking operations are not ordered by calls to `nvshmem_fence`." The
  two disagree on non-blocking put.
- **E-NV4** (NV-RMA, `nvshmem_put`). "The routines return after the data has
  been copied out of the source array on the local PE. The delivery of data
  words into the data object on the destination PE may occur in any order.
  Furthermore, two successive put routines may deliver data out of order unless
  a call to `nvshmem_fence` is introduced between the two calls."
- **E-NV5** (NV-RMA, `nvshmem_put_nbi`). "The routines return after initiating
  the operation. The operation is considered complete after a subsequent call
  to `nvshmem_quiet`."
- **E-NV6** (NV-RMA, `nvshmem_get`, `nvshmem_get_nbi`). Blocking: "The routines
  return after the data has been delivered to the dest array on the local PE."
  Non-blocking: "At the completion of `nvshmem_quiet`, the data has been
  delivered to the dest array on the local PE."
- **E-NV7** (NV-RMA, `nvshmem_p`). "these routines start the remote transfer
  and may return before the data is delivered to the remote PE. Use
  `nvshmem_quiet` to force completion".
- **E-NV8** (NV-SIG, `nvshmem_put_signal`). "The completion of signal update
  ... on the remote PE indicates the delivery of its corresponding dest data
  words". "Without a memory-ordering operation, there is no implied ordering
  between the signal update of a put-with-signal routine and another data
  transfer."
- **E-NV9** (NV-SIG, `nvshmem_put_signal_nbi`). "The routines return after
  initiating the operation. The operation is considered complete after a
  subsequent call to `nvshmem_quiet`." Two successive ones "may deliver data
  out of order unless a call to `nvshmem_fence` is introduced".
- **E-NV10** (NV-SIG, `nvshmemx_signal_op`). It "atomically updates sig_addr
  with signal using operation sig_op on the specified PE".
- **E-NV11** (NV-SYNC, `nvshmem_signal_wait_until`). It "blocks until the value
  contained in the signal data object, sig_addr, at the calling PE satisfies
  the wait condition". The object "is expected only to be updated as a signal,
  through the signaling operations". "Implementations must ensure that
  `nvshmem_signal_wait_until` does not return before the update of the memory
  indicated by sig_addr is fully complete."
- **E-NV12** (NV-SIG and NV-MM, "Atomicity Guarantees"). Signal operations
  "complete as if performed atomically with respect to" other signal
  operations, signal-fetch and "any point-to-point synchronization routine that
  accesses the signal data object using the same datatype". Undefined
  behaviour: "When NVSHMEM atomic operations and non-NVSHMEM operations (e.g.,
  load and store operations) are used to access the same location
  concurrently."
- **E-NV13** (NV-COLL, `nvshmem_barrier_all`, `nvshmem_barrier`). It "blocks
  the calling PE until all PEs have called `nvshmem_barrier_all`". Before that
  it "ensures completion of all previously issued memory stores and remote
  memory updates". The team form does the same over a team. The team-barrier description further states: "nvshmem_barrier ensures that all previously issued stores and remote memory updates, including AMOs and RMA operations, done by any of the PEs in the active set are complete before returning. On systems with only NVLink, updates are globally visible, whereas on systems with both NVLink and InfiniBand, NVSHMEM only guarantees that updates to the memory of a given PE are visible to that PE."
- **E-NV14** (NV-COLL, `nvshmem_sync`, `nvshmem_sync_all`). It blocks until all
  PEs of the team have called it and "only ensures completion and visibility of
  previously issued memory stores and does not ensure completion of remote
  memory updates issued via NVSHMEM routines".
- **E-NV15** (NV-MM, "Visibility Guarantees"). On systems with NVLink and
  InfiniBand the synchronization operations "only guarantee visibility of
  updates to the local PE's symmetric objects. However, on systems with only
  NVLink, these operations guarantee global visibility".
- **E-NV16** (NV-OVER, NVSHMEM API Extensions for GPU Threads). "These extension APIs are collective calls that must be called by every
  thread in the scope of the API and with exactly the same arguments."
  Otherwise "the call results in erroneous behavior or a deadlock".
- **E-NV17** (NV-SETUP, `nvshmem_ptr`). It "returns an address that may be used
  to directly reference dest on the specified PE ... ordinary loads and stores
  to dest may be performed". It returns a null pointer when the object cannot
  be reached by loads and stores.
- **E-NV18** (NV-SETUP, `nvshmemx_mc_ptr`). It "returns a multicast address
  corresponding to the specified symmetric address and the team", for use with
  multimem operations "exclusively". The hardware condition names "third-generation NVSwitch systems (NVLink4) with Hopper and later GPU architectures". "On unsupported platforms, this API returns a nullptr." Passing this multicast address to an NVSHMEM argument requiring a symmetric address "results in undefined behavior." These excerpts document the pointer requirement, not helper atomicity or reduction order; the access/conflict text is E-NV27.
- **E-NV19** (NV-SETUP, NV-TEAM). `nvshmem_my_pe` returns "an integer between
  0 and npes - 1"; `nvshmem_n_pes` "the number of PEs running in the program".
  `nvshmem_team_my_pe`, `nvshmem_team_n_pes` and `nvshmem_team_translate_pe`
  return the team-relative values, or -1 for an invalid team.
- **E-NV20** (NV-COLL, introduction, `nvshmem_broadcast`, `nvshmem_fcollect`).
  "Upon return from a broadcast routine ... the dest data object is updated.
  The source data object may be safely reused." The same holds for fcollect.
  "Concurrent accesses to symmetric memory by an NVSHMEM collective routine and
  any other means of access—where at least one updates the symmetric
  memory—results in undefined behavior."
- **E-NV21** (NV-FAQ). "For synchronization with flag, the application must use nvshmem_wait_until or nvshmem_test API. A plain while loop or if condition to check flag value is not sufficient. NVSHMEM needs to perform consistency operation to ensure that the data is visible to the GPU after synchronization using flag value."
- **E-NV22** (`latest only`, NV-LATEST ordering page, notes of `nvshmem_fence` and `nvshmem_quiet` on GPU
  threads). "In device code, making an operation visible to the thread that
  calls `nvshmem_fence` means ensuring that the operation has been issued
  before the fence is called". "after a block-scoped Put operation returns and
  all threads in the CTA synchronize with `__syncthreads`, a single thread in
  the CTA may call `nvshmem_fence` before issuing a signal". For `quiet`: "The
  same rule applies to GPU threads. If multiple threads issue NVSHMEM
  operations and then synchronize so that those operations have been issued
  before a single elected thread calls `nvshmem_quiet`, the single
  `nvshmem_quiet` call completes the preceding operations." Absent from the corresponding 3.6.5 archive page; not relied on (D7).
- **E-NV23** (`latest only`, NV-LATEST ordering page, `nvshmemx_flush`, `_warp`, `_block`, `_on_stream`). The
  routines "ensure local completion of outstanding nonblocking Put operations
  issued by the calling PE". "Flush does not guarantee that data is visible at
  the destination PE and does not replace `nvshmem_quiet` or `nvshmem_fence`." Absent from the corresponding 3.6.5 archive page; not relied on (D7).
- **E-NV24** (`latest only`, NV-LATEST signal page, counted signals). "The NVSHMEM counted signaling
  extensions copy a payload and increment a remote counter by the number of
  bytes delivered. They also provide operations to reset, load, and wait on
  the counted signal data object." The routines are
  `nvshmemx_signal_counted_reset`, `_load`, `_wait_until` and
  `nvshmemx_putmem_signal_counted_nbi_block`. Absent from the corresponding 3.6.5 archive page; not relied on (D7).
- **E-NV25** (NV-REL). 3.6.5 was published on 2026-03-24. 3.7.0 (2026-06-11):
  "Added `nvshmemx_flush` APIs to provide source-buffer reusability without
  guaranteeing remote visibility." 3.8.0 (2026-09-22): "Added counted-signal
  APIs (based on counted writes) for block-scoped CFT transfers." 3.8.0, bug
  fixes: "Fixed memory visibility after device-side signal waits." The release
  notes list features and fixes. They do not record changes of the
  documentation text.
- **E-NV26** (NV-REL, "Limitations", the same sentence in the notes of 3.6.5
  and of 3.8.0). "`nvshmem_barrier*`, `nvshmem_quiet`, and `nvshmem_wait_until`
  only ensure ordering and visibility between the source and destination PEs.
  They *do not* ensure global ordering and visibility."
- **E-NV27** (NV-USE, Data Consistency). The access list includes "Direct multimem load reduce and store broadcast operations (for example, using a pointer that was returned by nvshmemx_mc_ptr)". "Two operations that were issued by the same PE or different PEs, and that access the same memory location in parallel, are in conflict when one or more of the operations performs a write. These conflicts result in undefined behavior in the OpenSHMEM memory model. An exception is made when the operations are a combination of AMOs or AMOs and wait/test operations. A second exception is made when the operations are a combination of signal updates and wait/test operations." "Updates to globally accessible objects are unordered. A PE can enforce the ordering of its updates with respect to accesses performed by the target PE using the nvshmem_fence operation. When updates performed by a PE must be ordered or made visible to PEs other than the target PE, use the nvshmem_quiet operation." No direct reduce-store, reduction order or atomic multicast operation is specified by this access list. See [MultimemSources](MultimemSources.md).

No binding of TD-SHC names a flush routine or a counted-signal routine.

#### Comparison against the frozen 3.6.5 manual

“Same” refers to the passages used in the entry, not to entire pages. Every API
URL below returned 200 and version 3.6.5 on 2026-11-13. Release entries E-NV25/26
are explicitly release-note evidence, not version-tagged API pages.

| Evidence | 3.6.5 source | Comparison with the previous `latest` evidence |
| --- | --- | --- |
| E-NV1 | [gen/api/ordering.html](https://archive.docs.nvidia.com/nvshmem/api/3.6.5/gen/api/ordering.html#nvshmem-fence) | Same in 3.6.5. |
| E-NV2 | [gen/api/ordering.html](https://archive.docs.nvidia.com/nvshmem/api/3.6.5/gen/api/ordering.html#nvshmem-quiet) | Different: explicit NVLink-only global visibility; mixed-fabric destination visibility. No local/non-collective or non-notification wording. Thread note uses prior rather than before. 3.6.5 wording: "On systems with only NVLink, all operations on symmetric data objects are guaranteed to be complete and visible to all PEs when nvshmem_quiet returns. On systems with both NVLink and InfiniBand, visibility is only guaranteed at the destination PE."; "A call to nvshmem_quiet by a thread completes the operations posted prior to calling nvshmem_quiet." Latest wording: "is a local, non-collective operation"; "the call does not synchronize with or notify other PEs"; "Visibility is only guaranteed at the destination PE."; "posted before calling" (NV-LATEST). |
| E-NV3 | [ordering](https://archive.docs.nvidia.com/nvshmem/api/3.6.5/gen/api/ordering.html), [memory model](https://archive.docs.nvidia.com/nvshmem/api/3.6.5/gen/mem-model.html#ordering-of-operations) | Same in 3.6.5. |
| E-NV4 | [gen/api/rma.html](https://archive.docs.nvidia.com/nvshmem/api/3.6.5/gen/api/rma.html#nvshmem-put) | Same in 3.6.5. |
| E-NV5 | [gen/api/rma.html](https://archive.docs.nvidia.com/nvshmem/api/3.6.5/gen/api/rma.html#nvshmem-put-nbi) | Same in 3.6.5. |
| E-NV6 | [gen/api/rma.html](https://archive.docs.nvidia.com/nvshmem/api/3.6.5/gen/api/rma.html) | Same in 3.6.5. |
| E-NV7 | [gen/api/rma.html](https://archive.docs.nvidia.com/nvshmem/api/3.6.5/gen/api/rma.html#nvshmem-p) | Same in 3.6.5. |
| E-NV8 | [gen/api/signal.html](https://archive.docs.nvidia.com/nvshmem/api/3.6.5/gen/api/signal.html#nvshmem-put-signal) | Same in 3.6.5. |
| E-NV9 | [gen/api/signal.html](https://archive.docs.nvidia.com/nvshmem/api/3.6.5/gen/api/signal.html#nvshmem-put-signal-nbi) | Same in 3.6.5. |
| E-NV10 | [gen/api/signal.html](https://archive.docs.nvidia.com/nvshmem/api/3.6.5/gen/api/signal.html#nvshmemx-signal-op) | Same in 3.6.5. |
| E-NV11 | [gen/api/sync.html](https://archive.docs.nvidia.com/nvshmem/api/3.6.5/gen/api/sync.html#nvshmem-signal-wait-until) | Same in 3.6.5. |
| E-NV12 | [memory model](https://archive.docs.nvidia.com/nvshmem/api/3.6.5/gen/mem-model.html#atomicity-guarantees), [signals](https://archive.docs.nvidia.com/nvshmem/api/3.6.5/gen/api/signal.html) | Same guarantee; `e.g.` rather than `for example` in NV-MM (signal atomicity passage unchanged). 3.6.5 wording: "When NVSHMEM atomic operations and non-NVSHMEM operations (e.g., load and store operations) are used to access the same location concurrently." Latest wording: "When NVSHMEM atomic operations and non-NVSHMEM operations (for example, load and store operations) are used to access the same location concurrently." (NV-LATEST). |
| E-NV13 | [gen/api/collectives.html](https://archive.docs.nvidia.com/nvshmem/api/3.6.5/gen/api/collectives.html) | Same in 3.6.5. |
| E-NV14 | [gen/api/collectives.html](https://archive.docs.nvidia.com/nvshmem/api/3.6.5/gen/api/collectives.html#nvshmem-sync) | Same guarantee; `via` rather than `through`. 3.6.5 wording: "only ensures completion and visibility of previously issued memory stores and does not ensure completion of remote memory updates issued via NVSHMEM routines". Latest wording: "only ensures completion and visibility of previously issued memory stores and does not ensure completion of remote memory updates issued through NVSHMEM routines" (NV-LATEST). |
| E-NV15 | [gen/mem-model.html](https://archive.docs.nvidia.com/nvshmem/api/3.6.5/gen/mem-model.html#visibility-guarantees) | Same in 3.6.5. |
| E-NV16 | [api/overview.html](https://archive.docs.nvidia.com/nvshmem/api/3.6.5/api/overview.html) | Same guarantee; `These extension APIs` rather than repeating the two suffixes. 3.6.5 wording: "These extension APIs are collective calls that must be called by every thread in the scope of the API and with exactly the same arguments." Latest wording: "The `_block` and `_warp` extension APIs are collective calls that must be called by every thread in the scope of the API and with exactly the same arguments." (NV-LATEST). |
| E-NV17 | [gen/api/setup.html](https://archive.docs.nvidia.com/nvshmem/api/3.6.5/gen/api/setup.html#nvshmem-ptr) | Same in 3.6.5. |
| E-NV18 | [gen/api/setup.html](https://archive.docs.nvidia.com/nvshmem/api/3.6.5/gen/api/setup.html#nvshmemx-mc-ptr) | Same in 3.6.5. |
| E-NV19 | [teams](https://archive.docs.nvidia.com/nvshmem/api/3.6.5/gen/api/teams.html), [queries](https://archive.docs.nvidia.com/nvshmem/api/3.6.5/gen/api/setup.html) | Same in 3.6.5. |
| E-NV20 | [gen/api/collectives.html](https://archive.docs.nvidia.com/nvshmem/api/3.6.5/gen/api/collectives.html) | Same in 3.6.5. |
| E-NV21 | [faq.html](https://archive.docs.nvidia.com/nvshmem/api/3.6.5/faq.html) | Same in 3.6.5. |
| E-NV22 | [gen/api/ordering.html](https://archive.docs.nvidia.com/nvshmem/api/3.6.5/gen/api/ordering.html) | Absent from 3.6.5: both later GPU-thread notes. |
| E-NV23 | [gen/api/ordering.html](https://archive.docs.nvidia.com/nvshmem/api/3.6.5/gen/api/ordering.html) | Absent from 3.6.5: flush APIs. |
| E-NV24 | [gen/api/signal.html](https://archive.docs.nvidia.com/nvshmem/api/3.6.5/gen/api/signal.html) | Absent from 3.6.5: counted signals. |
| E-NV25 | [3.6.5](https://github.com/NVIDIA/nvshmem/releases/tag/v3.6.5-0), [3.7.0](https://github.com/NVIDIA/nvshmem/releases/tag/v3.7.0-0), [3.8.0](https://github.com/NVIDIA/nvshmem/releases/tag/v3.8.0-0) release notes | Release history: later additions/fix belong to 3.7.0/3.8.0, not this manual. |
| E-NV26 | [3.6.5 release notes](https://github.com/NVIDIA/nvshmem/releases/tag/v3.6.5-0) | Same limitation in 3.6.5; manual conflict remains open. |
| E-NV27 | [using.html](https://archive.docs.nvidia.com/nvshmem/api/3.6.5/using.html#data-consistency) | Same in 3.6.5; newly recorded access/conflict passage. |

### 4.3 CUDA C++ memory model

- **E-CU1** (CU-MM, "Scope Relationships"). "Each thread in the system is
  related to each other thread in the system by the system thread scope".
  "Each GPU thread is related to each other GPU thread in the same CUDA device
  and within the same memory synchronization domain by the device thread
  scope". Block scope relates the threads of one thread block.
- **E-CU2** (CU-MM, "Data Races"). A program has a data race if it contains
  "two potentially concurrent conflicting actions, at least one of which is not
  atomic at a scope that includes the thread that performed the other
  operation, and neither happens before the other". "Any such data race results
  in undefined behavior."
- **E-CU3** (CU-MM, fences; CU-FE). A release synchronizes with an acquire only
  if "each operation ... specifies a scope that includes the thread that
  performed each other operation". `cuda::atomic_thread_fence(order, scope)`
  "Establishes memory synchronization ordering of non-atomic and relaxed atomic
  accesses, as instructed by order, for all threads within scope".
- **E-CU4** (CU-AT). `cuda::atomic<T, Scope>` "has the same interface and
  semantics as `cuda::std::atomic`", with a thread scope.

- **E-CU5** (CU-MM, [Atomicity](https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory_model.html#atomicity), paraphrase, accessed 2026-11-13).
  For system-scope atomics on GPU memory accessed only by GPU threads, the
  native-atomic P2P attribute `cudaDevP2PAttrNativeAtomicSupported` must equal 1
  between each accessing GPU and the GPU containing the object; single-GPU
  concurrent access is the alternative. This is a platform condition, not a
  consequence of spelling the scope `sys`. The pinned TD-KCO docstring of
  [`barrier_all_intra_node_atomic_cas_block`, lines 155–157](https://github.com/ByteDance-Seed/Triton-distributed/blob/7908e4ea9010bb2238f9f05b28c904fd4b71c60a/python/triton_dist/kernels/nvidia/common_ops.py#L155-L157)
  independently requires atomic support and excludes unsupported PCIe memory.
  No hardware query was run. Whether D5/A1/A2 need a launch precondition or
  registered assumption is open (section 10 item 6); neither is added here.

### 4.4 TileScale

- **E-TS1** (TS-API, introduction, "Peer Memory Operations"). TileScale adds
  "experimental single-node, multi-GPU CUDA primitives". "Multi-node execution,
  NVSHMEM, and a general-purpose distributed runtime are outside the current
  scope." The peer operations "use CUDA IPC or VMM mappings ... They do not
  call NVSHMEM."
- **E-TS2** (TS-API, "Rank Queries"). `T.get_rank()`: "Current local process
  rank". `T.get_num_ranks()`: "Size of the initialized local process group".
  `kernel.initialize` "must run before kernels that call rank queries, remap
  peer addresses, or create remote TMA descriptors".
- **E-TS3** (TS-API, "Peer Memory Operations"). "A PE value of -1, or an
  omitted src_pe/dst_pe, keeps the local address. A non-negative PE remaps the
  corresponding address by allocator-relative offset. A copy cannot make both
  sides remote in the same operation."
- **E-TS4** (TS-API, same section). Remote TMA covers PEs 0 through 7. Outside
  that range a remote TMA copy "issues no operation, which stalls a load
  waiting on its mbarrier".
- **E-TS5** (TS-API; TS-COMM docstrings). `T.ld(src, value, scope="gpu",
  sem="weak", na, nc, src_pe)`, `T.st(dst, value, scope="gpu", sem="weak", na,
  dst_pe)`. `T.atom_add` "is the distributed uint32 atomic-add intrinsic and
  supports gpu or sys scope with relaxed, acquire, release, or acq_rel
  semantics"; its docstring adds "return the previous value". The values
  `weak` and `volatile` are not defined. `nc=True` "must not be used on memory
  a peer may write during the kernel".
- **E-TS6** (TS-API, "Wait and Barrier Operations"; TS-SYNC docstrings).
  `T.wait_eq` ... `T.wait_lt(value, expected, peer=-1, scope="sys",
  semantics="acquire")`: "Wait until value == expected" and likewise for the
  other relations. "Valid wait scopes are sys and gpu; valid semantics are
  acquire and volatile. A non-negative peer remaps the polled address to that
  peer."
- **E-TS7** (TS-API, same section). The barrier helpers are listed by name.
  "Cross-GPU correctness still depends on using peer-visible storage and the
  appropriate system-scope ordering."
- **E-TS8** (TS-SYNC docstrings). `barrier_blocks`: "Rendezvous all blocks of
  every rank at a system-level barrier. Issues a system-scope release fence
  before arriving, so writes made before the call are ordered ahead of this
  rank's arrival ... the barrier does not currently emit an acquire fence on
  exit". `sync_blocks`: "a counter rendezvous only ... it provides no memory
  ordering in either direction".
- **E-TS9** (TS-API, "Multimem Operations"; TS-MM docstrings). One-line
  purposes: reduce-load a multicast region, broadcast a local region,
  reduce-store, store or add a signal "through a multicast address". For
  `multimem_tma_store`: "Shared-memory producer synchronization and async
  bulk-group completion are managed by the caller", followed by the sequence
  `T.fence_proxy_async()`, `T.sync_threads()`, the store, `T.tma_store_arrive()`
  and `T.tma_store_wait(0, False)`. No statement on atomicity, on visibility at
  other ranks or on the order of a reduction.
- **E-TS10** (TS-COPY docstring). `T.tma_copy` is a "TMA copy with
  user-managed synchronization": a load issues no wait and the user waits on
  the barrier; a store "omits the wait so the user can batch multiple stores
  before calling `T.tma_store_wait()`".
- **E-TS11** (TS-ATOM docstrings). `atomic_add`: "Atomically add `value` into
  `dst`"; `memory_order` is used only on the scalar path; `dst_pe` is the
  "Remote PE for symmetric destination addressing". No scope is stated.
- **E-TS12** (TS-API, "Host Utilities"). `set_signal` "Enqueues
  `cuStreamWriteValue32`". `wait_eq` "Enqueues `cuStreamWaitValue32`", or the
  64-bit form with `require_i64=True`.
- **E-TS13** (TS-API, TS-COMM, TS-SYNC; the strings). TS-API lists the scope
  strings `gpu` and `sys` and the semantic strings `weak`, `volatile`,
  `relaxed`, `acquire`, `release` and `acq_rel`. It defines none of them, and
  no docstring does. For `T.ld` and `T.st` it shows only the defaults `gpu`
  and `weak`; the other semantic strings are listed for `T.atom_add` and for
  the waits. The docstrings of `ld` and `st` explain the two arguments by the
  instruction level and define no string. The scope `cta` appears in no
  document; the source of `ld` and `st` accepts it.

- **E-TS14** (TS-BUILTIN, docstrings at the pinned commit,
  [lines 1194–1201](https://github.com/tile-ai/tilescale/blob/71e71457d40ed6e6a2dad297936bfdbff21d104b/tilelang/language/builtin.py#L1194-L1201)).
  `fence_gpu` documents device scope; `fence_sys` documents system scope and
  cross-GPU visibility within a node. Neither states a memory order. The
  [syntax census](../census/MinimalSyntax.md#tilescale-kernels) records
  `fence_sys` in kernels 5 (specialized allgather reset, `71e7145`) and 21
  (the version-2 DeepEP dispatch candidate, `c0f80cc5`); the latter is not the
  project's version-1 DeepEP target. This is census evidence, not a run.

### 4.5 Base Triton

- **E-TR1** (TR-CORE, docstring; TR-API). `atomic_add(pointer, val,
  mask=None, sem=None, scope=None)`: "Performs an atomic add at the memory
  location specified by pointer." "Return the data stored at pointer before
  the atomic operation." `sem`: "Specifies the memory semantics for the
  operation. Acceptable values are "acquire", "release", "acq_rel" (stands for
  "ACQUIRE_RELEASE"), and "relaxed". If not provided, the function defaults to
  using "acq_rel" semantics." `scope`: "Defines the scope of threads that
  observe the synchronizing effect of the atomic operation. Acceptable values
  are "gpu" (default), "cta" (cooperative thread array, thread block), or
  "sys" (stands for "SYSTEM")." No string is defined beyond these glosses. The
  text is the same at `v3.5.1`, at the commit `f797708` that
  Triton-distributed bundles, and on the rendered page for `main` (compared on
  2026-11-11).

## 5. Tables

Row key: section and primitive name. `undocumented` in the `Evidence` column
means that no source of section 1 describes the name. An evidence ID followed
by "(docstring at the pinned commit)" means that the row rests on a docstring
and gets only what that docstring states.

### 5.1 `dl.*`

`dl` is `triton_dist.language`. Source of the names: TD-INIT.

| Primitive | Pattern | Evidence | Source | Treatment | Note |
| --- | --- | --- | --- | --- | --- |
| `wait` | b | E-TD1, E-TD2, E-TD3 | TD-SEM, TD-API | documented | `gpu`, `sys`, `acquire` and `release` are read by D5. With `relaxed` or `acq_rel`: A1, clause (a). The comparison relation ("reach") is not stated: W6, read by I4 (D6) |
| `consume_token` | none | E-TD2, E-TD4 | TD-SEM, TD-API | documented | Orders the access after the wait in the same thread |
| `rank` | none | E-TD6 | TD-API, TD-PRIM | documented | |
| `num_ranks` | none | E-TD6 | TD-API, TD-PRIM | documented | |
| `symm_at` | none | E-TD5 | TD-SEM, TD-API | documented | An address on the given rank. A load or store through it is a plain Triton access |
| `notify` | b | E-TD2, E-TD3, E-TD11 | TD-SEM, TD-API, TD-T01 | named assumption (A2) | Atomicity, scope and ordering of the signal write are not stated. The user accepted A2 on 2026-11-12 |
| `simt_exec_region` | none | E-TD9 | TD-API | documented | |
| `extract` | none | E-TD9 | TD-API | documented | |
| `insert` | none | E-TD9 | TD-API | documented | |
| `vector` | none | E-TD9 | TD-API | documented | |
| `make_vector` | none | E-TD9 | TD-API | documented | |
| `zeros_vector` | none | E-TD9 | TD-API | documented | |
| `ld_vector` | a | E-TD1, E-TD8, E-TD16 | TD-SEM, TD-API, TD-LEC | documented | One load per element. Strings as for `ld` of 5.4. The CUDA default for both arguments is the empty string: W1 |
| `st_vector` | a | E-TD1, E-TD8, E-TD16 | TD-SEM, TD-API, TD-LEC | documented | One store per element. Strings as for `st` of 5.4. The CUDA default for both arguments is the empty string: W1 |
| `SMEM_GENERIC_POINTER_ADDR_SPACE` | none | undocumented | TD-SMEM | weakest (W3) | A constant |
| `SharedMemoryDesc` | none | E-TD12 (docstring at the pinned commit) | TD-SMEM | documented | A handle type |
| `allocate_smem` | none | E-TD12 (docstring at the pinned commit) | TD-SMEM | documented | Block-local memory |
| `smem_index` | none | E-TD12 (docstring at the pinned commit) | TD-SMEM | documented | |
| `smem_load` | a | E-TD12 (docstring at the pinned commit) | TD-SMEM | documented | A plain load of block-local memory |
| `smem_store` | a | E-TD12 (docstring at the pinned commit) | TD-SMEM | documented | A plain store to block-local memory |
| `smem_get_ptr` | none | E-TD12 (docstring at the pinned commit) | TD-SMEM | documented | |
| `get_smem_shared_address_u32` | none | E-TD12 (docstring at the pinned commit) | TD-SMEM | documented | |
| `smem_dealloc` | none | E-TD12 (docstring at the pinned commit) | TD-SMEM | documented | |
| `tma` | none | E-TD13 (docstring at the pinned commit) | TD-TMA | weakest (W4) | A submodule; its names are in 5.2 |
| `extern_call` | none | E-TD9 | TD-API | weakest (W6) | The effect of the callee is not stated; a direct call is not modelled |

### 5.2 `dl.tma.*`

Source of the names: `TD:python/triton_dist/language/tma/__init__.py`. No
tutorial 01–08 uses this module.

| Primitive | Pattern | Evidence | Source | Treatment | Note |
| --- | --- | --- | --- | --- | --- |
| `create_tmap_2d` | a | E-TD13 (docstring at the pinned commit) | TD-TMA | weakest (W4) | Callees are in `tma_language.py`, not covered here |
| `tmap_update_address` | a | E-TD13 (docstring at the pinned commit) | TD-TMA | weakest (W4) | As above |
| `PipelineState` | none | E-TD13 (docstring at the pinned commit) | TD-TMA | documented | A value: stage index and phase |
| `TmaPipeline` | c | E-TD13 (docstring at the pinned commit) | TD-TMA | weakest (W4) | Transfers within one GPU; callees not covered here |
| `store_and_wait` | c | E-TD13 (docstring at the pinned commit) | TD-TMA | weakest (W4) | As above |

### 5.3 `libshmem_device.*`

Source of the names: TD-SH; of the bound NVSHMEM routine names: TD-SHC. The
front-end statement for every documented row is E-TD7. `—` in the `Bound
routine` column means that the CUDA back end has no definition.

| Primitive | Pattern | Bound routine | Evidence | Source | Treatment | Note |
| --- | --- | --- | --- | --- | --- | --- |
| `set_rocshmem_ctx` | none | — | E-TD15 (docstring at the pinned commit) | TD-SH | out of scope | ROCSHMEM only |
| `my_pe` | none | `nvshmem_my_pe` | E-NV19 | NV-SETUP, TD-API | documented | |
| `n_pes` | none | `nvshmem_n_pes` | E-NV19 | NV-SETUP, TD-API | documented | |
| `team_my_pe` | none | `nvshmem_team_my_pe` | E-NV19 | NV-TEAM, TD-API | documented | |
| `team_n_pes` | none | `nvshmem_team_n_pes` | E-NV19 | NV-TEAM, TD-API | documented | |
| `int_p` | c | `nvshmem_int_p` | E-NV7 | NV-RMA, TD-PRIM | documented | Complete only after `quiet` |
| `remote_ptr` | none | `nvshmem_ptr` | E-NV17, E-NV12 | NV-SETUP, TD-API | documented | An access through the address is a plain load or store |
| `remote_mc_ptr` | none | `nvshmemx_mc_ptr` | E-NV18 | NV-SETUP, TD-API | documented | The address only; the operations on it are rows of 5.4 |
| `barrier_all` | b | `nvshmem_barrier_all` | E-NV13, E-NV15 | NV-COLL, NV-MM, TD-API | documented | |
| `barrier_all_vec` | b | — | E-TD15 (docstring at the pinned commit) | TD-SH | out of scope | ACLSHMEM only |
| `barrier_all_block` | b | `nvshmemx_barrier_all_block` | E-NV13, E-NV15, E-NV16 | NV-COLL, NV-OVER, TD-API | documented | Every thread of the block calls it |
| `barrier_all_warp` | b | `nvshmemx_barrier_all_warp` | E-NV13, E-NV15, E-NV16 | NV-COLL, NV-OVER, TD-API | documented | Every thread of the warp calls it |
| `barrier_all_wave` | b | — | undocumented | TD-SH | out of scope | |
| `barrier_all_wg` | b | — | undocumented | TD-SH | out of scope | |
| `barrier` | b | `nvshmem_barrier` | E-NV13, E-NV15 | NV-COLL, TD-API | documented | Over a team |
| `barrier_vec` | b | — | E-TD15 (docstring at the pinned commit) | TD-SH | out of scope | ACLSHMEM only |
| `barrier_block` | b | `nvshmemx_barrier_block` | E-NV13, E-NV15, E-NV16 | NV-COLL, NV-OVER | documented | |
| `barrier_warp` | b | `nvshmemx_barrier_warp` | E-NV13, E-NV15, E-NV16 | NV-COLL, NV-OVER | documented | |
| `sync_all` | b | `nvshmem_sync_all` | E-NV14 | NV-COLL, TD-API | documented | Does not complete remote updates |
| `sync_all_block` | b | `nvshmemx_sync_all_block` | E-NV14, E-NV16 | NV-COLL, NV-OVER, TD-PRIM | documented | |
| `sync_all_warp` | b | `nvshmemx_sync_all_warp` | E-NV14, E-NV16 | NV-COLL, NV-OVER, TD-PRIM | documented | |
| `sync` | b | `nvshmem_sync` | E-NV14 | NV-COLL | documented | Defined by the CUDA back end only; not reachable as `libshmem_device.sync` |
| `team_sync_block` | b | `nvshmemx_team_sync_block` | E-NV14, E-NV16 | NV-COLL, NV-OVER | documented | |
| `team_sync_warp` | b | `nvshmemx_team_sync_warp` | E-NV14, E-NV16 | NV-COLL, NV-OVER | documented | |
| `quiet` | b | `nvshmem_quiet` | E-NV2, E-NV3 | NV-ORD, TD-API | documented | Completes the caller's operations; grants no notification guarantee (E-NV2 and W6). Not relied on: the flush routines (E-NV23, added in 3.7.0). Absent from 3.6.5, not relied on: the GPU-thread note (E-NV22) |
| `quiet_pe` | b | — | undocumented | TD-SH | out of scope | |
| `fence` | a | `nvshmem_fence` | E-NV1, E-NV3 | NV-ORD, TD-API | documented | Orders delivery per target PE; non-blocking operations are not ordered (decision D1). Absent from 3.6.5, not relied on: the GPU-thread note (E-NV22) |
| `getmem_nbi_block` | c | `nvshmemx_getmem_nbi_block` | E-NV6, E-NV16 | NV-RMA, NV-OVER, TD-API | documented | Complete after `quiet` |
| `getmem_block` | c | `nvshmemx_getmem_block` | E-NV6, E-NV16 | NV-RMA, NV-OVER, TD-API | documented | Complete at return |
| `getmem_nbi_warp` | c | `nvshmemx_getmem_nbi_warp` | E-NV6, E-NV16 | NV-RMA, NV-OVER, TD-API | documented | Complete after `quiet` |
| `getmem_warp` | c | `nvshmemx_getmem_warp` | E-NV6, E-NV16 | NV-RMA, NV-OVER, TD-API | documented | Complete at return |
| `getmem_nbi` | c | `nvshmem_getmem_nbi` | E-NV6 | NV-RMA, TD-API | documented | Complete after `quiet` |
| `getmem_nbi_wave` | c | — | undocumented | TD-SH | out of scope | |
| `getmem_nbi_wg` | c | — | undocumented | TD-SH | out of scope | |
| `getmem` | c | `nvshmem_getmem` | E-NV6 | NV-RMA, TD-API | documented | Complete at return |
| `getmem_wave` | c | — | undocumented | TD-SH | out of scope | |
| `getmem_wg` | c | — | undocumented | TD-SH | out of scope | |
| `putmem_block` | c | `nvshmemx_putmem_block` | E-NV4, E-NV16 | NV-RMA, NV-OVER, TD-API | documented | Delivered only after `quiet` or a barrier |
| `putmem_nbi_block` | c | `nvshmemx_putmem_nbi_block` | E-NV5, E-NV16 | NV-RMA, NV-OVER, TD-API | documented | Complete after `quiet` |
| `putmem_warp` | c | `nvshmemx_putmem_warp` | E-NV4, E-NV16 | NV-RMA, NV-OVER, TD-API | documented | As `putmem_block` |
| `putmem_nbi_warp` | c | `nvshmemx_putmem_nbi_warp` | E-NV5, E-NV16 | NV-RMA, NV-OVER, TD-API | documented | Complete after `quiet` |
| `putmem` | c | `nvshmem_putmem` | E-NV4 | NV-RMA, TD-API | documented | As `putmem_block` |
| `putmem_wave` | c | — | undocumented | TD-SH | out of scope | |
| `putmem_wg` | c | — | undocumented | TD-SH | out of scope | |
| `putmem_nbi` | c | `nvshmem_putmem_nbi` | E-NV5 | NV-RMA, TD-API | documented | Complete after `quiet` |
| `putmem_nbi_wave` | c | — | undocumented | TD-SH | out of scope | |
| `putmem_nbi_wg` | c | — | undocumented | TD-SH | out of scope | |
| `putmem_signal_nbi` | c | `nvshmem_putmem_signal_nbi` | E-NV9, E-NV12 | NV-SIG, TD-PRIM | documented | The signal completes its own payload only |
| `putmem_signal` | c | `nvshmem_putmem_signal` | E-NV8, E-NV12 | NV-SIG, TD-PRIM | documented | As above |
| `putmem_signal_nbi_block` | c | `nvshmemx_putmem_signal_nbi_block` | E-NV9, E-NV12, E-NV16 | NV-SIG, NV-OVER, TD-API | documented | As above |
| `putmem_signal_block` | c | `nvshmemx_putmem_signal_block` | E-NV8, E-NV12, E-NV16 | NV-SIG, NV-OVER, TD-API | documented | As above |
| `putmem_signal_nbi_warp` | c | `nvshmemx_putmem_signal_nbi_warp` | E-NV9, E-NV12, E-NV16 | NV-SIG, NV-OVER, TD-PRIM | documented | As above |
| `putmem_signal_warp` | c | `nvshmemx_putmem_signal_warp` | E-NV8, E-NV12, E-NV16 | NV-SIG, NV-OVER, TD-PRIM | documented | As above |
| `putmem_signal_wave` | c | — | undocumented | TD-SH | out of scope | |
| `putmem_signal_wg` | c | — | undocumented | TD-SH | out of scope | |
| `putmem_signal_nbi_wave` | c | — | undocumented | TD-SH | out of scope | |
| `putmem_signal_nbi_wg` | c | — | undocumented | TD-SH | out of scope | |
| `signal_op` | b | `nvshmemx_signal_op` | E-NV10, E-NV12 | NV-SIG, TD-API | documented | Atomic `set` or `add`; `set` is monotone only under the obligation of S2.4. Not relied on: counted signals (E-NV24, added in 3.8.0) |
| `signal_wait_until` | b | `nvshmem_signal_wait_until` | E-NV11, E-NV12, E-NV21 | NV-SYNC, NV-FAQ, TD-API | documented | The word may be written by signal operations only. Not relied on: counted signals (E-NV24, added in 3.8.0). A fix of 3.8.0 concerns visibility after this wait (E-NV25; section 10, item 3) |
| `uint64_wait_until_equals` | b | — | undocumented | TD-SH | out of scope | |
| `ulong_put_signal` | c | — | undocumented | TD-SH | out of scope | |
| `broadcastmem` | c | — | undocumented | TD-SH | out of scope | |
| `broadcastmem_warp` | c | — | undocumented | TD-SH | out of scope | |
| `broadcastmem_block` | c | `nvshmemx_broadcastmem_block` | E-NV20, E-NV16 | NV-COLL, NV-OVER, TD-API | documented | Collective; `dest` updated on the local PE at return |
| `broadcast` | c | `nvshmem_TYPENAME_broadcast` | E-NV20 | NV-COLL | documented | As above |
| `broadcast_warp` | c | `nvshmemx_TYPENAME_broadcast_warp` | E-NV20, E-NV16 | NV-COLL, NV-OVER | documented | As above |
| `broadcast_block` | c | `nvshmemx_TYPENAME_broadcast_block` | E-NV20, E-NV16 | NV-COLL, NV-OVER | documented | As above |
| `fcollectmem` | c | — | undocumented | TD-SH | out of scope | |
| `fcollectmem_warp` | c | — | undocumented | TD-SH | out of scope | |
| `fcollectmem_block` | c | — | undocumented | TD-SH | out of scope | Listed in TD-API, but the CUDA back end does not define it |
| `fcollect` | c | `nvshmem_TYPENAME_fcollect` | E-NV20 | NV-COLL | documented | Collective; `dest` updated on the local PE at return |
| `fcollect_warp` | c | `nvshmemx_TYPENAME_fcollect_warp` | E-NV20, E-NV16 | NV-COLL, NV-OVER | documented | As above |
| `fcollect_block` | c | `nvshmemx_TYPENAME_fcollect_block` | E-NV20, E-NV16 | NV-COLL, NV-OVER | documented | As above |
| `putmem_rma` | c | `nvshmemi_transfer_rma_put` | undocumented | TD-SHC | weakest (W5) | The bound routine is internal to NVSHMEM and absent from its API documentation |
| `putmem_rma_block` | c | `nvshmemi_transfer_rma_put_block` | undocumented | TD-SHC | weakest (W5) | As above |
| `putmem_rma_warp` | c | `nvshmemi_transfer_rma_put_warp` | undocumented | TD-SHC | weakest (W5) | As above |
| `putmem_rma_nbi` | c | `nvshmemi_transfer_rma_put_nbi` | undocumented | TD-SHC | weakest (W5) | As above |
| `putmem_rma_nbi_block` | c | `nvshmemi_transfer_rma_put_nbi_block` | undocumented | TD-SHC | weakest (W5) | As above |
| `putmem_rma_nbi_warp` | c | `nvshmemi_transfer_rma_put_nbi_warp` | undocumented | TD-SHC | weakest (W5) | As above |
| `putmem_signal_rma` | c | `nvshmemi_transfer_put_signal` | undocumented | TD-SHC | weakest (W5) | As above |
| `putmem_signal_rma_nbi` | c | `nvshmemi_transfer_put_signal_nbi` | undocumented | TD-SHC | weakest (W5) | As above |
| `putmem_signal_rma_warp` | c | `nvshmemi_transfer_put_signal_warp` | undocumented | TD-SHC | weakest (W5) | As above |
| `putmem_signal_rma_nbi_warp` | c | `nvshmemi_transfer_put_signal_nbi_warp` | undocumented | TD-SHC | weakest (W5) | As above |
| `putmem_signal_rma_block` | c | `nvshmemi_transfer_put_signal_block` | undocumented | TD-SHC | weakest (W5) | As above |
| `putmem_signal_rma_nbi_block` | c | `nvshmemi_transfer_put_signal_nbi_block` | undocumented | TD-SHC | weakest (W5) | As above |
| `team_translate_pe` | none | `nvshmem_team_translate_pe` | E-NV19 | NV-TEAM | documented | |

### 5.4 `language_extra`

The first 13 rows are the names of the generic module
`TD:python/triton_dist/language/extra/language_extra.py`, which TD-API
documents. The other 40 exist only in the CUDA back end (TD-LEC) and no
document describes them; the one docstring among them states no guarantee
(E-TD14). `asm` in the `Note` column means that the source
defines the helper with inline assembly, which is the condition of the hard
constraint on undocumented helpers.

| Primitive | Pattern | Evidence | Source | Treatment | Note |
| --- | --- | --- | --- | --- | --- |
| `tid` | none | E-TD8 | TD-API | documented | |
| `atomic_cas` | a | E-TD1, E-TD8 | TD-SEM, TD-API | named assumption (A1) | Clause (b): no document states the return value. The update with `gpu`, `sys`, `acquire` or `release` is read by D5; with `cta`, `relaxed` or `acq_rel` by clause (a) |
| `atomic_add` | a | E-TD1, E-TD8, E-TD14 (docstring at the pinned commit), E-TD16 | TD-SEM, TD-API, TD-LEC | named assumption (A1) | Clause (b): the docstring says only "the result of atomic_add". Strings as for `atomic_cas`. The CUDA default semantic is `relaxed`: clause (a) |
| `__syncthreads` | b | E-TD8 | TD-API | documented | A barrier phase of one thread block |
| `ld` | a | E-TD1, E-TD8, E-TD16 | TD-SEM, TD-API, TD-LEC | documented | With `gpu`, `sys`, `acquire` or `release`: D5. With `cta`, `relaxed` or `acq_rel`: A1, clause (a). The CUDA default semantic is `relaxed`, so a call without a semantic needs clause (a) |
| `st` | a | E-TD1, E-TD8, E-TD16 | TD-SEM, TD-API, TD-LEC | documented | As `ld` |
| `ld_vector` | a | E-TD1, E-TD8, E-TD16 | TD-SEM, TD-API, TD-LEC | documented | The same primitive as `dl.ld_vector` |
| `st_vector` | a | E-TD1, E-TD8, E-TD16 | TD-SEM, TD-API, TD-LEC | documented | The same primitive as `dl.st_vector` |
| `pack` | none | E-TD8 | TD-API | documented | |
| `unpack` | none | E-TD8 | TD-API | documented | |
| `threads_per_warp` | none | E-TD8 | TD-API | documented | |
| `num_threads` | none | E-TD8 | TD-API | documented | |
| `num_warps` | none | E-TD8 | TD-API | documented | |
| `tl_type_to_ptx_suffix` | none | undocumented | TD-LEC | weakest (W3) | A compile-time helper |
| `__fence` | a | undocumented | TD-LEC | weakest (W2) | asm; a fence with no documented effect |
| `tma_sync` | b | undocumented | TD-LEC | weakest (W2) | asm |
| `load_v4_u32` | a | undocumented | TD-LEC | weakest (W1) | asm |
| `load_v4_b32` | a | undocumented | TD-LEC | weakest (W1) | asm; its scope and semantic arguments give nothing |
| `load_v4_s32` | a | undocumented | TD-LEC | weakest (W1) | asm |
| `load_v2_b64` | a | undocumented | TD-LEC | weakest (W1) | asm |
| `st_v4_u32` | a | undocumented | TD-LEC | weakest (W1) | asm |
| `st_v4_b32` | a | undocumented | TD-LEC | weakest (W1) | asm; its scope and semantic arguments give nothing |
| `st_v2_u32` | a | undocumented | TD-LEC | weakest (W1) | asm |
| `multimem_st_b64` | a | undocumented | TD-LEC | weakest (W1) | asm; a plain store to the replica on every PE of the team |
| `multimem_st_b32` | a | undocumented | TD-LEC | weakest (W1) | As above |
| `multimem_st_v2` | a | undocumented | TD-LEC | weakest (W1) | As above |
| `multimem_st_v4` | a | undocumented | TD-LEC | weakest (W1) | As above |
| `multimem_st_p_b32` | a | undocumented | TD-LEC | weakest (W1) | As above |
| `multimem_ld_reduce_p_v4` | a | undocumented | TD-LEC | weakest (W1) | asm; a plain load of every replica; the combined value is unspecified |
| `multimem_ld_reduce_v4` | a | undocumented | TD-LEC | weakest (W1) | As above. Its docstring at the pinned commit names an instruction and states no guarantee (E-TD14) |
| `laneid` | none | undocumented | TD-LEC | weakest (W3) | |
| `ntid` | none | undocumented | TD-LEC | weakest (W3) | |
| `red_release` | a | undocumented | TD-LEC | weakest (W2) | asm; a plain read-modify-write |
| `arrive_inc` | b | undocumented | TD-LEC | weakest (W4) | Triton source; calls `__syncthreads` and `red_release` |
| `arrive_inc_asm` | a | undocumented | TD-LEC | weakest (W2) | asm; a plain read-modify-write |
| `ld_b32` | a | undocumented | TD-LEC | weakest (W1) | |
| `ld_acquire` | a | undocumented | TD-LEC | weakest (W1) | Despite its name no acquire is assumed (section 9, item 10) |
| `st_b32` | a | undocumented | TD-LEC | weakest (W1) | |
| `atomic_add_per_warp` | a | undocumented | TD-LEC | weakest (W4) | Triton source; calls `atomic_add` |
| `wait_eq` | b | undocumented | TD-LEC | weakest (W4) | Triton source; calls `ld_acquire` and `__syncthreads` |
| `__shfl_sync_with_mode_i32` | none | undocumented | TD-LEC | weakest (W3) | asm |
| `__shfl_sync_i32` | none | undocumented | TD-LEC | weakest (W4) | Triton source |
| `__shfl_up_sync_i32` | none | undocumented | TD-LEC | weakest (W4) | Triton source |
| `__shfl_down_sync_i32` | none | undocumented | TD-LEC | weakest (W4) | Triton source |
| `__shfl_xor_sync_i32` | none | undocumented | TD-LEC | weakest (W4) | Triton source |
| `__ballot_sync` | none | undocumented | TD-LEC | weakest (W3) | asm |
| `globaltimer_lo` | none | undocumented | TD-LEC | weakest (W3) | asm |
| `globaltimer_hi` | none | undocumented | TD-LEC | weakest (W3) | asm |
| `globaltimer` | none | undocumented | TD-LEC | weakest (W3) | asm |
| `smid` | none | undocumented | TD-LEC | weakest (W3) | asm |
| `membar` | a | undocumented | TD-LEC | weakest (W2) | asm; a fence with no documented effect |
| `fence` | a | undocumented | TD-LEC | weakest (W2) | asm; not `libshmem_device.fence` |
| `pack_b32_v2` | none | undocumented | TD-LEC | weakest (W3) | asm; a pure value |

### 5.5 TileScale, kernel side

The names are used as `T.<name>` in kernel code. NVSHMEM documentation does
not apply to any row (E-TS1).

| Primitive | Pattern | Evidence | Source | Treatment | Note |
| --- | --- | --- | --- | --- | --- |
| `fence_gpu` | a | E-TS14 (docstring at the pinned commit) | TS-BUILTIN | weakest (W6) | Device scope is stated; no memory order, hence no ordering/synchronization edge is granted |
| `fence_sys` | a | E-TS14 (docstring at the pinned commit) | TS-BUILTIN | weakest (W6) | System scope is stated; no memory order, hence no ordering/synchronization edge is granted |
| `get_rank` | none | E-TS2 | TS-API | documented | |
| `get_num_ranks` | none | E-TS2 | TS-API | documented | |
| `put_warp` | a | E-TS1, E-TS3 | TS-API, TS-COMM | weakest (W1) | The address remapping is documented; the access is not |
| `get_warp` | a | E-TS1, E-TS3 | TS-API, TS-COMM | weakest (W1) | As above |
| `put_block` | a | E-TS1, E-TS3 | TS-API, TS-COMM | weakest (W1) | As above |
| `get_block` | a | E-TS1, E-TS3 | TS-API, TS-COMM | weakest (W1) | As above |
| `ld` | a | E-TS5, E-TS13 | TS-API, TS-COMM | named assumption (A1) | Clause (c), for every string: `gpu`, `sys`, `acquire`, `relaxed`. With `weak` (the default), `volatile`, `cta` or `nc=True`: W1 |
| `st` | a | E-TS5, E-TS13 | TS-API, TS-COMM | named assumption (A1) | Clause (c), for every string: `gpu`, `sys`, `release`, `relaxed`. With `weak` (the default), `volatile` or `cta`: W1 |
| `atom_add` | a | E-TS5, E-TS13 | TS-API, TS-COMM | named assumption (A1) | Clause (c), for every string: `gpu`, `sys`, `relaxed`, `acquire`, `release`, `acq_rel`. The return value is documented (docstring at the pinned commit) |
| `wait_eq` | b | E-TS6, E-TS13 | TS-API, TS-SYNC | named assumption (A1) | Clause (c), for every string: `gpu`, `sys`, `acquire`. With `volatile`: a poll by plain loads (W1) |
| `wait_ne` | b | E-TS6 | TS-API, TS-SYNC | named assumption (A1) | As above |
| `wait_ge` | b | E-TS6 | TS-API, TS-SYNC | named assumption (A1) | As above |
| `wait_le` | b | E-TS6 | TS-API, TS-SYNC | named assumption (A1) | As above |
| `wait_gt` | b | E-TS6 | TS-API, TS-SYNC | named assumption (A1) | As above |
| `wait_lt` | b | E-TS6 | TS-API, TS-SYNC | named assumption (A1) | As above |
| `init_barrier_gpu` | b | E-TS7 | TS-API, TS-SYNC | weakest (W6) | The counter protocol and any memory ordering are not stated |
| `arrive_barrier_gpu` | b | E-TS7 | TS-API, TS-SYNC | weakest (W6) | As above |
| `wait_barrier_gpu` | b | E-TS7 | TS-API, TS-SYNC | weakest (W6) | As above |
| `sync_barrier_gpu` | b | E-TS7 | TS-API, TS-SYNC | weakest (W6) | As above |
| `barrier_blocks` | b | E-TS7, E-TS8 (docstring at the pinned commit) | TS-API, TS-SYNC | documented | A rendezvous with a release at system scope on entry and no acquire on exit |
| `sync_blocks` | b | E-TS7, E-TS8 (docstring at the pinned commit) | TS-API, TS-SYNC | documented | A rendezvous with no memory ordering |
| `multimem_ld_reduce` | a | E-TS9 | TS-API, TS-MM | weakest (W1) | A plain load of every replica; the order of the reduction is unspecified |
| `multimem_st` | a | E-TS9 | TS-API, TS-MM | weakest (W1) | A plain store to every replica |
| `multimem_red` | a | E-TS9 | TS-API, TS-MM | weakest (W1) | A plain read-modify-write of every replica |
| `multimem_tma_store` | c | E-TS9 | TS-API, TS-MM | weakest (W6) | The local sequence is documented; visibility at other ranks is not |
| `multimem_signal` | b | E-TS9 | TS-API, TS-MM | weakest (W2) | A plain store to every replica of the signal |
| `multimem_signal_add` | b | E-TS9 | TS-API, TS-MM | weakest (W2) | A plain read-modify-write of every replica |
| `copy` | a | E-TS3 | TS-API, TS-COPY | weakest (W1) | Row for the `src_pe` and `dst_pe` arguments; the local copy is base TileLang |
| `tma_copy` | c | E-TS3, E-TS4, E-TS10 | TS-API, TS-COPY | weakest (W6) | Completion by the user's wait is documented; visibility at the remote rank is not |
| `atomic_add` | a | E-TS11 (docstring at the pinned commit) | TS-ATOM | weakest (W6) | Row for `dst_pe`. Atomic, but no scope is stated |
| `atomic_addx2` | a | E-TS11 (docstring at the pinned commit) | TS-ATOM | weakest (W6) | As above |

### 5.6 TileScale, host side

Source of the names: `TS:tilelang/distributed/host.py`.

| Primitive | Pattern | Evidence | Source | Treatment | Note |
| --- | --- | --- | --- | --- | --- |
| `set_signal` | b | E-TS12 | TS-API | weakest (W6) | A stream-ordered write of a 32-bit word; atomicity and ordering are not stated |
| `wait_eq` | b | E-TS12 | TS-API | weakest (W6) | A stream-ordered wait; what it makes visible is not stated |

### 5.7 Further names of the tutorials in scope

The user decided on 2026-11-12 to add these three names. The first two are
host functions of `triton_dist.kernels.common_ops` (tutorials 06 and 07); the
third is base Triton (tutorial 08).

| Primitive | Pattern | Evidence | Source | Treatment | Note |
| --- | --- | --- | --- | --- | --- |
| `common_ops.set_signal` | b | E-TD17 | TD-KCO, TD-T07 | weakest (W2) | Host side. No document describes it, and the bound driver routine is outside the sources of section 1. A plain store of the signal word, with no ordering after the copy enqueued before it. Consequence: I6 |
| `common_ops.wait_eq` | b | E-TD17 | TD-KCO, TD-T07 | weakest (W1) | Host side, as above. A poll by plain loads; its return creates no synchronization edge |
| `tl.atomic_add` | a | E-TR1 (docstring at the pinned commit), E-TD18 | TR-CORE, TR-API, TD-T08 | named assumption (A1) | Atomicity and the return of the previous value are documented. Clause (d), for every string: Triton lists them and defines none, also not `release` and `gpu` of tutorial 08. Without `sem` the default is `acq_rel` |

## 6. Inferences and decisions that apply to many rows

**Rules for the weakest reasonable semantics.**

| Rule | Applies to | Semantics |
| --- | --- | --- |
| W1 | An undocumented load, store or copy | A plain, unordered, non-atomic access to each element. A scope or semantic argument gives no ordering |
| W2 | An undocumented update, signal or fence | An update is a plain, non-atomic read-modify-write. A fence orders nothing and creates no synchronization edge |
| W3 | An undocumented value or query | No memory event. The result is an unspecified value of the result type |
| W4 | An undocumented function written in front-end source (`@triton.jit`, or a class of such methods) | The semantics of its body, each callee by its own row. The function adds no guarantee |
| W5 | An undocumented transfer | An initiation event with no completion event. From the initiation on the destination holds unspecified data, and no documented operation completes the transfer. A signal that belongs to it is not guaranteed to be written |
| W6 | An aspect that the documentation of a primitive omits | The row names the aspect. The omitted guarantee does not hold; the documented rest does |

**Inferences.**

- I1. Under W1 and W2 a conflicting concurrent access from another thread is a
  data race (E-CU2), which the semantic architecture treats as a failed
  execution. An undocumented helper is therefore usable only on data that no
  other agent touches concurrently.
- I2. `dl.wait` and the TileScale waits are not NVSHMEM routines. E-NV21 and
  E-NV1 then give the reading side nothing: after an NVSHMEM `fence` or `quiet`
  on the writing side, a reader that polls with `dl.wait` has no NVSHMEM
  guarantee. Its guarantee can come only from the CUDA C++ model: through D5
  for `dl.wait`, through A1 for the TileScale waits. This is the finding of
  ticket #11.
- I3. A wait at `gpu` scope creates no synchronization edge with a writer on
  another rank, because the device scope relates only threads of one device
  (E-CU1, E-CU3, E-TD1). This is the standing decision of ticket #24.
- I4. For `dl.wait` the comparison is not stated (W6). The proof may use only
  what both readings give: after the return, each signal was read with a value
  at least `waitValue`; the wait is guaranteed to return only if each signal
  holds exactly `waitValue` from some point until the wait reads it. The user
  adopted this reading on 2026-11-12 (D6).
- I5. Direct reading of the frozen manual replaces the prior indirect
  release-note inference. Section 4.2's comparison establishes the 3.6.5
  passages, including the different quiet visibility wording (E-NV2).
  E-NV22/23/24 are absent and remain unusable. A later implementation fix
  (E-NV25) is not proof that the earlier manual lacked E-NV11's contract.
- I6. Under W2 the store of `common_ops.set_signal` is plain. The kernel of
  tutorial 07 reads the same words with `dl.wait` (E-TD17). By I1 that pair is
  a data race. Tutorial 07 as written is therefore not provable under this
  table (section 10, item 2).

**Decisions.**

- D1. `fence` does not order non-blocking operations. The sources disagree
  (E-NV3); the weaker reading is taken, as ticket #11 decided.
- D2. Barrier and quiet visibility is taken as global only on a system with
  NVLink alone (E-NV15); otherwise it covers only the local PE's objects.
  Whether the two-GPU node of the project is such a system was not checked
  here. The release notes state a limit without that exception (E-NV26);
  whether D2 is weakened is open (section 10, item 4).
- D3. Where E-TD10 contradicts E-TD1, the definition E-TD1 governs. The example
  and best practice 5 of TD-SEM are not guarantees. The user confirmed this on
  2026-11-12; the contradiction is not reported upstream.
- D4. A `_block` or `_warp` routine is one event of the whole block or warp. A
  kernel in which not every thread calls it with the same arguments is
  rejected (E-NV16).
- D5. Documented reading of scope and order (user's decision of 2026-11-12).
  In kernel code of Triton-distributed the scope strings `gpu` and `sys`
  denote `cuda::thread_scope_device` and `cuda::thread_scope_system`, and the
  semantic strings `acquire` and `release` denote the memory order of the same
  name. A Triton-distributed primitive that takes these strings performs, for
  each element, an atomic operation of the CUDA C++ memory model at that scope
  and order: `ld` and `ld_vector` a load; `st` and `st_vector` a store;
  `atomic_add` and `atomic_cas` an atomic update; `wait` a sequence of loads
  that ends with a load satisfying its condition (D6). The reading formalises
  the prose definitions of E-TD1 with E-CU1, E-CU3 and E-CU4. It is part of
  the semantics, not a registered assumption, and a row that rests on it is
  `documented`. It covers these four strings only; everything else is in A1.
- D6. The comparison of `dl.wait` is the intersection of both readings, I4
  (user's decision of 2026-11-12).
- D7. NVSHMEM items that a proof does not rely on (rule of section 1; user's
  decision of 2026-11-12).

  | Item | Evidence | Dated by | Status | Rows that say so |
  | --- | --- | --- | --- | --- |
  | Flush routines | E-NV23 | Release notes: added in 3.7.0 (E-NV25); absent from the 3.6.5 ordering page | After 3.6.5: not relied on | `quiet`. No row binds a flush routine |
  | Counted signals | E-NV24 | Release notes: added in 3.8.0 (E-NV25); absent from the 3.6.5 signal page | After 3.6.5: not relied on | `signal_op`, `signal_wait_until`. No row binds a counted-signal routine |
  | GPU-thread notes on `fence` and `quiet` | E-NV22 | Direct archive comparison: absent from the 3.6.5 ordering page | Not part of the version of record: not relied on | `fence`, `quiet` |

  E-NV22 formerly supplied the cross-thread claim that one elected GPU thread’s
  fence/quiet can order/complete earlier operations issued by other block threads
  after synchronization; that claim is not used for 3.6.5.

  Consequence of the third line: a `fence` or a `quiet` is taken to cover the
  operations that the calling thread issued. That a `fence` after
  `__syncthreads` also covers the stores of the other threads of the block,
  the pattern of tutorial 01 (E-TD11), is not assumed.

## 7. Named assumptions

Each entry has the fields of the registry of ticket S5.3 as the map states
them: name, statement, source or rationale, affected kernels. The class of
ticket S5.1 and the status are added. The freeze ticket S5.1 fixes the names.
Affected kernels were found by text search at the pins; reachability from each
kernel entry point was not traced.

### A1 `undocumented_scope_order_vocabulary`

A1 is the remainder of the former `scope_order_vocabulary` after the user's
decision of 2026-11-12. The four strings that the Triton-distributed
documentation defines moved to D5 and are no longer assumed. The name of the
remainder was chosen in this revision.

| Field | Value |
| --- | --- |
| Name | `undocumented_scope_order_vocabulary` |
| Class | semantic source |
| Status | Kept as a registered named assumption, in this narrower form, by the user's decision of 2026-11-12. Clause (d) was added by this revision for the row of section 5.7 and is not covered by that decision (section 10, item 1) |
| Statement | Where no document defines a string or a return value, the reading of D5 is assumed: the string denotes the CUDA C++ thread scope or memory order of the same name, and the primitive performs, for each element, an atomic operation of the CUDA C++ memory model at that scope and order. (a) In kernel code of Triton-distributed the scope string `cta` denotes `cuda::thread_scope_block`, and the semantic strings `relaxed` and `acq_rel` denote the memory order of the same name. (b) `language_extra.atomic_add` is a fetch-add that returns the previous value, and `language_extra.atomic_cas` is a strong compare-exchange that returns the value read. (c) In kernel code of TileScale the scope strings `gpu` and `sys` denote `cuda::thread_scope_device` and `cuda::thread_scope_system`, and the semantic strings `relaxed`, `acquire`, `release` and `acq_rel` denote the memory order of the same name; `T.ld` is a load, `T.st` a store, `T.atom_add` a fetch-add, and a wait a sequence of loads that ends with a load satisfying its condition. (d) For base Triton `tl.atomic_add` in kernel code of Triton-distributed the scope strings `cta`, `gpu` and `sys` denote `cuda::thread_scope_block`, `cuda::thread_scope_device` and `cuda::thread_scope_system`, and the semantic strings `relaxed`, `acquire`, `release` and `acq_rel` denote the memory order of the same name. Nothing is assumed for any other string (`weak`, `volatile`, `sc`, the empty string), nor for `cta` in TileScale |
| Source or rationale | (a) TD-SEM defines only `gpu`, `sys`, `acquire` and `release` (E-TD1, D5); TD-API takes a scope and a semantic without listing their values (E-TD8), and the CUDA definitions default to `relaxed` (E-TD16). (b) TD-API says "Atomic add operation" and "Atomic compare-and-swap" and describes no return value (E-TD8); the docstring says "the result of atomic_add" (E-TD14). (c) TS-API lists its strings and defines none; it does not list `cta` (E-TS13). The return value of `T.atom_add` is not assumed: its docstring states it (E-TS5). (d) The Triton documentation lists the strings with glosses and defines none; it does state atomicity and the return value, which are not assumed (E-TR1). For all clauses: the map names the CUDA C++ memory model as the source of semantics for scoped accesses (E-CU1, E-CU3, E-CU4) |
| Affected kernels | Listed for tutorials 01 to 08 and the helper modules they import; other modules under `kernels/nvidia/` were not listed. Triton-distributed, clause (a): `kernels/nvidia/reduce_scatter.py` and `kernels/nvidia/allgather_gemm.py` (`st` without a semantic, hence `relaxed`). Clause (b): `kernels/nvidia/common_ops.py` (`atomic_cas`, `atomic_add`; tutorial 05 calls its `barrier_all_intra_node_atomic_cas_block`), `kernels/nvidia/reduce_scatter.py` and `kernels/nvidia/gemm_reduce_scatter.py` (`atomic_add`). Clause (d): tutorial 08 and `kernels/nvidia/gemm_reduce_scatter.py` (`tl.atomic_add` with `release` and `gpu`). Tutorials 05 to 08 import `common_ops.py`, `reduce_scatter.py` and `allgather_gemm.py`; `gemm_reduce_scatter.py` holds the same call as tutorial 08. No longer affected: `dl.wait(..., "gpu", "acquire")` of tutorials 01 and 07, and every `ld` or `st` with `gpu` or `sys` and `acquire` or `release` (D5). TileScale, clause (c): `examples/distributed/allgather_gemm/example_allgather_gemm_overlapped.py`, `example_allgather_gemm_specialized.py` and `gemm_allreduce/example_gemm_ar_specialized.py` (`T.wait_eq`, `T.wait_ge`); `gemm_reducescatter/example_gemm_rs_overlapped.py` (`T.st`, `T.atom_add`). The targets are the `_overlapped` variants (section 9, item 9) |

### A2 `notify_atomic_release`

| Field | Value |
| --- | --- |
| Name | `notify_atomic_release` |
| Class | semantic source |
| Status | Accepted by the user on 2026-11-12 as a registered named assumption |
| Statement | `dl.notify(ptr, rank, signal, sig_op, comm_scope)` performs one atomic update of the 64-bit signal word at `ptr` on rank `rank`, with release ordering, by the calling thread: a store of `signal` for `sig_op="set"`, a fetch-add of `signal` for `sig_op="add"`. The scope is the system scope for `comm_scope="intra_node"` and the device scope for `comm_scope="gpu"`. Nothing is assumed for `comm_scope="inter_node"`, and nothing about stores issued by other threads |
| Source or rationale | The documentation says only "Send a signal to a remote rank" with `sig_op` `"set"` or `"add"` (E-TD2, E-TD3). It states no atomicity, scope or ordering, and it does not list the value `gpu`, which tutorial 08 passes. The examples of TD-SEM and TD-T01 pair `notify` with `wait(..., "acquire")` as the only synchronization between producer and consumer (E-TD10, E-TD11). Without the assumption `notify` falls under W2, and each `notify` races with the matching `wait` |
| Affected kernels | Tutorial 01 (two calls, `intra_node`); tutorial 08 and `kernels/nvidia/gemm_reduce_scatter.py` (`comm_scope="gpu"`, own rank) |

## 8. Counts

| Section | Rows | documented | named assumption | weakest | out of scope |
| --- | --- | --- | --- | --- | --- |
| 5.1 `dl.*` | 25 | 21 | 1 | 3 | 0 |
| 5.2 `dl.tma.*` | 5 | 1 | 0 | 4 | 0 |
| 5.3 `libshmem_device.*` | 86 | 49 | 0 | 12 | 25 |
| 5.4 `language_extra` | 53 | 11 | 2 | 40 | 0 |
| 5.5 TileScale, kernel side | 33 | 4 | 9 | 20 | 0 |
| 5.6 TileScale, host side | 2 | 0 | 0 | 2 | 0 |
| 5.7 Further names of the tutorials | 3 | 0 | 1 | 2 | 0 |
| Total | 207 | 86 | 13 | 83 | 25 |

By pattern: a 52, b 45, c 60, none 50.

Of the 13 rows under a named assumption, 12 are under A1 and one (`dl.notify`)
is under A2.

Before the decisions of 2026-11-12 the table had 202 rows: 79 documented, 19
under a named assumption, 79 weakest and 25 out of scope. Seven rows moved
from A1 to `documented` (D5): `wait`, `ld_vector` and `st_vector` of 5.1, and
`ld`, `st`, `ld_vector` and `st_vector` of 5.4. Three rows are new (5.7). No
other existing row changed its treatment. S6.8 adds two weakest fence rows
to 5.5, with no assumption added or removed.

## 9. Decisions of 2026-11-12

The user decided the open questions of the first draft on 2026-11-12. Each
item names where this document applies the decision.

1. **A1 is split.** The reading of `gpu`, `sys`, `acquire` and `release`,
   which the Triton-distributed documentation defines in prose, is part of the
   semantics and counts as documented (D5). What no document states stays a
   registered named assumption in a narrower form (A1, section 7).
2. **A2 is accepted** as a registered named assumption (section 7).
3. **Docstrings** in the pinned source count as front-end documentation of
   lower rank (section 1). The rows that rest on one say so.
4. **Extent.** `set_signal` and `wait_eq` of `triton_dist.kernels.common_ops`
   and base Triton `tl.atomic_add` with `sem` and `scope` get rows (section
   5.7). The other names outside the modules of the ticket stay out until a
   kernel in scope uses them (section 2).
5. **NVSHMEM version.** The version of record is 3.6.5, the one on the GPU
   node; the original reading used `latest` 3.8.0, and S6.8 directly
   checks the frozen 3.6.5 archive. A proof may rely only on
   guarantees that already hold in 3.6.5 (section 1, D7).
6. **`dl.wait`.** Its comparison is the intersection of both readings (I4,
   D6).
7. **Contradiction in the Triton-distributed documentation.** It is resolved
   by the definition and is not reported upstream (D3).
8. **`out of scope`** is an accepted fourth label (section 1).
9. **TileScale `_specialized` variants.** No named assumption is introduced
   for them; the targets are the `_overlapped` variants. `volatile` and `weak`
   semantics, the multimem signals and `T.atomic_add(dst_pe=)` keep the
   weakest semantics, so `example_allgather_gemm_specialized.py`,
   `example_gemm_ar_specialized.py` and `example_gemm_rs_specialized.py`
   cannot be proved under this table.
10. **Undocumented CUDA helpers outside the tutorials.** They are decided when
    a kernel in scope needs one. Until then W1 and W2 apply. For reference,
    the search under `kernels/nvidia/` finds `ld_acquire` 19 times, `ld_b32`
    11, `membar` 11, `multimem_st_b64` 9, `load_v4_u32` 6,
    `atomic_add_per_warp` 5 and `multimem_ld_reduce_v4` 4.

## 10. Still open

These points came up while the decisions were applied. No decision covers
them.

1. **Base Triton strings.** `tl.atomic_add` is a base Triton operation, and
   the Triton documentation lists its strings without defining them (E-TR1).
   This revision put all of them under A1, clause (d), by analogy with
   TileScale. The alternative is to let D5 cover `gpu`, `sys`, `acquire` and
   `release` for any operation in kernel code of Triton-distributed; the row
   would then be `documented` for the call of tutorial 08. Either way the row
   needs a confirmation.
2. **Tutorial 07 and the host signals.** `common_ops.set_signal` has the
   weakest semantics, so it races with the `dl.wait` of the kernel (I6).
   Tutorial 07 as written needs a further decision on the host signal; no
   named assumption is introduced here. In addition, the calls in tutorials 06
   and 07 do not match the definitions at the pin (E-TD17); this was read, not
   run.
3. **Contract versus implementation.** The frozen manual now directly
   establishes E-NV11's signal-wait contract. The 3.8.0 release notes report a
   later implementation visibility fix (E-NV25). The applicability of that bug
   to the project's installed 3.6.5/runtime remains unchecked; this does not
   erase the earlier documented guarantee or establish a passing GPU run.
4. **Global visibility: conflict within 3.6.5.** The two readings are (a)
   retain D2's NVLink-only exception from the manual, or (b) weaken visibility
   to source/destination only following the release limitation. The following
   are quotations of 3.6.5 passages, not a decision:

   | Passage | What the 3.6.5 source says |
   | --- | --- |
   | [Ordering, quiet](https://archive.docs.nvidia.com/nvshmem/api/3.6.5/gen/api/ordering.html#nvshmem-quiet) (E-NV2) | "On systems with only NVLink, all operations on symmetric data objects are guaranteed to be complete and visible to all PEs when nvshmem_quiet returns. On systems with both NVLink and InfiniBand, visibility is only guaranteed at the destination PE." |
   | [Collectives, barrier](https://archive.docs.nvidia.com/nvshmem/api/3.6.5/gen/api/collectives.html#nvshmem-barrier) (E-NV13) | "nvshmem_barrier ensures that all previously issued stores and remote memory updates, including AMOs and RMA operations, done by any of the PEs in the active set are complete before returning. On systems with only NVLink, updates are globally visible, whereas on systems with both NVLink and InfiniBand, NVSHMEM only guarantees that updates to the memory of a given PE are visible to that PE." |
   | [Memory Model, Visibility Guarantees](https://archive.docs.nvidia.com/nvshmem/api/3.6.5/gen/mem-model.html#visibility-guarantees) (E-NV15) | "On systems with both NVLink and InfiniBand, the NVSHMEM synchronization operations including nvshmem_barrier, nvshmem_barrier_all, nvshmem_quiet, nvshmem_wait_until_*, and nvshmem_test_* only guarantee visibility of updates to the local PE’s symmetric objects. However, on systems with only NVLink, these operations guarantee global visibility of updates to symmetric objects." |
   | [Using NVSHMEM, Data Consistency](https://archive.docs.nvidia.com/nvshmem/api/3.6.5/using.html#data-consistency) (E-NV27) | "When updates performed by a PE must be ordered or made visible to PEs other than the target PE, use the nvshmem_quiet operation." |
   | [3.6.5 release Limitations](https://github.com/NVIDIA/nvshmem/releases/tag/v3.6.5-0) (E-NV26) | "`nvshmem_barrier*`, `nvshmem_quiet`, and `nvshmem_wait_until` only ensure ordering and visibility between the source and destination PEs. They *do not* ensure global ordering and visibility." |
   | [Ordering, fence Notes](https://archive.docs.nvidia.com/nvshmem/api/3.6.5/gen/api/ordering.html#nvshmem-fence) (E-NV1; unchanged in latest) | "nvshmem_quiet guarantees completion of all operations on symmetric data objects which makes the updates visible to all other PEs." |

   The user still chooses whether to weaken D2. The project's topology was
   not checked, and the stronger reading is not silently made universal.
5. **Triton version in the `triton-dist` environment.** Triton-distributed
   bundles Triton at commit `f797708`, not `v3.5.1`. The passage cited for
   `tl.atomic_add` is the same at both (E-TR1). If further base Triton rows
   are added, the commit to cite has to be chosen.

6. **Cross-GPU atomic support.** E-CU5 requires a native-atomic P2P property
   for GPU-memory system-scope atomics. D5, A1 and A2 do not currently state
   that condition. The user must decide whether to record it as a launch
   precondition or an assumption. No assumption or launch contract is changed.

Not checked in this revision: the call graph from each kernel to the helpers
it reaches; runtime applicability of the later signal-wait fix; the
NVSHMEM versions and P2P capabilities on the GPU node.
