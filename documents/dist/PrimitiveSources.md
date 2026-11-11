# Documentation sources of the distributed primitives

Status: draft of ticket S2.2, **not signed**. Sources were read on
2026-11-11. Nothing here is a proved result; the table fixes what the
semantics of `VeriTile/Dist/` may assume about each primitive.

The table has one row per primitive of Triton-distributed (`dl.*`,
`libshmem_device.*`, `language_extra`) and of TileScale. A row gives the
pattern, the documented guarantee, the citation and the treatment. The Lean
instance table of ticket S2.6 refers to a row by section and primitive name.

## 1. Rules

**What counts as a source** (hard constraints of the map):

- Front-end documentation: Triton, TileLang, Triton-distributed, TileScale.
- The CUDA C++ memory model: `cuda::thread_scope_*` and the libcu++ atomics.
- The NVSHMEM API documentation.

Nothing at or below PTX is cited. A guarantee that the documentation does not
state does not hold. A source file of a front end is used for two things only:
that a name exists, and which documented routine a binding names. Docstrings in
the pinned source are treated as front-end documentation of lower rank and are
marked `docstring` (open question 5).

**Pins.** Triton-distributed `7908e4e` (full hash
`7908e4ea9010bb2238f9f05b28c904fd4b71c60a`); TileScale `71e7145` (full hash
`71e71457d40ed6e6a2dad297936bfdbff21d104b`).

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
It follows the map, which excludes AMD back ends (open question 8).

**Evidence, inference, decision.** Section 4 holds evidence: what a source
says, quoted. In the tables the columns `Evidence` and `Source` are evidence,
`Note` is inference and `Treatment` is the decision. Sections 6 and 7 hold the
decisions that apply to many rows.

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
| TileScale, kernel side | top-level functions of `tilelang/language/distributed/{common,comm,sync,multimem}.py` (27), plus the functions of `copy_op.py`, `atomic.py` and `builtin.py` that take `src_pe` or `dst_pe` (4) | 31 |
| TileScale, host side | the signal utilities of `tilelang/distributed/host.py` | 2 |
| Total | | 202 |

Of the 85 generic `libshmem_device` functions, 60 have a definition in
`extra/cuda/libnvshmem_device.py` and 25 do not.

Not listed, because they are constants or types and not operations: the
comparison, signal, atomic-operation and team constants of
`libshmem_device.py`; the 6 constants of `dl.tma`; the TileScale enums
`BinaryRelation`, `WaitScope`, `WaitSemantics` and `MultimemReduceOp`.

Not covered, because they lie outside the modules the ticket names (open
question 6): `extra/cuda/tma_language.py` (40 functions), the host-side
functions of `triton_dist.utils`, the helpers under `triton_dist/kernels/`,
base Triton (`tl.atomic_add` with `sem` and `scope`) and base TileLang
(`T.tma_store_wait`, `T.sync_threads`).

## 3. Sources

Paths with prefix `TD:` are in Triton-distributed at `7908e4e`; paths with
prefix `TS:` are in TileScale at `71e7145`. Every URL was accessed on
2026-11-11.

| ID | Source |
| --- | --- |
| TD-PRIM | `TD:docs/primitives.md` |
| TD-API | `TD:docs/python-api/triton-dist.language.rst` |
| TD-SEM | `TD:docs/python-api/triton-dist.semantics.rst` |
| TD-T01 | `TD:tutorials/01-distributed-notify-wait.py` |
| TD-INIT | `TD:python/triton_dist/language/__init__.py` (names only) |
| TD-SMEM | `TD:python/triton_dist/language/smem_ops.py` (docstrings) |
| TD-TMA | `TD:python/triton_dist/language/tma/pipeline.py`, `TD:python/triton_dist/language/tma/tensormap.py` (docstrings) |
| TD-SH | `TD:python/triton_dist/language/extra/libshmem_device.py` (names only) |
| TD-SHC | `TD:python/triton_dist/language/extra/cuda/libnvshmem_device.py` (names and bound routine names only) |
| TD-LEC | `TD:python/triton_dist/language/extra/cuda/language_extra.py` (names only) |
| TS-API | `TS:docs/distributed_api_reference.md` |
| TS-COMM | `TS:tilelang/language/distributed/comm.py` (docstrings) |
| TS-SYNC | `TS:tilelang/language/distributed/sync.py` (docstrings) |
| TS-MM | `TS:tilelang/language/distributed/multimem.py` (docstrings) |
| TS-COPY | `TS:tilelang/language/copy_op.py` (docstrings) |
| TS-ATOM | `TS:tilelang/language/atomic.py` (docstrings) |
| NV-MM | NVSHMEM 3.8.0, Memory Model: <https://docs.nvidia.com/nvshmem/api/latest/gen/mem-model.html> |
| NV-ORD | NVSHMEM 3.8.0, Memory Ordering: <https://docs.nvidia.com/nvshmem/api/latest/gen/api/ordering.html> |
| NV-RMA | NVSHMEM 3.8.0, Remote Memory Access: <https://docs.nvidia.com/nvshmem/api/latest/gen/api/rma.html> |
| NV-SIG | NVSHMEM 3.8.0, Signaling Operations: <https://docs.nvidia.com/nvshmem/api/latest/gen/api/signal.html> |
| NV-SYNC | NVSHMEM 3.8.0, Point-To-Point Synchronization: <https://docs.nvidia.com/nvshmem/api/latest/gen/api/sync.html> |
| NV-COLL | NVSHMEM 3.8.0, Collective Communication: <https://docs.nvidia.com/nvshmem/api/latest/gen/api/collectives.html> |
| NV-SETUP | NVSHMEM 3.8.0, Library Setup, Exit, and Query: <https://docs.nvidia.com/nvshmem/api/latest/gen/api/setup.html> |
| NV-TEAM | NVSHMEM 3.8.0, Team Management: <https://docs.nvidia.com/nvshmem/api/latest/gen/api/teams.html> |
| NV-OVER | NVSHMEM 3.8.0, API overview: <https://docs.nvidia.com/nvshmem/api/latest/api/overview.html> |
| NV-FAQ | NVSHMEM 3.8.0, FAQ: <https://docs.nvidia.com/nvshmem/api/latest/faq.html> |
| CU-MM | libcu++, Memory model: <https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/memory_model.html> |
| CU-AT | libcu++, `cuda::atomic`: <https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/synchronization_primitives/atomic.html> |
| CU-FE | libcu++, `cuda::atomic_thread_fence`: <https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/synchronization_primitives/atomic/atomic_thread_fence.html> |

The NVSHMEM pages are served under `latest` only; a path with the version
number returned 404. The pages report version 3.8.0. Their text has changed
since the research ticket #11 read them on 2026-10-20 (open question 7). The
libcu++ pages are served under `unstable`; the unversioned path returned 404.

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

### 4.2 NVSHMEM

- **E-NV1** (NV-ORD, `nvshmem_fence`). "All operations on symmetric data
  objects issued to a particular PE prior to the call to `nvshmem_fence` are
  guaranteed to be delivered before any subsequent operations on symmetric data
  objects to the same PE. `nvshmem_fence` guarantees order of delivery, not
  completion." It "can enforce memory store ordering only for the calling
  thread" and "does not have an effect on the ordering between memory accesses
  issued by the target PE".
- **E-NV2** (NV-ORD, `nvshmem_quiet`). It "ensures completion of all operations
  on symmetric data objects issued by the calling PE". It "is a local,
  non-collective operation"; "the call does not synchronize with or notify
  other PEs". "Visibility is only guaranteed at the destination PE." It covers
  the operations "posted before calling" by the calling thread.
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
  behaviour: "When NVSHMEM atomic operations and non-NVSHMEM operations (for
  example, load and store operations) are used to access the same location
  concurrently."
- **E-NV13** (NV-COLL, `nvshmem_barrier_all`, `nvshmem_barrier`). It "blocks
  the calling PE until all PEs have called `nvshmem_barrier_all`". Before that
  it "ensures completion of all previously issued memory stores and remote
  memory updates". The team form does the same over a team.
- **E-NV14** (NV-COLL, `nvshmem_sync`, `nvshmem_sync_all`). It blocks until all
  PEs of the team have called it and "only ensures completion and visibility of
  previously issued memory stores and does not ensure completion of remote
  memory updates issued through NVSHMEM routines".
- **E-NV15** (NV-MM, "Visibility Guarantees"). On systems with NVLink and
  InfiniBand the synchronization operations "only guarantee visibility of
  updates to the local PE's symmetric objects. However, on systems with only
  NVLink, these operations guarantee global visibility".
- **E-NV16** (NV-OVER, "NVSHMEM API Extensions for GPU Threads"). "The `_block`
  and `_warp` extension APIs are collective calls that must be called by every
  thread in the scope of the API and with exactly the same arguments."
  Otherwise "the call results in erroneous behavior or a deadlock".
- **E-NV17** (NV-SETUP, `nvshmem_ptr`). It "returns an address that may be used
  to directly reference dest on the specified PE ... ordinary loads and stores
  to dest may be performed". It returns a null pointer when the object cannot
  be reached by loads and stores.
- **E-NV18** (NV-SETUP, `nvshmemx_mc_ptr`). It "returns a multicast address
  corresponding to the specified symmetric address and the team", for use with
  multimem operations "exclusively". The page states no semantics for those
  operations.
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
- **E-NV21** (NV-FAQ). "For synchronization with flag, the application must use
  `nvshmem_wait_until` or `nvshmem_test` API. A plain while loop or if
  condition to check flag value is not sufficient."

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

## 5. Tables

Row key: section and primitive name. `undocumented` in the `Evidence` column
means that no source of section 1 describes the name.

### 5.1 `dl.*`

`dl` is `triton_dist.language`. Source of the names: TD-INIT.

| Primitive | Pattern | Evidence | Source | Treatment | Note |
| --- | --- | --- | --- | --- | --- |
| `wait` | b | E-TD1, E-TD2, E-TD3 | TD-SEM, TD-API | named assumption (A1) | The comparison relation ("reach") is not stated: W6 |
| `consume_token` | none | E-TD2, E-TD4 | TD-SEM, TD-API | documented | Orders the access after the wait in the same thread |
| `rank` | none | E-TD6 | TD-API, TD-PRIM | documented | |
| `num_ranks` | none | E-TD6 | TD-API, TD-PRIM | documented | |
| `symm_at` | none | E-TD5 | TD-SEM, TD-API | documented | An address on the given rank. A load or store through it is a plain Triton access |
| `notify` | b | E-TD2, E-TD3, E-TD11 | TD-SEM, TD-API, TD-T01 | named assumption (A2) | Atomicity, scope and ordering of the signal write are not stated |
| `simt_exec_region` | none | E-TD9 | TD-API | documented | |
| `extract` | none | E-TD9 | TD-API | documented | |
| `insert` | none | E-TD9 | TD-API | documented | |
| `vector` | none | E-TD9 | TD-API | documented | |
| `make_vector` | none | E-TD9 | TD-API | documented | |
| `zeros_vector` | none | E-TD9 | TD-API | documented | |
| `ld_vector` | a | E-TD8 | TD-API | named assumption (A1) | One load per element |
| `st_vector` | a | E-TD8 | TD-API | named assumption (A1) | One store per element |
| `SMEM_GENERIC_POINTER_ADDR_SPACE` | none | undocumented | TD-SMEM | weakest (W3) | A constant |
| `SharedMemoryDesc` | none | E-TD12 (docstring) | TD-SMEM | documented | A handle type |
| `allocate_smem` | none | E-TD12 (docstring) | TD-SMEM | documented | Block-local memory |
| `smem_index` | none | E-TD12 (docstring) | TD-SMEM | documented | |
| `smem_load` | a | E-TD12 (docstring) | TD-SMEM | documented | A plain load of block-local memory |
| `smem_store` | a | E-TD12 (docstring) | TD-SMEM | documented | A plain store to block-local memory |
| `smem_get_ptr` | none | E-TD12 (docstring) | TD-SMEM | documented | |
| `get_smem_shared_address_u32` | none | E-TD12 (docstring) | TD-SMEM | documented | |
| `smem_dealloc` | none | E-TD12 (docstring) | TD-SMEM | documented | |
| `tma` | none | E-TD13 (docstring) | TD-TMA | weakest (W4) | A submodule; its names are in 5.2 |
| `extern_call` | none | E-TD9 | TD-API | weakest (W6) | The effect of the callee is not stated; a direct call is not modelled |

### 5.2 `dl.tma.*`

Source of the names: `TD:python/triton_dist/language/tma/__init__.py`. No
tutorial 01–08 uses this module.

| Primitive | Pattern | Evidence | Source | Treatment | Note |
| --- | --- | --- | --- | --- | --- |
| `create_tmap_2d` | a | E-TD13 (docstring) | TD-TMA | weakest (W4) | Callees are in `tma_language.py`, not covered here |
| `tmap_update_address` | a | E-TD13 (docstring) | TD-TMA | weakest (W4) | As above |
| `PipelineState` | none | E-TD13 (docstring) | TD-TMA | documented | A value: stage index and phase |
| `TmaPipeline` | c | E-TD13 (docstring) | TD-TMA | weakest (W4) | Transfers within one GPU; callees not covered here |
| `store_and_wait` | c | E-TD13 (docstring) | TD-TMA | weakest (W4) | As above |

### 5.3 `libshmem_device.*`

Source of the names: TD-SH; of the bound NVSHMEM routine names: TD-SHC. The
front-end statement for every documented row is E-TD7. `—` in the `Bound
routine` column means that the CUDA back end has no definition.

| Primitive | Pattern | Bound routine | Evidence | Source | Treatment | Note |
| --- | --- | --- | --- | --- | --- | --- |
| `set_rocshmem_ctx` | none | — | undocumented | TD-SH | out of scope | ROCSHMEM only |
| `my_pe` | none | `nvshmem_my_pe` | E-NV19 | NV-SETUP, TD-API | documented | |
| `n_pes` | none | `nvshmem_n_pes` | E-NV19 | NV-SETUP, TD-API | documented | |
| `team_my_pe` | none | `nvshmem_team_my_pe` | E-NV19 | NV-TEAM, TD-API | documented | |
| `team_n_pes` | none | `nvshmem_team_n_pes` | E-NV19 | NV-TEAM, TD-API | documented | |
| `int_p` | c | `nvshmem_int_p` | E-NV7 | NV-RMA, TD-PRIM | documented | Complete only after `quiet` |
| `remote_ptr` | none | `nvshmem_ptr` | E-NV17, E-NV12 | NV-SETUP, TD-API | documented | An access through the address is a plain load or store |
| `remote_mc_ptr` | none | `nvshmemx_mc_ptr` | E-NV18 | NV-SETUP, TD-API | documented | The address only; the operations on it are rows of 5.4 |
| `barrier_all` | b | `nvshmem_barrier_all` | E-NV13, E-NV15 | NV-COLL, NV-MM, TD-API | documented | |
| `barrier_all_vec` | b | — | undocumented | TD-SH | out of scope | |
| `barrier_all_block` | b | `nvshmemx_barrier_all_block` | E-NV13, E-NV15, E-NV16 | NV-COLL, NV-OVER, TD-API | documented | Every thread of the block calls it |
| `barrier_all_warp` | b | `nvshmemx_barrier_all_warp` | E-NV13, E-NV15, E-NV16 | NV-COLL, NV-OVER, TD-API | documented | Every thread of the warp calls it |
| `barrier_all_wave` | b | — | undocumented | TD-SH | out of scope | |
| `barrier_all_wg` | b | — | undocumented | TD-SH | out of scope | |
| `barrier` | b | `nvshmem_barrier` | E-NV13, E-NV15 | NV-COLL, TD-API | documented | Over a team |
| `barrier_vec` | b | — | undocumented | TD-SH | out of scope | |
| `barrier_block` | b | `nvshmemx_barrier_block` | E-NV13, E-NV15, E-NV16 | NV-COLL, NV-OVER | documented | |
| `barrier_warp` | b | `nvshmemx_barrier_warp` | E-NV13, E-NV15, E-NV16 | NV-COLL, NV-OVER | documented | |
| `sync_all` | b | `nvshmem_sync_all` | E-NV14 | NV-COLL, TD-API | documented | Does not complete remote updates |
| `sync_all_block` | b | `nvshmemx_sync_all_block` | E-NV14, E-NV16 | NV-COLL, NV-OVER, TD-PRIM | documented | |
| `sync_all_warp` | b | `nvshmemx_sync_all_warp` | E-NV14, E-NV16 | NV-COLL, NV-OVER, TD-PRIM | documented | |
| `sync` | b | `nvshmem_sync` | E-NV14 | NV-COLL | documented | Defined by the CUDA back end only; not reachable as `libshmem_device.sync` |
| `team_sync_block` | b | `nvshmemx_team_sync_block` | E-NV14, E-NV16 | NV-COLL, NV-OVER | documented | |
| `team_sync_warp` | b | `nvshmemx_team_sync_warp` | E-NV14, E-NV16 | NV-COLL, NV-OVER | documented | |
| `quiet` | b | `nvshmem_quiet` | E-NV2, E-NV3 | NV-ORD, TD-API | documented | Completes the caller's operations; notifies nobody |
| `quiet_pe` | b | — | undocumented | TD-SH | out of scope | |
| `fence` | a | `nvshmem_fence` | E-NV1, E-NV3 | NV-ORD, TD-API | documented | Orders delivery per target PE; non-blocking operations are not ordered (decision D1) |
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
| `signal_op` | b | `nvshmemx_signal_op` | E-NV10, E-NV12 | NV-SIG, TD-API | documented | Atomic `set` or `add`; `set` is monotone only under the obligation of S2.4 |
| `signal_wait_until` | b | `nvshmem_signal_wait_until` | E-NV11, E-NV12, E-NV21 | NV-SYNC, NV-FAQ, TD-API | documented | The word may be written by signal operations only |
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
document describes them. `asm` in the `Note` column means that the source
defines the helper with inline assembly, which is the condition of the hard
constraint on undocumented helpers.

| Primitive | Pattern | Evidence | Source | Treatment | Note |
| --- | --- | --- | --- | --- | --- |
| `tid` | none | E-TD8 | TD-API | documented | |
| `atomic_cas` | a | E-TD8 | TD-API | named assumption (A1) | The return value is not documented; A1 fixes it |
| `atomic_add` | a | E-TD8 | TD-API | named assumption (A1) | As above |
| `__syncthreads` | b | E-TD8 | TD-API | documented | A barrier phase of one thread block |
| `ld` | a | E-TD8 | TD-API | named assumption (A1) | |
| `st` | a | E-TD8 | TD-API | named assumption (A1) | |
| `ld_vector` | a | E-TD8 | TD-API | named assumption (A1) | The same primitive as `dl.ld_vector` |
| `st_vector` | a | E-TD8 | TD-API | named assumption (A1) | The same primitive as `dl.st_vector` |
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
| `multimem_ld_reduce_v4` | a | undocumented | TD-LEC | weakest (W1) | As above. Its docstring names the instruction and states no guarantee |
| `laneid` | none | undocumented | TD-LEC | weakest (W3) | |
| `ntid` | none | undocumented | TD-LEC | weakest (W3) | |
| `red_release` | a | undocumented | TD-LEC | weakest (W2) | asm; a plain read-modify-write |
| `arrive_inc` | b | undocumented | TD-LEC | weakest (W4) | Triton source; calls `__syncthreads` and `red_release` |
| `arrive_inc_asm` | a | undocumented | TD-LEC | weakest (W2) | asm; a plain read-modify-write |
| `ld_b32` | a | undocumented | TD-LEC | weakest (W1) | |
| `ld_acquire` | a | undocumented | TD-LEC | weakest (W1) | Despite its name no acquire is assumed (open question 4) |
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
| `get_rank` | none | E-TS2 | TS-API | documented | |
| `get_num_ranks` | none | E-TS2 | TS-API | documented | |
| `put_warp` | a | E-TS1, E-TS3 | TS-API, TS-COMM | weakest (W1) | The address remapping is documented; the access is not |
| `get_warp` | a | E-TS1, E-TS3 | TS-API, TS-COMM | weakest (W1) | As above |
| `put_block` | a | E-TS1, E-TS3 | TS-API, TS-COMM | weakest (W1) | As above |
| `get_block` | a | E-TS1, E-TS3 | TS-API, TS-COMM | weakest (W1) | As above |
| `ld` | a | E-TS5 | TS-API, TS-COMM | named assumption (A1) | For `acquire` and `relaxed`. With `weak`, `volatile` or `nc=True`: W1 |
| `st` | a | E-TS5 | TS-API, TS-COMM | named assumption (A1) | For `release` and `relaxed`. With `weak` or `volatile`: W1 |
| `atom_add` | a | E-TS5 | TS-API, TS-COMM | named assumption (A1) | The return value is documented (docstring) |
| `wait_eq` | b | E-TS6 | TS-API, TS-SYNC | named assumption (A1) | For `acquire`. With `volatile`: a poll by plain loads (W1) |
| `wait_ne` | b | E-TS6 | TS-API, TS-SYNC | named assumption (A1) | As above |
| `wait_ge` | b | E-TS6 | TS-API, TS-SYNC | named assumption (A1) | As above |
| `wait_le` | b | E-TS6 | TS-API, TS-SYNC | named assumption (A1) | As above |
| `wait_gt` | b | E-TS6 | TS-API, TS-SYNC | named assumption (A1) | As above |
| `wait_lt` | b | E-TS6 | TS-API, TS-SYNC | named assumption (A1) | As above |
| `init_barrier_gpu` | b | E-TS7 | TS-API, TS-SYNC | weakest (W6) | The counter protocol and any memory ordering are not stated |
| `arrive_barrier_gpu` | b | E-TS7 | TS-API, TS-SYNC | weakest (W6) | As above |
| `wait_barrier_gpu` | b | E-TS7 | TS-API, TS-SYNC | weakest (W6) | As above |
| `sync_barrier_gpu` | b | E-TS7 | TS-API, TS-SYNC | weakest (W6) | As above |
| `barrier_blocks` | b | E-TS7, E-TS8 (docstring) | TS-API, TS-SYNC | documented | A rendezvous with a release at system scope on entry and no acquire on exit |
| `sync_blocks` | b | E-TS7, E-TS8 (docstring) | TS-API, TS-SYNC | documented | A rendezvous with no memory ordering |
| `multimem_ld_reduce` | a | E-TS9 | TS-API, TS-MM | weakest (W1) | A plain load of every replica; the order of the reduction is unspecified |
| `multimem_st` | a | E-TS9 | TS-API, TS-MM | weakest (W1) | A plain store to every replica |
| `multimem_red` | a | E-TS9 | TS-API, TS-MM | weakest (W1) | A plain read-modify-write of every replica |
| `multimem_tma_store` | c | E-TS9 | TS-API, TS-MM | weakest (W6) | The local sequence is documented; visibility at other ranks is not |
| `multimem_signal` | b | E-TS9 | TS-API, TS-MM | weakest (W2) | A plain store to every replica of the signal |
| `multimem_signal_add` | b | E-TS9 | TS-API, TS-MM | weakest (W2) | A plain read-modify-write of every replica |
| `copy` | a | E-TS3 | TS-API, TS-COPY | weakest (W1) | Row for the `src_pe` and `dst_pe` arguments; the local copy is base TileLang |
| `tma_copy` | c | E-TS3, E-TS4, E-TS10 | TS-API, TS-COPY | weakest (W6) | Completion by the user's wait is documented; visibility at the remote rank is not |
| `atomic_add` | a | E-TS11 (docstring) | TS-ATOM | weakest (W6) | Row for `dst_pe`. Atomic, but no scope is stated |
| `atomic_addx2` | a | E-TS11 (docstring) | TS-ATOM | weakest (W6) | As above |

### 5.6 TileScale, host side

Source of the names: `TS:tilelang/distributed/host.py`.

| Primitive | Pattern | Evidence | Source | Treatment | Note |
| --- | --- | --- | --- | --- | --- |
| `set_signal` | b | E-TS12 | TS-API | weakest (W6) | A stream-ordered write of a 32-bit word; atomicity and ordering are not stated |
| `wait_eq` | b | E-TS12 | TS-API | weakest (W6) | A stream-ordered wait; what it makes visible is not stated |

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
  guarantee. Its guarantee can come only from the CUDA C++ model, through A1.
  This is the finding of ticket #11.
- I3. A wait at `gpu` scope creates no synchronization edge with a writer on
  another rank, because the device scope relates only threads of one device
  (E-CU1, E-CU3, E-TD1). This is the standing decision of ticket #24.
- I4. For `dl.wait` the comparison is not stated (W6). The proof may use only
  what both readings give: after the return, each signal was read with a value
  at least `waitValue`; the wait is guaranteed to return only if each signal
  holds exactly `waitValue` from some point until the wait reads it.

**Decisions.**

- D1. `fence` does not order non-blocking operations. The sources disagree
  (E-NV3); the weaker reading is taken, as ticket #11 decided.
- D2. Barrier and quiet visibility is taken as global only on a system with
  NVLink alone (E-NV15); otherwise it covers only the local PE's objects.
  Whether the two-GPU node of the project is such a system was not checked
  here.
- D3. Where E-TD10 contradicts E-TD1, the definition E-TD1 governs. The example
  and best practice 5 of TD-SEM are not guarantees (open question 3).
- D4. A `_block` or `_warp` routine is one event of the whole block or warp. A
  kernel in which not every thread calls it with the same arguments is
  rejected (E-NV16).

## 7. Named assumptions

Each entry has the fields of the registry of ticket S5.3 as the map states
them: name, statement, source or rationale, affected kernels. The class of
ticket S5.1 is added. The names are proposals; the freeze ticket S5.1 fixes
them. Affected kernels were found by text search at the pins; reachability
from each kernel entry point was not traced.

### A1 `scope_order_vocabulary`

| Field | Value |
| --- | --- |
| Name | `scope_order_vocabulary` |
| Class | semantic source |
| Statement | In kernel code of Triton-distributed and TileScale the scope strings `cta`, `gpu` and `sys` denote `cuda::thread_scope_block`, `cuda::thread_scope_device` and `cuda::thread_scope_system`, and the semantic strings `relaxed`, `acquire`, `release` and `acq_rel` denote the memory order of the same name. A primitive that takes a scope and a semantic performs, for each element, an atomic operation of the CUDA C++ memory model at that scope and order: `ld` a load; `st` a store; `atomic_add` and `atom_add` a fetch-add that returns the previous value; `atomic_cas` a strong compare-exchange that returns the value read; a wait a sequence of loads that ends with a load satisfying its condition. Nothing is assumed for any other string (`weak`, `volatile`, `sc`, the empty string) |
| Source or rationale | TD-SEM defines `gpu`, `sys`, `acquire` and `release` in prose (E-TD1) that matches the scope relations and the synchronization rule of CU-MM (E-CU1, E-CU3). TD-API and TS-API list the accepted strings without defining them (E-TD8, E-TS5, E-TS6). No document states the identification, the values `cta`, `relaxed` and `acq_rel`, or the return values of `atomic_add` and `atomic_cas`. The map names the CUDA C++ memory model as the source of semantics for scoped accesses |
| Affected kernels | Triton-distributed: tutorial 01 and tutorial 07 (`dl.wait`); the helper modules `kernels/nvidia/common_ops.py` (`atomic_cas`, `atomic_add`, `ld`, `st`), `kernels/nvidia/reduce_scatter.py` (`ld`, `st`, `atomic_add`) and `kernels/nvidia/allgather_gemm.py` (`dl.wait`, `st`), which tutorials 05 to 08 import. TileScale: `examples/distributed/allgather_gemm/example_allgather_gemm_overlapped.py`, `example_allgather_gemm_specialized.py` and `gemm_allreduce/example_gemm_ar_specialized.py` (`T.wait_eq`, `T.wait_ge`); `gemm_reducescatter/example_gemm_rs_overlapped.py` (`T.st`, `T.atom_add`) |

### A2 `notify_atomic_release`

| Field | Value |
| --- | --- |
| Name | `notify_atomic_release` |
| Class | semantic source |
| Statement | `dl.notify(ptr, rank, signal, sig_op, comm_scope)` performs one atomic update of the 64-bit signal word at `ptr` on rank `rank`, with release ordering, by the calling thread: a store of `signal` for `sig_op="set"`, a fetch-add of `signal` for `sig_op="add"`. The scope is the system scope for `comm_scope="intra_node"` and the device scope for `comm_scope="gpu"`. Nothing is assumed for `comm_scope="inter_node"`, and nothing about stores issued by other threads |
| Source or rationale | The documentation says only "Send a signal to a remote rank" with `sig_op` `"set"` or `"add"` (E-TD2, E-TD3). It states no atomicity, scope or ordering, and it does not list the value `gpu`, which tutorial 08 passes. The examples of TD-SEM and TD-T01 pair `notify` with `wait(..., "acquire")` as the only synchronization between producer and consumer (E-TD10, E-TD11). Without the assumption `notify` falls under W2, and each `notify` races with the matching `wait` |
| Affected kernels | Tutorial 01 (two calls, `intra_node`); tutorial 08 and `kernels/nvidia/gemm_reduce_scatter.py` (`comm_scope="gpu"`, own rank) |

## 8. Counts

| Section | Rows | documented | named assumption | weakest | out of scope |
| --- | --- | --- | --- | --- | --- |
| 5.1 `dl.*` | 25 | 18 | 4 | 3 | 0 |
| 5.2 `dl.tma.*` | 5 | 1 | 0 | 4 | 0 |
| 5.3 `libshmem_device.*` | 86 | 49 | 0 | 12 | 25 |
| 5.4 `language_extra` | 53 | 7 | 6 | 40 | 0 |
| 5.5 TileScale, kernel side | 31 | 4 | 9 | 18 | 0 |
| 5.6 TileScale, host side | 2 | 0 | 0 | 2 | 0 |
| Total | 202 | 79 | 19 | 79 | 25 |

By pattern: a 49, b 43, c 60, none 50.

## 9. Open questions for the user

1. **A1 as an assumption or as the baseline.** Every theorem about a scoped
   access would print A1. Should the reading of the scope and semantic strings
   be a registered assumption, as proposed, or part of the trusted statement of
   the semantics?
2. **A2.** Is the assumption on `dl.notify` accepted? The alternative is W2:
   tutorials 01 and 08 then cannot be proved as written, and only a variant
   that uses `signal_op` with `signal_wait_until` rests on documented
   guarantees alone.
3. **Contradiction in the Triton-distributed documentation.** Its example and
   its best practice 5 recommend `"gpu"` scope across the GPUs of one node,
   against its own definition of `"gpu"` (E-TD10, E-TD1). Decision D3 follows
   ticket #24. Should the contradiction be reported upstream?
4. **Undocumented CUDA helpers in kernels outside the tutorials.** Under
   `kernels/nvidia/` the search finds `ld_acquire` 19 times, `ld_b32` 11,
   `membar` 11, `multimem_st_b64` 9, `load_v4_u32` 6, `atomic_add_per_warp` 5
   and `multimem_ld_reduce_v4` 4. With W1 and W2 a kernel that shares such data
   between agents fails. If a benchmark kernel (DeepEP, all-to-all MoE) needs
   one of them, a further named assumption is required; none is proposed here.
5. **Docstrings as documentation.** The rows of `smem_*`, `dl.tma`,
   `barrier_blocks`, `sync_blocks` and the TileScale `atomic_add` rest on
   docstrings in the pinned source. Do docstrings count as front-end
   documentation?
6. **Extent of the table.** Not covered: `extra/cuda/tma_language.py` (40
   functions), the host-side functions of `triton_dist.utils` (for example
   `nvshmem_barrier_all_on_stream`), the helpers `set_signal` and `wait_eq`
   that tutorial 07 imports from `triton_dist.kernels.common_ops`, base Triton
   `tl.atomic_add(sem=, scope=)` (tutorial 08) and the base TileLang TMA
   operations. Should any of them get rows?
7. **NVSHMEM version.** The documentation is served only as `latest` (3.8.0
   on 2026-11-11) and changed after ticket #11 read it. Should the project pin
   an NVSHMEM version, and is 3.8.0 the version on the GPU node?
8. **TileScale gaps and the label `out of scope`.** `volatile` and `weak`
   semantics, the multimem signals and `T.atomic_add(dst_pe=)` have no
   documented ordering, so the examples that use them
   (`example_allgather_gemm_specialized.py`, `example_gemm_ar_specialized.py`,
   `example_gemm_rs_specialized.py`) cannot be proved under the weakest
   semantics. Are named assumptions wanted for them? And is `out of scope`
   acceptable as a fourth label for the 25 names without a CUDA definition?
