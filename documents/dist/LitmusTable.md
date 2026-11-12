# Litmus table from documentation examples

Status: delivered for ticket S8.1 (#71) on 2026-11-12, not yet reviewed by the
user. Nothing here is a proved result, and nothing was executed: every source
was read as text. The ticket carries the `sign-off` label; section 9 lists the
rows whose expected outcome is least certain.

A row is a small program, a few lines per agent, taken from an example or a
normative sentence of documentation, together with the set of outcomes that
the documentation allows or forbids. The rows are the test bed of the event
model: ticket S8.2 runs each row on the model and ticket C-I5 on the GPU, and
both check that what is observed is among what is allowed. A row is therefore
only as good as its source. Its outcome set follows from sentences that are
quoted here, not from what an implementation does and not from general
knowledge of weak memory.

Vocabulary: [PrimitiveSources.md](PrimitiveSources.md) for the primitives,
their patterns, the evidence items `E-…`, the rules W1 to W6, the decisions D1
to D7, the inferences I1 to I6 and the assumptions A1 and A2;
[`EventSpec.lean`](../../VeriTile/Staging/Dist/EventSpec.lean) for `Event`,
`Mode`, `Counter` and the pattern lemmas;
[EventControlFlow.md](EventControlFlow.md) for the kernels K1 to K28.

## 1. Rules

**Sources** (hard constraints of the map, issue #1). Front-end documentation
of Triton-distributed and TileScale, including docstrings and comments of the
tutorials at the pinned commits; the CUDA C++ memory model (libcu++); the
NVSHMEM API documentation. Nothing at or below PTX is cited, no PTX litmus
test is reused, and the branch `gpu-validation` was not opened. A source file
of a front end is used only as section 1 of `PrimitiveSources.md` allows:
that a name exists with its parameters, and the docstrings and comments it
carries.

**Version of record.** NVSHMEM 3.6.5. Its frozen manual is at
<https://archive.docs.nvidia.com/nvshmem/api/3.6.5/>, listed by
<https://docs.nvidia.com/nvshmem/api/versions1.json>. Every NVSHMEM sentence
quoted in this document was read there on 2026-11-12. `PrimitiveSources.md`
quotes the same routines from `latest` (3.8.0) and says that the text of 3.6.5
could not be obtained; section 8 lists where the two differ.

**Marks.** Each row has one:

| Mark | Meaning |
| --- | --- |
| Q | Decided by the quoted sentences alone |
| R(x) | Decided only under x, a reading, decision or assumption of `PrimitiveSources.md`. D5 is a reading that the project counts as documented; A1 and A2 are registered assumptions; D1, D3 and D6 are decisions between conflicting or incomplete sources |
| N | Not decided by documentation. The row says what is missing and takes the weakest treatment (W1 to W6) |
| D2 | The two readings of decision D2 give different outcome sets. Both are given and none is chosen |

**Outcomes.** "Allowed" is the set of final values that the documentation
permits; every other value is forbidden. "Undefined behaviour" means that the
documentation gives the program no meaning (a data race of the CUDA C++ model,
E-CU2, or a conflict of the NVSHMEM model, L-NV3). In this project's model
that is an execution failure (ticket #14). For such a row the model must
report the failure; a GPU run of it shows nothing, because no observed value
is forbidden.

**Agents.** As in `EventControlFlow.md`: a CTA where the source has no thread
index, a thread where it has one, and the host of a rank for host code.
Whether a statement of a CTA is executed by one or by all of its threads is
part of the lowering, which is trust base, and of the map from primitives to
events (ticket S2.6). The rows whose outcome depends on it say so (LIT-a08,
LIT-b03).

**Condition P-ATOM.** Several rows use atomic operations at system scope on
memory of another GPU. The CUDA C++ model makes such an operation atomic only
on a platform condition (L-CU5): the native-atomic attribute of the
peer-to-peer link between each accessing GPU and the GPU that holds the
object. Where a row says "if P-ATOM", its outcome holds when the attribute is
1 for the two devices; otherwise the operations are not atomic and the row is
a data race. No document of the project registers this condition (section 8).

**Events.** The last field of a row names the constructors of `EventSpec.lean`
that the row exercises. The map from primitives to events is fixed by ticket
S2.6, not here.

## 2. Sources

The IDs of section 3 of `PrimitiveSources.md` are used as they are (`TD-SEM`,
`TD-API`, `TD-T01`, `TS-API`, `CU-MM`, …). The table adds these. Paths with
prefix `TD:` are in Triton-distributed at
`7908e4ea9010bb2238f9f05b28c904fd4b71c60a`, with prefix `TS:` in TileScale at
`71e71457d40ed6e6a2dad297936bfdbff21d104b`, with prefix `TS9:` in TileScale at
`9ba8a605d347f600a95d0275e2639cb00122c084`. Every URL was accessed on
2026-11-12.

| ID | Source |
| --- | --- |
| NV-VER | Index of NVSHMEM manual versions: <https://docs.nvidia.com/nvshmem/api/versions1.json> |
| NV365-ORD | NVSHMEM 3.6.5, Memory Ordering: <https://archive.docs.nvidia.com/nvshmem/api/3.6.5/gen/api/ordering.html> |
| NV365-MM | NVSHMEM 3.6.5, Memory Model: <https://archive.docs.nvidia.com/nvshmem/api/3.6.5/gen/mem-model.html> |
| NV365-RMA | NVSHMEM 3.6.5, Remote Memory Access: <https://archive.docs.nvidia.com/nvshmem/api/3.6.5/gen/api/rma.html> |
| NV365-SIG | NVSHMEM 3.6.5, Signaling Operations: <https://archive.docs.nvidia.com/nvshmem/api/3.6.5/gen/api/signal.html> |
| NV365-SYNC | NVSHMEM 3.6.5, Point-To-Point Synchronization: <https://archive.docs.nvidia.com/nvshmem/api/3.6.5/gen/api/sync.html> |
| NV365-COLL | NVSHMEM 3.6.5, Collective Communication: <https://archive.docs.nvidia.com/nvshmem/api/3.6.5/gen/api/collectives.html> |
| NV365-OVER | NVSHMEM 3.6.5, Overview of the APIs: <https://archive.docs.nvidia.com/nvshmem/api/3.6.5/api/overview.html> |
| NV365-USE | NVSHMEM 3.6.5, Using NVSHMEM: <https://archive.docs.nvidia.com/nvshmem/api/3.6.5/using.html> |
| NV365-FAQ | NVSHMEM 3.6.5, Troubleshooting And FAQs: <https://archive.docs.nvidia.com/nvshmem/api/3.6.5/faq.html> |
| TD-T03 | `TD:tutorials/03-inter-node-allgather.py` |
| TD-T04 | `TD:tutorials/04-deepseek-infer-all2all.py` |
| TD-CO | `TD:python/triton_dist/kernels/nvidia/common_ops.py` (names, signatures, one docstring) |
| TS-BUILTIN | `TS:tilelang/language/builtin.py` (docstrings) |
| TS-INSTR | `TS:docs/programming_guides/instructions.md` |
| TS9-DEEPEP | `TS9:examples/distributed/deepseek_deepep/deepep.md` |

`CU-MM`, `CU-AT` and `CU-FE` were read again on 2026-11-12 at the URLs of
`PrimitiveSources.md` (the `unstable` pages of libcu++), together with the
page of `cuda::barrier`
(<https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/synchronization_primitives/barrier.html>).
The last paragraph of CU-MM points to a set of litmus tests below the
front-end language; that link was not followed.

## 3. Evidence added by this table

An item of `PrimitiveSources.md` is cited by its ID and not repeated. The
items below are passages that it does not hold. Each is a quotation.

### 3.1 NVSHMEM 3.6.5

- **L-NV1** (NV365-ORD, notes of `nvshmem_fence`). "The nvshmem_fence routine
  can enforce memory store ordering only for the calling thread. Thus, to
  ensure ordering for memory stores performed by a thread that is not the
  thread calling nvshmem_fence, the update must be made visible to the calling
  thread according to the rules of the memory model associated with the
  threading environment." E-NV1 has the first sentence. The page of 3.6.5
  has no note on GPU threads: the passage E-NV22 is not in it.
- **L-NV2** (NV365-ORD, `nvshmem_quiet`). "On systems with only NVLink, all
  operations on symmetric data objects are guaranteed to be complete and
  visible to all PEs when nvshmem_quiet returns. On systems with both NVLink
  and InfiniBand, visibility is only guaranteed at the destination PE." And:
  "A call to nvshmem_quiet by a thread completes the operations posted prior
  to calling nvshmem_quiet."
- **L-NV3** (NV365-USE, "Data Consistency"). "Two operations that were issued
  by the same PE or different PEs, and that access the same memory location in
  parallel, are in conflict when one or more of the operations performs a
  write. These conflicts result in undefined behavior in the OpenSHMEM memory
  model. An exception is made when the operations are a combination of AMOs or
  AMOs and wait/test operations. A second exception is made when the
  operations are a combination of signal updates and wait/test operations."
- **L-NV4** (NV365-USE, "Data Consistency"). "Updates to globally accessible
  objects are unordered. A PE can enforce the ordering of its updates with
  respect to accesses performed by the target PE using the nvshmem_fence
  operation. When updates performed by a PE must be ordered or made visible to
  PEs other than the target PE, use the nvshmem_quiet operation."
- **L-NV5** (NV365-USE, "Data Consistency"). Among the cases in which "two
  memory accesses by the same PE are guaranteed to occur in order": "The first
  access is a wait or test call that is followed by a read operation and both
  operations target local symmetric memory." And two accesses that "are
  separated by an appropriate ordering operation" according to a table: after
  a blocking first access, "Fence/quiet/barrier" for the same target PE and
  "Quiet/barrier" for a different target PE; after a non-blocking first
  access, "Quiet/barrier" for both.
- **L-NV6** (NV365-USE, "Data Consistency", on `nvshmem_quiet`). "Access to
  all PEs (for example, to any location in the PGAS) are ordered, so accesses
  that occurred before the quiet operation can be observed by all PEs as
  having occurred before accesses after the quiet operation. PEs must use
  appropriate synchronization operations, for example, wait/test operations,
  to observe the ordering enforced at the PE that performed the quiet
  operation." On `nvshmem_fence`: "accesses that occurred before the fence
  operation can be observed by the PE that is local to the corresponding
  partition of the PGAS as having occurred before accesses after the fence
  operation."
- **L-NV7** (NV365-SIG, introduction). The example of the page is the
  sequence `nvshmem_putmem(dest, src, size, (pe+1) % npes);`
  `nvshmem_quiet();` `nvshmemx_int_signal(flag, 1, (pe+1) % npes);`
  `nvshmem_int_wait_until(flag, NVSHMEM_CMP_EQ, 1);`, introduced by: "The
  following example shows signal operations used to implement neighbor
  communication in a ring."
- **L-NV8** (NV365-SIG, notes of `nvshmem_put_signal`). "For example, the
  completion of the signal update in a sequence consisting of a put routine
  followed by a put-with-signal routine does not imply delivery of the put
  routine's data." E-NV8 has the two sentences before it.
- **L-NV9** (NV365-SIG, "Available Signal Operators"; NV365-SYNC,
  `nvshmem_signal_wait_until`). "An update to signal data object is an atomic
  add operation." The wait returns "the contents of the signal data object,
  sig_addr, at the calling PE that satisfies the wait condition."
- **L-NV10** (NV365-MM, "Ordering of Operations"). "Non-blocking operations
  are not ordered by calls to nvshmem_fence. Instead, they must be completed
  using the nvshmem_quiet operation." E-NV3 has the first sentence.
- **L-NV11** (NV365-USE, "Data Consistency"). "While updates are unordered,
  updates made by using NVSHMEM APIs are guaranteed to eventually complete
  without any additional actions performed by the source or the target PE."
- **L-NV12** (NV365-COLL, `nvshmem_barrier`). "nvshmem_barrier ensures that
  all previously issued stores and remote memory updates, including AMOs and
  RMA operations, done by any of the PEs in the active set are complete before
  returning. On systems with only NVLink, updates are globally visible,
  whereas on systems with both NVLink and InfiniBand, NVSHMEM only guarantees
  that updates to the memory of a given PE are visible to that PE."
- **L-NV13** (NV365-USE, "Communication Model"). "CUDA kernels that use
  synchronization or collective APIs must be launched by using the collective
  launch API to guarantee deadlock-free progress and completion."
- **L-NV14** (NV365-FAQ). "For synchronization with flag, the application must
  use nvshmem_wait_until or nvshmem_test API. A plain while loop or if
  condition to check flag value is not sufficient. NVSHMEM needs to perform
  consistency operation to ensure that the data is visible to the GPU after
  synchronization using flag value." E-NV21 has the first two sentences.
- **L-NV15** (NV365-MM, "Differences Between NVSHMEM and OpenSHMEM"). The page
  has a litmus example of its own with "Allowed output: i = 1, j = 0", and one
  with atomic fetch-and-add of which it says "NVSHMEM does not guarantee" the
  order of the two results. Both use atomic memory operations of NVSHMEM;
  section 7 says why they have no row.

### 3.2 CUDA C++ memory model

- **L-CU1** (CU-MM, "Example: Message Passing", first listing). "The following
  example passes a message stored to the x variable by a thread in block 0 to
  a thread in block 1 via the flag f". Thread 0 of block 0 executes `x = 42;`
  and then `flag.store(1, memory_order_release);`; thread 0 of block 1
  executes `while(flag.load(memory_order_acquire) != 1);` and then
  `assert(x == 42);`. Both use `cuda::atomic_ref<int,
  cuda::thread_scope_device> flag(f);`.
- **L-CU2** (CU-MM, same section, second listing). The store uses
  `cuda::thread_scope_block` and the load `cuda::thread_scope_device`. "In the
  following variation of the previous example, two threads concurrently access
  the f object without synchronization, which leads to a data race, and
  exhibits undefined behavior". The reason given: the loading thread "is not
  in a scope included by the store operation performed in Block 0".
- **L-CU3** (CU-FE, "Example"). Block 0 executes `*data = 42;`, a fence with
  `cuda::memory_order_release` and `cuda::thread_scope_device`, and
  `flag->test_and_set(cuda::std::memory_order_relaxed);`. The other block
  waits on the flag, executes a fence with `cuda::memory_order_acquire` and
  `cuda::thread_scope_device`, and prints `*data` with the comment `// Prints
  42`. A comment says: "an atomic operation is required to set up the
  synchronization".
- **L-CU4** (CU-AT, "Concurrency Restrictions"). An object of type
  `cuda::atomic` "shall not be accessed concurrently by CPU and GPU threads
  unless" it is in unified memory or in CPU memory, each with a property of
  the platform.
- **L-CU5** (CU-MM, "Atomicity"). "An atomic operation is atomic at the scope
  it specifies if" either "it specifies a scope other than
  thread_scope_system", or the scope is the system scope and one of several
  conditions holds. For an object in GPU memory that only GPU threads access,
  the condition is that the attribute `cudaDevP2PAttrNativeAtomicSupported`
  is 1 between each accessing device and the device where the object resides,
  "or only GPU threads from a single GPU concurrently access it."

### 3.3 Triton-distributed

- **L-TD1** (TD-SEM, "Token-Based Data Dependency"). The listing has the
  comments "Without consume_token: compiler might reorder load before wait"
  and, on the load, "BUG: might execute before wait!". E-TD4 has the comment
  of the corrected listing.
- **L-TD2** (TD-T03, comment at lines 82 to 84). "libshmem_device.putmem_signal_block
  calls NVSHMEM API nvshmemx_putmem_signal_block".
- **L-TD3** (TD-T04, comments at lines 260 and 263). "4. fence data transfer.
  Then wait for signal". "notify the target rank (here is the pid-th rank) that
  the data is ready by setting the signal".
- **L-TD4** (TD-T03, comment at line 77). "wait for all peers done".
- **L-TD5** (TD-CO, docstring of `barrier_all_intra_node_atomic_cas_block`,
  lines 155 to 157). "NOTE: this function should only be called with atomic
  support. memory over PCI-e does not support atomic r/w. DON'T use this
  function on such platforms."
- **L-TD6** (TD-T01, comment at lines 165 to 166). "You need a
  barrier all to make sure the above initialization is visible to all the
  other ranks."

### 3.4 TileScale

- **L-TS1** (TS9-DEEPEP, "Implementation Notes", line 190). "Queue semantics:
  senders update channel_tail_idx with release semantics; receivers poll
  heads/tails with acquire/volatile loads to ensure visibility across PEs."
- **L-TS2** (TS9-DEEPEP, line 189). "Negative offset encoding: senders write
  channel start/end offsets as -value-1 so that a zero token count is
  distinguishable from an uninitialized 0."
- **L-TS3** (TS-BUILTIN, docstrings; TS-INSTR, line 191). `sync_threads`:
  "Synchronize all threads in a block." In TS-INSTR: "Block-wide barrier".
  `fence_gpu`: "Issue a memory fence at GPU (device) scope." `fence_sys`:
  "Issue a memory fence at system scope (visible across GPUs in a node)."
  None of them states a memory order.
- **L-TS4** (TS-BUILTIN, docstring of `tma_store_wait`). "Waits until the
  number of outstanding TMA store groups is at most count."

## 4. Summary

32 rows: 10 of pattern (a), 13 of pattern (b), 9 of pattern (c). The files are
in [`tests/dist_litmus/`](../../tests/dist_litmus/README.md). "TD" is
Triton-distributed, "TS" is TileScale. "UB" is undefined behaviour.

| Row | Example | Front end | Expected | Mark |
| --- | --- | --- | --- | --- |
| LIT-a01 | Message passing through a flag at device scope | TD | 42 | R(D5) |
| LIT-a02 | The flag is stored at block scope | TD | UB | R(A1 a) |
| LIT-a03 | Message passing between two ranks at system scope | TD | 42 if P-ATOM | R(D5) |
| LIT-a04 | Between two ranks with the flag at device scope | TD | UB | R(D5) |
| LIT-a05 | Counter of finished tiles | TD | exactly one signal | Q |
| LIT-a06 | Handshake by compare-and-swap | TD | the peer's value if P-ATOM | R(A1 b) |
| LIT-a07 | One lane writes a slot and publishes the tail | TS | 42 if P-ATOM | R(A1 c) |
| LIT-a08 | A warp copies, the block synchronizes, another lane publishes | TS | UB | N |
| LIT-a09 | Load with `nc=True` of a word a peer writes | TS | UB | Q |
| LIT-a10 | Message passing through a relaxed flag and two fences | TS | UB | N |
| LIT-b01 | Store, notify, wait at system scope | TD | 42 if P-ATOM | R(A2) |
| LIT-b02 | The same with the wait at device scope | TD | UB | R(D5) |
| LIT-b03 | Stores by all threads, `__syncthreads`, `fence`, notify | TD | UB | N |
| LIT-b04 | Load after a wait whose token is not consumed | TD | UB | Q |
| LIT-b05 | Two ranks add to a signal, a third waits for the sum | TD | 2 | Q |
| LIT-b06 | The receiver polls the flag with an ordinary load | TD | UB | Q |
| LIT-b07 | `dl.wait` on a word that a put-with-signal writes | TD | UB | R(D5) |
| LIT-b08 | Host `set_signal` while a kernel waits with `dl.wait` | TD | UB | N |
| LIT-b09 | Host `set_signal` while a kernel waits with `T.wait_eq` | TS | UB | N |
| LIT-b10 | One thread per peer waits, then `__syncthreads` | TD | all signals seen | Q |
| LIT-b11 | Store before `T.barrier_blocks`, load by the peer after it | TS | UB | Q |
| LIT-b12 | Relaxed store, `T.wait_ne`, volatile load | TS | UB | N |
| LIT-b13 | Store before `barrier_all_block`, loads by two other ranks | TD | 42 on the target; third rank by reading | D2 |
| LIT-c01 | Put-with-signal, wait, read | TD | payload | Q |
| LIT-c02 | Put, then put-with-signal; the first put is read | TD | UB | Q |
| LIT-c03 | Put, `fence`, put-with-signal; the first put is read | TD | payload | Q |
| LIT-c04 | Non-blocking put and signal without `quiet` | TD | UB | Q |
| LIT-c05 | Non-blocking put, `fence`, signal | TD | UB | R(D1) |
| LIT-c06 | Non-blocking put, `quiet`, signal | TD | payload | Q |
| LIT-c07 | Non-blocking get, `quiet`, load | TD | the remote value | Q |
| LIT-c08 | Put to rank 1, `quiet`, signal to rank 2, which reads rank 1 | TD | by reading | D2 |
| LIT-c09 | Remote TMA store, store wait, flag | TS | UB | N |

Counts by mark: Q 13, R 10, N 7, D2 2 (one of them, LIT-b13, is Q for the
target rank).

## 5. Rows

Every row has the same fields. "Decided by" quotes the sentences or names the
item that holds the quotation.

### 5.1 Pattern (a): scoped access and atomics

#### LIT-a01. Message passing through a flag at device scope

- **File.** `a01_message_passing_device.py`, Triton-distributed.
- **Example.** L-CU1.
- **Agents and steps.** Thread 0 of CTA 0 and thread 0 of CTA 1 of one rank.
  The first stores 42 to `x` with an ordinary store, then stores 1 to `flag`
  with `st`, scope `gpu`, semantic `release`. The second loads `flag` with
  `ld`, scope `gpu`, semantic `acquire`, until it reads 1, then loads `x` with
  an ordinary load and stores the value to `out`.
- **Asked.** The value of `out`.
- **Expected.** Allowed: 42. Forbidden: 0, the initial value of `x`.
- **Decided by.** L-CU1: the example "passes a message stored to the x
  variable by a thread in block 0 to a thread in block 1 via the flag f", and
  its last statement is `assert(x == 42);`. E-CU1: the two threads are in "the
  same CUDA device".
- **Mark.** R(D5). The example is CUDA C++. That `st` and `ld` with the
  strings `gpu`, `release` and `acquire` are these operations is the reading
  D5.
- **Events.** `access store plain`; `access store (atomic gpu release)`;
  `access load (atomic gpu acquire)`, repeated; `access load plain`.

#### LIT-a02. The flag is stored at block scope

- **File.** `a02_message_passing_block_scope_store.py`, Triton-distributed.
- **Example.** L-CU2.
- **Agents and steps.** As LIT-a01, but the store of `flag` has scope `cta`.
- **Asked.** Whether the program has a defined outcome.
- **Expected.** Undefined behaviour.
- **Decided by.** L-CU2: "two threads concurrently access the f object without
  synchronization, which leads to a data race, and exhibits undefined
  behavior"; the loading thread "is not in a scope included by the store
  operation performed in Block 0".
- **Mark.** R(A1, clause a), for the string `cta`. Without A1 the store is an
  ordinary one (W1) and the row is a data race by I1: the outcome is the same.
- **Events.** `access store (atomic cta release)`; `access load (atomic gpu
  acquire)`. `Mode.AtomicFor` fails, so the two accesses are a `Conflict`.

#### LIT-a03. Message passing between two ranks at system scope

- **File.** `a03_message_passing_system.py`, Triton-distributed.
- **Example.** L-CU1 with the scope of E-CU1; the store and the load of the
  flag are those of K16 (`TD-CO`, lines 177 and 178).
- **Agents and steps.** Thread 0 of a CTA on rank 0 and thread 0 of a CTA on
  rank 1. The first stores 42 to `x` of rank 1 through `dl.symm_at` with an
  ordinary store, then stores 1 to `flag` of rank 1 with `st`, scope `sys`,
  semantic `release`. The second loads its own `flag` with `ld`, scope `sys`,
  semantic `acquire`, until it reads 1, then loads its own `x`.
- **Asked.** The value of `out` on rank 1.
- **Expected.** If P-ATOM: allowed 42, forbidden 0. Otherwise undefined
  behaviour.
- **Decided by.** E-CU1: "Each thread in the system is related to each other
  thread in the system by the system thread scope". E-TD1: `"sys"` is
  "Synchronization visible across the entire system (multiple GPUs/nodes)".
  L-CU1 for the pattern, L-CU5 for the condition, E-TD5 for the access
  through `dl.symm_at`.
- **Mark.** R(D5).
- **Events.** `access store plain` to memory of rank 1 by an agent of rank 0;
  `access store (atomic sys release)`; `access load (atomic sys acquire)`,
  repeated; `access load plain`.

#### LIT-a04. Between two ranks with the flag at device scope

- **File.** `a04_message_passing_two_ranks_device_scope.py`,
  Triton-distributed.
- **Example.** E-TD1 against E-CU1.
- **Agents and steps.** As LIT-a03 with scope `gpu` in both `st` and `ld`.
- **Asked.** Whether the program has a defined outcome.
- **Expected.** Undefined behaviour.
- **Decided by.** E-TD1: `"gpu"` is "Synchronization visible only within the
  same GPU". E-CU1: the device scope relates GPU threads "in the same CUDA
  device". E-CU2: a data race needs only that one of the two actions "is not
  atomic at a scope that includes the thread that performed the other
  operation". This is inference I3.
- **Mark.** R(D5). The example and best practice 5 of TD-SEM say the opposite
  (E-TD10); decision D3 lets the definition govern.
- **Events.** `access store (atomic gpu release)` by an agent of rank 0 on a
  word of rank 1; `access load (atomic gpu acquire)` by rank 1. Lemma
  `not_syncEdge_of_acquire_gpu`; the two accesses are a `Conflict`.

#### LIT-a05. Counter of finished tiles

- **File.** `a05_atomic_counter_last_tile.py`, Triton-distributed.
- **Example.** E-TD18 (tutorial 08, K19).
- **Agents and steps.** CTA 0 and CTA 1 of one rank. Each executes `val =
  tl.atomic_add(counter, 1, sem="release", scope="gpu")`, stores `val`, and
  calls `dl.notify(barrier, rank, signal=1, comm_scope="gpu")` if `val` is 1.
- **Asked.** The two values of `val`, and how many CTAs call `dl.notify`.
- **Expected.** Allowed: the values are 0 and 1 in either order, the counter
  ends at 2, and exactly one CTA calls `dl.notify`. Forbidden: two equal
  values, no call, two calls.
- **Decided by.** E-TR1: "Performs an atomic add at the memory location
  specified by pointer." and "Return the data stored at pointer before the
  atomic operation." The comment of the tutorial (E-TD18): "it means the
  current tile is the last one to be completed for this rank".
- **Mark.** Q. The strings `release` and `gpu` do not enter the question.
  What a reader of `barrier` may conclude after the signal needs A1, clause d,
  and A2; no row asks it.
- **Events.** `access update (atomic gpu release)` twice; `bump` of a `word`
  with `set`, under a branch on the value that the update returned (the class
  B-sched of `EventControlFlow.md`).

#### LIT-a06. Handshake by compare-and-swap

- **File.** `a06_cas_handshake.py`, Triton-distributed.
- **Example.** K11, the kernel that tutorial 05 launches (`TD-CO`, lines 154
  to 168), with its docstring L-TD5.
- **Agents and steps.** Thread 0 of a CTA on rank 0 and on rank 1; the
  program is symmetric. Rank `r` with peer `p` stores `42 + r` to `x` of the
  peer with an ordinary store; repeats `atomic_cas(flag[r] of the peer, 0, 1,
  "sys", "release")` until it returns 0; repeats `atomic_cas(own flag[p], 1,
  0, "sys", "acquire")` until it returns 1; loads its own `x`.
- **Asked.** The value of `out` on each rank.
- **Expected.** Under A1, clause b, and if P-ATOM: allowed `42 + p` on rank
  `r`, forbidden 0. Without clause b the returned value is unspecified (W3),
  the second loop may end at any time, and the load can race with the store
  of the peer: undefined behaviour.
- **Decided by.** E-TD8: "Atomic compare-and-swap", with no description of the
  returned value. L-TD5 names the condition P-ATOM in the words of the front
  end. L-CU1 for the pattern.
- **Mark.** R(A1, clause b); D5 for the strings.
- **Events.** `access store plain`; `access update (atomic sys release)`,
  repeated; `access update (atomic sys acquire)`, repeated; `access load
  plain`. The two loops are of class L-sched.

#### LIT-a07. One lane writes a slot and publishes the tail

- **File.** `a07_queue_tail_release_acquire.py`, TileScale.
- **Example.** L-TS1.
- **Agents and steps.** Lane 0 of a block on rank 0 and lane 0 of a block on
  rank 1. The first executes `T.st(data[0], 42, dst_pe=1)`, an ordinary store
  to the peer because the default semantic is `weak`, then `T.st(tail[0], 1,
  scope="sys", sem="release", dst_pe=1)`. The second executes
  `T.wait_ge(tail[0], 1)`, whose defaults are scope `sys` and semantics
  `acquire` (E-TS6), then loads `data[0]` with `T.ld` and its defaults.
- **Asked.** The value of `out` on rank 1.
- **Expected.** Under A1, clause c, and if P-ATOM: allowed 42, forbidden 0.
  Without A1 every access is an ordinary one (W1) and the row is a data race.
- **Decided by.** L-TS1: "senders update channel_tail_idx with release
  semantics; receivers poll heads/tails with acquire/volatile loads to ensure
  visibility across PEs." E-TS13: the strings are listed and not defined.
  L-CU1 for the pattern.
- **Mark.** R(A1, clause c).
- **Events.** `access store plain` to memory of rank 1; `access store (atomic
  sys release)`; `wait` on a `word` with `ge`, mode `atomic sys acquire`;
  `access load plain`.

#### LIT-a08. A warp copies, the block synchronizes, another lane publishes

- **File.** `a08_warp_copy_barrier_release_by_one_lane.py`, TileScale.
- **Example.** The sender and receiver loops of K25 (`DD:363-418`,
  `DD:451-478` of `EventControlFlow.md`), with L-TS1 and L-TS3.
- **Agents and steps.** On rank 0 the 32 lanes of warp 0 and lane 32; on rank
  1 lane 32 and the lanes of warp 0. Rank 0: warp 0 copies 32 words to `buf`
  of rank 1 with `T.put_warp`; every thread calls `T.sync_threads()`; lane 32
  executes `T.st(tail[0], 1, scope="sys", sem="release", dst_pe=1)`. Rank 1:
  lane 32 executes `T.wait_ge(tail[0], 1)`; every thread calls
  `T.sync_threads()`; warp 0 copies `buf` to `out`.
- **Asked.** Whether `out` equals `src`.
- **Expected.** Not decided. Two orderings are needed that no source states:
  on rank 0 from the stores of each copying lane to the release store of lane
  32, and on rank 1 from the wait of lane 32 to the loads of warp 0. The
  documentation of the barrier is L-TS3: "Synchronize all threads in a
  block." It states a rendezvous and no memory ordering. Weakest treatment:
  the copies of rank 1 are not ordered after the copies of rank 0, a data
  race, undefined behaviour. If a block barrier orders what each thread did
  before it ahead of what every thread does after it, the outcome under A1,
  clause c, is `out == src`.
- **Decided by.** Nothing. L-TS3 and E-TS1 ("They do not call NVSHMEM.") bound
  what is stated.
- **Mark.** N.
- **Events.** `access store plain` by 32 agents; `bump` and `wait` on a
  `phase` counter; `access store (atomic sys release)`; `wait` with `ge`;
  `access load plain`. The event draft gives the barrier the scope `cta` and
  leaves to ticket S5.2 whether a chain through it reaches the other rank.

#### LIT-a09. Load with `nc=True` of a word a peer writes

- **File.** `a09_nc_load_of_peer_written_word.py`, TileScale.
- **Example.** The docstring of `T.ld` (E-TS5).
- **Agents and steps.** Lane 0 of a block on rank 0 stores 42 to `word` of
  rank 1 with `T.st(word[0], 42, dst_pe=1)`. Lane 0 of a block on rank 1
  loads `word` with `T.ld(word[0], v, nc=True)`.
- **Asked.** Whether the program has a defined outcome.
- **Expected.** Undefined behaviour.
- **Decided by.** E-TS5: with `nc=True` the load "must not be used on memory a
  peer may write during the kernel".
- **Mark.** Q, on a docstring at the pinned commit. The weakest treatment
  gives the same: both accesses are ordinary (W1), a data race by I1.
- **Events.** `access store plain`; `access load plain`. Lemma
  `conflict_of_plain`.

#### LIT-a10. Message passing through a relaxed flag and two fences

- **File.** `a10_message_passing_with_fences.py`, TileScale.
- **Example.** L-CU3.
- **Agents and steps.** Lane 0 of block 0 and lane 0 of block 1 of one rank.
  The first stores 42 to `data` with an ordinary store, calls `T.fence_gpu()`,
  and stores 1 to `flag` with scope `gpu` and semantic `relaxed`. The second
  loads `flag` with scope `gpu` and semantic `relaxed` until it is not 0,
  calls `T.fence_gpu()`, and loads `data` with an ordinary load.
- **Asked.** The value of `out`.
- **Expected.** Not decided. In CUDA C++ the example prints 42 (L-CU3),
  because the first fence is a release fence and the second an acquire fence.
  The front end documents the scope of `T.fence_gpu` and no order (L-TS3), and
  `PrimitiveSources.md` has no row for it. Weakest treatment (W2): the fence
  orders nothing, the load of `data` races with the store, undefined
  behaviour. If `T.fence_gpu` is read as a fence with both orders, the outcome
  under A1, clause c, is 42. Triton-distributed has no documented fence with
  a scope either: `fence`, `membar` and `__fence` of `language_extra` are
  under W2.
- **Decided by.** Nothing in the front end.
- **Mark.** N.
- **Events.** `fence plain`, twice; `access store (atomic gpu relaxed)`;
  `access load (atomic gpu relaxed)`, repeated. A `fence` in mode `atomic`
  has no source in either front end.

### 5.2 Pattern (b): counter and wait

#### LIT-b01. Store, notify, wait at system scope

- **File.** `b01_notify_wait_system.py`, Triton-distributed.
- **Example.** E-TD10, with the scope `sys` that E-TD1 defines for several
  GPUs.
- **Agents and steps.** A CTA on rank 0 and a CTA on rank 1. The producer
  stores 42 to `queue` of rank 1 through `dl.symm_at`, then calls
  `dl.notify(signal, 1, signal=1, sig_op="set", comm_scope="intra_node")`. The
  consumer calls `dl.wait(signal, 1, "sys", "acquire", waitValue=1)`, consumes
  the token, and loads `queue`.
- **Asked.** The value of `out` on rank 1.
- **Expected.** Under A2 and if P-ATOM: allowed 42, forbidden 0. Without A2
  the notify is an ordinary update (W2), which races with the wait.
- **Decided by.** E-TD2: `wait` is "Wait until n signals at ptr reach expected
  values", `notify` is "Send a signal to a specific rank". E-TD1: `"acquire"`
  "Ensures all previous writes by other threads are visible after the wait".
  E-TD4: the load is "Guaranteed to execute after wait".
- **Mark.** R(A2). The wait is read by D5. D6 does not matter: the signal
  takes the values 0 and 1 only.
- **Events.** `access store plain`; `bump` of a `word` with `set`, mode
  `atomic sys release`; `wait` on the `word`, mode `atomic sys acquire`;
  `access load plain`.

#### LIT-b02. The same with the wait at device scope

- **File.** `b02_notify_wait_device_scope_two_ranks.py`, Triton-distributed.
- **Example.** E-TD10 as written, and K1 of tutorial 01 (`T01:109-133`).
- **Agents and steps.** As LIT-b01, but the wait has scope `gpu`.
- **Asked.** Whether the program has a defined outcome.
- **Expected.** Undefined behaviour.
- **Decided by.** E-TD1: `"gpu"` is "Synchronization visible only within the
  same GPU". E-CU1 and E-CU2 as in LIT-a04.
- **Mark.** R(D5). The listing of TD-SEM comments the load with "Now safe to
  compute on the tile" (E-TD10); decision D3 lets the definition govern. The
  listing also leaves `comm_scope` at its default `inter_node`, for which A2
  assumes nothing; the file passes `intra_node`, as tutorial 01 does.
- **Events.** `bump` in mode `atomic sys release` by an agent of rank 0;
  `wait` in mode `atomic gpu acquire` by rank 1. Lemma
  `not_syncEdge_of_acquire_gpu`; `Conflict`.

#### LIT-b03. Stores by all threads, `__syncthreads`, `fence`, notify

- **File.** `b03_stores_syncthreads_fence_notify.py`, Triton-distributed.
- **Example.** The producer of tutorial 01 (`T01:100-116`), E-TD11. The wait
  of the file has scope `sys`, so that LIT-b02 does not apply.
- **Agents and steps.** The threads of a CTA on rank 0, and a CTA on rank 1.
  Each thread of the producer stores its elements of a tile to `queue` of
  rank 1 with an ordinary store. The CTA calls `__syncthreads()` and
  `libshmem_device.fence()`. One thread performs the `dl.notify`; the tutorial
  says that the notify "is also single thread scope" (E-TD11). The consumer
  waits at scope `sys` and loads the tile.
- **Asked.** The value that the consumer reads for an element that a thread
  other than the notifying one stored.
- **Expected.** Not decided. The element of the notifying thread itself is as
  in LIT-b01. For the others the documentation of 3.6.5 hands the question to
  the memory model of the threads (L-NV1), and no source says that
  `__syncthreads` makes the stores of the other threads visible to the thread
  that fences and notifies: TD-API has one line, "Block-level synchronization
  barrier" (E-TD8). Weakest treatment, as decision D7 takes it: no ordering,
  a data race between the store and the load, undefined behaviour. If the
  barrier orders the stores of all threads ahead of the notify, the outcome
  is the tile.
- **Decided by.** L-NV1 decides that the fence alone does not cover the other
  threads. Nothing decides the barrier. The note E-NV22, which speaks of this
  pattern, is in `latest` and not in the manual of 3.6.5; it concerns a
  block-scoped put followed by an NVSHMEM signal, not ordinary stores followed
  by `dl.notify`.
- **Mark.** N.
- **Events.** `access store plain` by each thread; `bump` and `wait` on a
  `phase` counter; `fence` in mode `shmem`; `bump` of a `word`, mode `atomic
  sys release`; `wait`; `access load plain`. This is the hazard that the head
  of `EventSpec.lean` leaves to ticket S5.2.

#### LIT-b04. Load after a wait whose token is not consumed

- **File.** `b04_wait_without_consume_token.py`, Triton-distributed.
- **Example.** L-TD1.
- **Agents and steps.** As LIT-b01, but the consumer loads `queue` without
  `dl.consume_token`.
- **Asked.** Whether the load is ordered after the wait.
- **Expected.** It is not: the load may execute before the wait, hence
  concurrently with the store of the producer. Undefined behaviour.
- **Decided by.** L-TD1: "Without consume_token: compiler might reorder load
  before wait", and on the load "BUG: might execute before wait!". With the
  token the load is "Guaranteed to execute after wait" (E-TD4).
- **Mark.** Q.
- **Events.** As LIT-b01. The `access load plain` of the consumer is not
  after its `wait` in the order of the consumer's own events.

#### LIT-b05. Two ranks add to a signal, a third waits for the sum

- **File.** `b05_signal_add_wait_ge.py`, Triton-distributed.
- **Example.** L-NV9, E-NV10, E-NV11.
- **Agents and steps.** Thread 0 of a CTA on each of the ranks 0, 1 and 2.
  Ranks 1 and 2 each call `signal_op(signal, 1, NVSHMEM_SIGNAL_ADD, 0)`. Rank
  0 calls `signal_wait_until(signal, NVSHMEM_CMP_GE, 2)` and stores the value
  it returns.
- **Asked.** The returned value, and the final value of the signal word.
- **Expected.** Allowed: both are 2. Forbidden: a signal word that holds 1
  after both signal operations have completed; a returned value below 2.
- **Decided by.** L-NV9: "An update to signal data object is an atomic add
  operation." E-NV10: the routine "atomically updates sig_addr with signal
  using operation sig_op on the specified PE". L-NV3, second exception, and
  E-NV12: signal updates and a wait on the same word are not a conflict.
  L-NV9 for the returned value.
- **Mark.** Q. D2: the same under both readings; the waiting rank reads its
  own word.
- **Events.** `bump` of a `word` with `add`, mode `shmem`, twice; `wait` on
  the `word` with `ge`, mode `shmem`.

#### LIT-b06. The receiver polls the flag with an ordinary load

- **File.** `b06_plain_poll_of_signal_word.py`, Triton-distributed.
- **Example.** L-NV14; the sender is the example L-NV7.
- **Agents and steps.** A CTA on rank 0 puts a buffer to rank 1 with
  `putmem_block`, calls `quiet`, and its thread 0 calls `signal_op(signal, 1,
  NVSHMEM_SIGNAL_SET, 1)`. Thread 0 of a CTA on rank 1 loads `signal` with
  `tl.load` until it reads 1, then loads the buffer.
- **Asked.** Whether the program has a defined outcome.
- **Expected.** Undefined behaviour.
- **Decided by.** L-NV14: "A plain while loop or if condition to check flag
  value is not sufficient." L-NV3: a signal update and a direct load of the
  same word are a conflict; the second exception covers "signal updates and
  wait/test operations" only. E-NV11: the word "is expected only to be
  updated as a signal".
- **Mark.** Q. D2: the same under both readings.
- **Events.** `initiate` and `complete` of a transfer; `wait` on a
  `delivered` counter for the `quiet`; `bump` of a `word` with `set`, mode
  `shmem`; `access load plain` of that word. Lemma `conflict_of_plain`.

#### LIT-b07. `dl.wait` on a word that a put-with-signal writes

- **File.** `b07_dl_wait_on_put_signal_word.py`, Triton-distributed.
- **Example.** The inter-node path of tutorial 07: the wait of K17
  (`T07:181-185`) on the words that K18 writes (`T07:293-301`).
- **Agents and steps.** A CTA on rank 0 calls `putmem_signal_block` with
  `NVSHMEM_SIGNAL_SET` to rank 1. A CTA on rank 1 calls `dl.wait(signal, 1,
  "gpu", "acquire", waitValue=1)`, consumes the token, and loads the buffer.
- **Asked.** Whether the program has a defined outcome.
- **Expected.** Undefined behaviour.
- **Decided by.** L-NV3: a signal update conflicts with every concurrent
  access to the word other than "wait/test operations". E-NV12: signalling
  operations are atomic only with respect to other signal operations, the
  signal fetch and the point-to-point synchronization routines of NVSHMEM.
  No document says that `dl.wait` is one of them (I2).
- **Mark.** R(D5): under D5 the wait is a sequence of loads of the CUDA C++
  model, which is not an NVSHMEM routine. D2: the same under both readings.
- **Events.** `initiate` and `complete` of a transfer whose completion is a
  `bump` of a `word` in mode `shmem`; `wait` on that `word` in mode `atomic
  gpu acquire`. Lemma `conflict_of_atomic_of_shmem`.

#### LIT-b08. Host `set_signal` while a kernel waits with `dl.wait`

- **File.** `b08_host_set_signal_kernel_wait.py`, Triton-distributed.
- **Example.** The intra-node path of tutorial 07 (`T07:325-334` and
  `T07:181-185`), E-TD17.
- **Agents and steps.** The host of rank 0 and a CTA on rank 1. The host
  enqueues on a stream a copy to the buffer of rank 1 and then
  `set_signal(peer_signal, 1, stream)`. The CTA calls `dl.wait(ready, 1,
  "gpu", "acquire", waitValue=1)`, consumes the token, and loads the buffer.
- **Asked.** Whether the program has a defined outcome.
- **Expected.** Not decided. No document describes `set_signal` (E-TD17); the
  tutorial comments the call with "Notify the peer that the transmission is
  done". Weakest treatment (W2): an ordinary store of the signal word with no
  ordering after the copy. It conflicts with the wait of the kernel, a data
  race, undefined behaviour. This is inference I6.
- **Decided by.** Nothing. For comparison, the CUDA C++ model restricts
  concurrent access to an atomic object by CPU and GPU threads (L-CU4); which
  thread performs the write of `set_signal` is not documented.
- **Mark.** N.
- **Events.** `bump` of a `word` with `set`, mode `plain`, by a rank agent;
  `wait` on the `word`, mode `atomic gpu acquire`. This is the example at the
  end of `EventSpec.lean`.

#### LIT-b09. Host `set_signal` while a kernel waits with `T.wait_eq`

- **File.** `b09_host_set_signal_kernel_wait_tilescale.py`, TileScale.
- **Example.** E-TS12 and E-TS6; the wait is that of
  `TS:examples/distributed/allgather_gemm/example_allgather_gemm_overlapped.py`,
  line 103.
- **Agents and steps.** The host of a rank and a lane of a block of the same
  rank, on two streams. The host enqueues a copy into `buf` and then
  `set_signal(signal, 1, stream)`. The lane calls `T.wait_eq(signal[0], 1)`
  and loads `buf`.
- **Asked.** Whether the program has a defined outcome.
- **Expected.** Not decided. E-TS12 says what `set_signal` enqueues and
  states neither atomicity nor ordering (W6). Weakest treatment: an ordinary
  store, which conflicts with the wait. Undefined behaviour.
- **Decided by.** Nothing.
- **Mark.** N.
- **Events.** `bump` of a `word` with `set`, mode `plain`; `wait` with `eq`,
  mode `atomic sys acquire` under A1, clause c.

#### LIT-b10. One thread per peer waits, then `__syncthreads`

- **File.** `b10_thread_waits_then_syncthreads.py`, Triton-distributed.
- **Example.** K3 of tutorial 03 (`T03:65-78`), L-TD4.
- **Agents and steps.** In the program whose index equals the rank, thread
  `t` calls `signal_wait_until(flag + t, NVSHMEM_CMP_EQ, 1)` for each peer
  `t`; then the CTA calls `__syncthreads()` and stores 1 to `done`. In the
  other programs thread 0 of each peer calls `signal_op` with
  `NVSHMEM_SIGNAL_SET` on its word of the waiting rank.
- **Asked.** Whether the CTA can store to `done` before every waiting thread
  has seen its signal.
- **Expected.** Forbidden. Allowed: `done` is written only after all waits
  have returned.
- **Decided by.** E-TD8: `__syncthreads` is a "Block-level synchronization
  barrier". L-TD4: "wait for all peers done". E-NV11: the wait "blocks until
  the value contained in the signal data object, sig_addr, at the calling PE
  satisfies the wait condition".
- **Mark.** Q. The row uses the rendezvous of the barrier only. What the
  barrier does to memory ordering is the open point of LIT-a08 and LIT-b03.
  D2: the same under both readings.
- **Events.** `wait` on a `word` with `eq`, mode `shmem`, by several threads;
  `bump` and `wait` on a `phase` counter; `bump` of a `word` with `set`, mode
  `shmem`.

#### LIT-b11. Store before `T.barrier_blocks`, load by the peer after it

- **File.** `b11_barrier_blocks_then_load.py`, TileScale.
- **Example.** The docstrings E-TS8; the form is that of K23 and K24, which
  store to peers between two rendezvous.
- **Agents and steps.** A block on rank 0 and a block on rank 1. Lane 0 of
  rank 0 stores 42 to `x` of rank 1 with `T.st(x[0], 42, dst_pe=1)`. Both
  blocks call `T.barrier_blocks(barrier)`. Lane 0 of rank 1 then loads `x`
  with an ordinary load.
- **Asked.** Whether the load is ordered after the store.
- **Expected.** It is not: undefined behaviour.
- **Decided by.** E-TS8: "the barrier does not currently emit an acquire fence
  on exit, so ordering peer writes against loads issued after the call still
  requires an explicit fence". With `T.sync_blocks` in place of
  `T.barrier_blocks` the same holds a fortiori: it "provides no memory
  ordering in either direction".
- **Mark.** Q, on a docstring at the pinned commit. The docstring implies
  that an explicit fence after the barrier gives the ordering. Which fence is
  not said, and `T.fence_sys` documents no order (L-TS3): that variant is not
  decided and has no row.
- **Events.** `access store plain` to memory of rank 1; `bump` and `wait` on a
  `rendezvous` counter, with a release at scope `sys` on entry and mode
  `plain` on exit; `access load plain`.

#### LIT-b12. Relaxed store, `T.wait_ne`, volatile load

- **File.** `b12_wait_ne_then_volatile_load.py`, TileScale.
- **Example.** L-TS2 and L-TS1; the calls are those of K25 (`DD:314`,
  `DD:431-435`).
- **Agents and steps.** Lane 0 of a block on rank 0 executes `T.st(start[0],
  -7 - 1, scope="sys", sem="relaxed", dst_pe=1)`. Lane 0 of a block on rank 1
  executes `T.wait_ne(start[0], 0)`, then `T.ld(start[0], v, sem="volatile")`,
  and stores `-v - 1` to `out`.
- **Asked.** The value of `out`.
- **Expected.** Not decided. Under A1, clause c, and if P-ATOM the wait itself
  is defined: the store and the loads of the wait are atomic at system scope,
  and the wait returns after it has read -8. The second load is not: no
  document defines `volatile` (E-TS13), so it is an ordinary load (W1), and
  the relaxed store has no release side, so nothing orders the store before
  that load. A data race, undefined behaviour. With `release` in the store
  the acquire of the wait would order the load after the store, and the
  outcome would be 7.
- **Decided by.** Nothing for `volatile`. L-TS1 states an intent ("to ensure
  visibility across PEs") and no definition.
- **Mark.** N. At `9ba8a605` the waits take no scope or semantics argument and
  their docstring is one line; the row is stated for the interface of
  `71e7145`.
- **Events.** `access store (atomic sys relaxed)`; `wait` on a `word` with
  `ne`, mode `atomic sys acquire`; `access load plain`.

#### LIT-b13. Store before `barrier_all_block`, loads by two other ranks

- **File.** `b13_barrier_all_third_rank.py`, Triton-distributed.
- **Example.** E-NV13 with E-NV15; the host form is in tutorial 01 with the
  comment L-TD6.
- **Agents and steps.** A CTA on each of the ranks 0, 1 and 2. Thread 0 of
  rank 0 stores 42 to `x` of rank 1 through `dl.symm_at`. Every CTA calls
  `libshmem_device.barrier_all_block()`. Then thread 0 of rank 1 loads its own
  `x`, and thread 0 of rank 2 loads `x` of rank 1 through `dl.symm_at`.
- **Asked.** The value of `out` on rank 1 and on rank 2.
- **Expected.** Rank 1: allowed 42, under both readings. Rank 2, reading
  `global`: allowed 42. Rank 2, reading `ownRank`: the barrier guarantees
  nothing about memory of rank 1 to rank 2, the load is not ordered after the
  store, undefined behaviour.
- **Decided by.** E-NV13: `nvshmem_barrier_all` "ensures completion of all
  previously issued memory stores and remote memory updates". How far the
  completed updates are visible is E-NV15, which names this routine: the
  synchronization operations "only guarantee visibility of updates to the
  local PE's symmetric objects. However, on systems with only NVLink, these
  operations guarantee global visibility". The description of the team form
  has both cases in one sentence (L-NV12). The release notes of 3.6.5 state
  the weaker case without the exception (E-NV26). L-NV13 is a condition on
  the launch.
- **Mark.** D2 for rank 2; Q for rank 1.
- **Events.** `access store plain` to memory of rank 1 by an agent of rank 0;
  `bump` and `wait` on a `rendezvous` counter in mode `shmem`; `access load
  plain` twice. Lemma `SyncEdge.shmem_ownRank`.

### 5.3 Pattern (c): asynchronous transfer

In the rows of this section the reader is thread 0 of a CTA on the target
rank: it calls `signal_wait_until(signal, NVSHMEM_CMP_EQ, 1)` and then loads
the first word of the destination. A routine with the suffix `_block` is one
event of the whole block (decision D4, E-NV16).

#### LIT-c01. Put-with-signal, wait, read

- **File.** `c01_put_signal_wait.py`, Triton-distributed.
- **Example.** E-NV8; K2, K3 and K18 (tutorials 02, 03 and 07), L-TD2.
- **Agents and steps.** A CTA on rank 0 calls `putmem_signal_block(buf, buf,
  NBYTES, signal, 1, NVSHMEM_SIGNAL_SET, 1)`. The reader on rank 1 waits and
  loads `buf`.
- **Asked.** The value that the reader loads.
- **Expected.** Allowed: the payload. Forbidden: the old content of `buf`.
- **Decided by.** E-NV8: "The completion of signal update based on the sig_op
  signal operator using the signal flag on the remote PE indicates the
  delivery of its corresponding dest data words into the data object on the
  remote PE." E-NV11: the wait does not return "before the update of the
  memory indicated by sig_addr is fully complete". L-NV5: a wait "followed by
  a read operation", both on local symmetric memory, occur in order.
- **Mark.** Q. D2: the same under both readings; the reader is the
  destination and reads its own memory.
- **Events.** `initiate` of a transfer with `nbi = false`; `complete`, whose
  completion is a `bump` of a `word` with `set`, mode `shmem`; `wait` on that
  `word` with `eq`, mode `shmem`; `access load plain`. Lemma
  `SyncEdge.of_complete`.

#### LIT-c02. Put, then put-with-signal; the first put is read

- **File.** `c02_put_then_put_signal.py`, Triton-distributed.
- **Example.** L-NV8.
- **Agents and steps.** A CTA on rank 0 calls `putmem_block(a, a, NBYTES, 1)`
  and then `putmem_signal_block` for a second buffer `b` with the signal. The
  reader on rank 1 waits and loads `a`.
- **Asked.** Whether the load is ordered after the delivery of `a`.
- **Expected.** It is not: undefined behaviour.
- **Decided by.** L-NV8: "the completion of the signal update in a sequence
  consisting of a put routine followed by a put-with-signal routine does not
  imply delivery of the put routine's data". E-NV4: the put returns "after
  the data has been copied out of the source array on the local PE". L-NV3:
  the load conflicts with the delivery.
- **Mark.** Q. D2: the same under both readings.
- **Events.** Two transfers; the first has no completion bound to the signal.
  Lemmas `inFlight_of_initiate` and `conflict_complete`.

#### LIT-c03. Put, `fence`, put-with-signal; the first put is read

- **File.** `c03_put_fence_put_signal.py`, Triton-distributed.
- **Example.** E-NV1 with E-NV8.
- **Agents and steps.** As LIT-c02 with `libshmem_device.fence()` between the
  two calls.
- **Asked.** The value that the reader loads from `a`.
- **Expected.** Allowed: the payload of `a`. Forbidden: its old content.
- **Decided by.** E-NV1: "All operations on symmetric data objects issued to
  a particular PE prior to the call to nvshmem_fence are guaranteed to be
  delivered before any subsequent operations on symmetric data objects to the
  same PE." E-NV8: only "Without a memory-ordering operation" is there no
  ordering between the signal and another transfer. L-NV5: after a blocking
  access to the same target PE the ordering operations are
  "Fence/quiet/barrier". L-NV6 for the reader: the target observes the order
  through a wait.
- **Mark.** Q. D2: the same under both readings.
- **Events.** Two transfers; `fence` in mode `shmem`; `wait`; `access load
  plain`.

#### LIT-c04. Non-blocking put and signal without `quiet`

- **File.** `c04_put_nbi_signal_without_quiet.py`, Triton-distributed.
- **Example.** E-NV5, L-NV10.
- **Agents and steps.** A CTA on rank 0 calls `putmem_nbi_block(buf, buf,
  NBYTES, 1)`; its thread 0 calls `signal_op(signal, 1, NVSHMEM_SIGNAL_SET,
  1)`. Neither `quiet` nor `fence` is called. The reader on rank 1 waits and
  loads `buf`.
- **Asked.** Whether the load is ordered after the delivery of the put.
- **Expected.** It is not: undefined behaviour.
- **Decided by.** E-NV5: "The operation is considered complete after a
  subsequent call to nvshmem_quiet." L-NV10: non-blocking operations "must be
  completed using the nvshmem_quiet operation". L-NV4: "Updates to globally
  accessible objects are unordered." L-NV5: after a non-blocking access the
  ordering operations are "Quiet/barrier". The put does complete at some
  time (L-NV11); nothing relates that time to the signal.
- **Mark.** Q. D2: the same under both readings.
- **Events.** `initiate` of a transfer with `nbi = true`; `bump` of a `word`
  with `set`, mode `shmem`; `wait`; `access load plain` of a destination that
  is in flight. Lemma `inFlight_of_initiate`.

#### LIT-c05. Non-blocking put, `fence`, signal

- **File.** `c05_put_nbi_fence_signal.py`, Triton-distributed.
- **Example.** K6 of tutorial 04 (`T04:233-276`), L-TD3.
- **Agents and steps.** As LIT-c04 with `libshmem_device.fence()` between the
  put and the signal.
- **Asked.** The value that the reader loads.
- **Expected.** Under decision D1: undefined behaviour, as LIT-c04. Under the
  other reading of the sources the fence orders the delivery of the put
  before the signal, and the outcome is the payload.
- **Decided by.** The sources disagree. L-NV10: "Non-blocking operations are
  not ordered by calls to nvshmem_fence." L-NV5 gives "Quiet/barrier" after a
  non-blocking access. Against them, the table of the ordering page marks the
  non-blocking put under Fence (E-NV3), and the description of the put says
  that two puts "may deliver data out of order unless a call to nvshmem_fence
  is introduced between the two calls" (E-NV4, repeated for the non-blocking
  form).
- **Mark.** R(D1). D2: the same under both readings.
- **Events.** As LIT-c04 with `fence` in mode `shmem`.

#### LIT-c06. Non-blocking put, `quiet`, signal

- **File.** `c06_put_nbi_quiet_signal.py`, Triton-distributed.
- **Example.** E-NV5 and the example L-NV7.
- **Agents and steps.** As LIT-c04 with `libshmem_device.quiet()` between the
  put and the signal. Every thread of the block calls `quiet` after the put
  of the block has returned.
- **Asked.** The value that the reader loads.
- **Expected.** Allowed: the payload. Forbidden: the old content.
- **Decided by.** The description of the non-blocking put: "At the completion
  of nvshmem_quiet, the data has been copied into the dest array on the
  destination PE." L-NV2: the quiet "completes the operations posted prior to
  calling", and visibility holds at least "at the destination PE". E-NV11 and
  L-NV5 for the reader.
- **Mark.** Q. D2: the same under both readings; the reader is the
  destination.
- **Events.** `initiate` of a transfer with `nbi = true`; `complete`; `wait`
  on a `delivered` counter; `bump` of a `word`; `wait`; `access load plain`.

#### LIT-c07. Non-blocking get, `quiet`, load

- **File.** `c07_get_nbi_quiet_load.py`, Triton-distributed.
- **Example.** E-NV6.
- **Agents and steps.** A CTA on rank 0 calls `getmem_nbi_block(local, src,
  NBYTES, 1)` and `quiet()`; its thread 0 loads `local`. Rank 1 does not write
  `src` during the kernel.
- **Asked.** The value that rank 0 loads.
- **Expected.** Allowed: the value of `src` on rank 1. Forbidden: the old
  content of `local`. Without the `quiet` the load would not be ordered after
  the delivery.
- **Decided by.** E-NV6: "At the completion of nvshmem_quiet, the data has
  been delivered to the dest array on the local PE."
- **Mark.** Q. D2: the same under both readings.
- **Events.** `initiate` and `complete` of a transfer whose source is on rank
  1 and whose destination is local; `wait` on a `delivered` counter; `access
  load plain`.

#### LIT-c08. Put to rank 1, `quiet`, signal to rank 2, which reads rank 1

- **File.** `c08_put_nbi_quiet_third_rank.py`, Triton-distributed.
- **Example.** L-NV2, E-NV15, E-NV26.
- **Agents and steps.** A CTA on rank 0 calls `putmem_nbi_block` to rank 1,
  then `quiet()`; its thread 0 calls `signal_op` with `NVSHMEM_SIGNAL_SET` on
  the signal of rank 2. Thread 0 of a CTA on rank 2 waits on its signal and
  then loads the destination on rank 1 through `dl.symm_at`.
- **Asked.** The value that rank 2 loads.
- **Expected.** Reading `global`: allowed the payload, forbidden the old
  content. Reading `ownRank`: the quiet makes the data visible at rank 1 only,
  the load of rank 2 is not ordered after the delivery, undefined behaviour.
- **Decided by.** L-NV2 has both cases. For `global` also L-NV4 ("made
  visible to PEs other than the target PE") and L-NV6 ("can be observed by
  all PEs"), which carry no condition on the system. For `ownRank` also
  E-NV26: the routines "only ensure ordering and visibility between the
  source and destination PEs".
- **Mark.** D2.
- **Events.** As LIT-c06, with an `access load plain` of memory of rank 1 by
  an agent of rank 2. Lemma `SyncEdge.shmem_ownRank`.

#### LIT-c09. Remote TMA store, store wait, flag

- **File.** `c09_remote_tma_store_then_signal.py`, TileScale.
- **Example.** E-TS10 and L-TS4.
- **Agents and steps.** A block on rank 0 copies `src` to shared memory,
  calls `T.sync_threads()`, `T.tma_copy(shared, dst[0, 0], dst_pe=1)` and
  `T.tma_store_wait(0, False)`; its lane 0 then executes `T.st(flag[0], 1,
  scope="sys", sem="release", dst_pe=1)`. Lane 0 of a block on rank 1 executes
  `T.wait_eq(flag[0], 1)` and loads `dst`.
- **Asked.** The value that rank 1 loads.
- **Expected.** Not decided. The documentation states when the store is
  complete for the block that issued it (L-TS4) and nothing about a reader on
  the destination rank (W6). Weakest treatment: the load is of a destination
  whose transfer has no completion that the reader can observe, undefined
  behaviour. If the store wait is read as completion for every rank, the
  outcome under A1, clause c, is the payload.
- **Decided by.** Nothing for the reader. E-TS10: `T.tma_copy` "omits the wait
  so the user can batch multiple stores before calling T.tma_store_wait()
  explicitly". E-TS7: "Cross-GPU correctness still depends on using
  peer-visible storage and the appropriate system-scope ordering."
- **Mark.** N.
- **Events.** `initiate` and `complete` of a transfer whose destination is on
  rank 1; `wait` on its completion counter; `access store (atomic sys
  release)`; `wait` with `eq`; `access load plain`.

## 6. The three known hazards

| Hazard | Row | What the documentation allows |
| --- | --- | --- |
| A host `set_signal` concurrent with a kernel wait (tutorial 07) | LIT-b08; LIT-b09 for TileScale | Nothing: the host routine is not described. Weakest treatment: undefined behaviour |
| A `fence` after `__syncthreads`, followed by a notify of one thread (tutorial 01) | LIT-b03; LIT-a08 for the DeepEP port | The fence covers the calling thread only (L-NV1). Whether the barrier adds the other threads is stated by no source of 3.6.5 |
| A wait after a non-blocking put without `quiet` (tutorial 04) | LIT-c04; LIT-c05 with the fence of the tutorial | Undefined behaviour without an ordering routine. With the fence the sources disagree; decision D1 takes the weaker one |

Two further rows state a hazard of a tutorial as written that
`PrimitiveSources.md` already implies: LIT-b02 (tutorial 01 waits at scope
`gpu` on a word that another rank writes, I3) and LIT-b07 (tutorial 07 waits
with `dl.wait` on a word that an NVSHMEM signal writes, I2 with E-NV12).

## 7. Coverage

### 7.1 Constructors, modes and counters

| Item | Rows | No row |
| --- | --- | --- |
| `Event.access`, `load` and `store` | every row of pattern (a); the reads of patterns (b) and (c) | |
| `Event.access`, `update` | LIT-a05, LIT-a06 | |
| `Event.fence` | `plain`: LIT-a10. `shmem`: LIT-b03, LIT-c03, LIT-c05 | `atomic`: no front end documents a fence with an order (LIT-a10) |
| `Event.bump` | `word`, `set`: LIT-a05, LIT-b01 to LIT-b04, LIT-b06, LIT-b08 to LIT-b10, LIT-c04 to LIT-c06, LIT-c08; as the completion of a transfer: LIT-b07, LIT-c01 to LIT-c03. `word`, `add`: LIT-b05 | `add` in mode `atomic` (`dl.notify` with `sig_op="add"`) |
| `Event.wait` | `eq`: LIT-b09, LIT-b10, LIT-c01 to LIT-c06, LIT-c08, LIT-c09. `ge`: LIT-a07, LIT-a08, LIT-b05. `ne`: LIT-b12. `dl.wait` under D6: LIT-b01 to LIT-b04, LIT-b07, LIT-b08 | `le`, `gt`, `lt`: documented for the TileScale waits (E-TS6) and as NVSHMEM constants; no example and no target kernel uses them |
| `Event.initiate`, `Event.complete` | `nbi = false`: LIT-b06, LIT-b07, LIT-c01 to LIT-c03. `nbi = true`: LIT-c04 to LIT-c06, LIT-c08. Get: LIT-c07. TileScale: LIT-c09 | A transfer under W5 (`putmem_rma*`): undocumented, no example |
| `Mode.plain` | most rows; as the deciding mode in LIT-a08 to LIT-a10, LIT-b06, LIT-b08, LIT-b09, LIT-b12 | |
| `Mode.atomic` | scope `cta`: LIT-a02. Scope `gpu`: LIT-a01, LIT-a04, LIT-a05, LIT-a10, LIT-b02, LIT-b07, LIT-b08. Scope `sys`: LIT-a03, LIT-a06 to LIT-a08, LIT-b01 to LIT-b04, LIT-b09, LIT-b12, LIT-c09 | |
| `Mode.shmem` | LIT-b03, LIT-b05 to LIT-b07, LIT-b10, LIT-b13, every row of pattern (c) but LIT-c09 | `access` in mode `shmem`: the front end binds no NVSHMEM atomic memory operation |
| `MemOrder` | `relaxed`: LIT-a10, LIT-b12. `acquire`, `release`: most rows | `acqRel`: documented as the default of `tl.atomic_add` (E-TR1); no example uses it |
| `Counter.word` | most rows | |
| `Counter.phase` | LIT-a08, LIT-b03, LIT-b10, LIT-c09 | |
| `Counter.rendezvous` | LIT-b11, LIT-b13 | |
| `Counter.delivered` | LIT-b06, LIT-c06 to LIT-c08 | |
| `ShmemReach` | both values: LIT-b13, LIT-c08 | |

Front ends: 24 rows are Triton-distributed and 8 are TileScale. Target
kernels of `EventControlFlow.md` whose form a row takes: K1 (LIT-b02,
LIT-b03), K2, K3 and K18 (LIT-c01, LIT-b10), K6 (LIT-c04, LIT-c05), K11
(LIT-a06), K16 (LIT-a03), K17 (LIT-b07, LIT-b08), K19 (LIT-a05), K23 and K24
(LIT-b11), K25 (LIT-a07, LIT-a08, LIT-b12). K4, K12 and K20, K15, and K26 to
K28 have no row of their own; their event forms are those of the rows above.

### 7.2 Examples looked at and left out

| Example | Reason |
| --- | --- |
| NV365-MM, the two examples of L-NV15 | They use NVSHMEM atomic memory operations. `libshmem_device` at the pin binds none, so the examples cannot be written in the front end |
| NV365-ORD, examples of `nvshmem_fence` and `nvshmem_quiet` | The page of 3.6.5 names a listing file and shows only one sentence about it. LIT-c03 and LIT-c06 hold the property for one target PE |
| NV365-SIG, example of `nvshmem_put_signal`; NV365-SYNC, examples of the `wait_until` and `test` families | The pages of 3.6.5 name listing files without their text. Of these routines only `signal_wait_until` has a definition for the CUDA back end at the pin |
| NV365-COLL, broadcast, collect and reductions (E-NV20) | Collectives. No kernel of tutorials 01 to 08 calls one |
| NV365-FAQ, the questions on the `_on_stream` routines | Host routines; kernel code has no counterpart |
| CU-AT, "Example" | Declarations only; no interaction between threads |
| CU-MM, the litmus tests its last paragraph links to | Below the front-end language; not opened |
| TD-SEM, threadblock swizzling and the two overlap diagrams | They state no outcome |
| Tutorial 02, the host copies through symmetric pointers; tutorial 05, K10; tutorial 06, K12 | Host code, or a kernel whose only synchronization is a host barrier after it. LIT-b08 has a host copy with a signal, LIT-b13 the barrier in kernel code |
| `dl.tma.*`, the shared-memory operations, `extern_call` | Within one GPU or undocumented (W4, W6); no tutorial 01 to 08 uses them |
| TileScale, `T.atom_add` followed by `T.st` in `example_gemm_rs_overlapped.py` | The TileScale twin of LIT-a05; left out as a variant |
| TileScale multimem operations and signals | `MultimemSources.md`: no concurrent guarantee is documented, and the kernels that use them are not targets |
| TileScale, `T.init_barrier_gpu` and the other GPU barrier helpers | Listed by name only (E-TS7) |
| TD, `docs/kernels/nvidia/*.rst` and `docs/layers/nvidia/*.rst` | Searched for sentences on visibility, ordering and races: the matches are code listings. Not read in full |
| Tutorials 03a, 09 and 10 | Out of scope by the map |

## 8. Findings that concern other documents

These are observations. This ticket owns no other file and changes none.

1. **The manual of 3.6.5 can be read.** `PrimitiveSources.md` (sections 3 and
   10, item 3) says that its text could not be obtained. It is at the address
   of section 1, found through NV-VER.
2. **The quotations of `PrimitiveSources.md`, section 4.2, against 3.6.5.**
   They were compared by a script, word for word. Absent from 3.6.5: E-NV22
   (the notes on GPU threads under `nvshmem_fence` and `nvshmem_quiet`),
   E-NV23 (the flush routines) and E-NV24 (counted signals). Different in
   3.6.5: E-NV2, where 3.6.5 has neither "is a local, non-collective
   operation" nor the unconditional sentence on visibility, and has the two
   sentences of L-NV2 instead. Different in wording only: E-NV12, E-NV14 and
   E-NV16. The other quotations are in 3.6.5 as quoted. E-NV25 and E-NV26 are
   release notes and were not compared.
3. **Consequence for D7.** The third line of the table of D7 calls E-NV22
   "undated". It can now be dated by the manuals: not in 3.6.5.
4. **Consequence for D2.** The manual of 3.6.5 has the exception for systems
   with NVLink alone in three places (L-NV2, L-NV12, E-NV15), and its section
   "Data Consistency" speaks of all PEs without a condition (L-NV4, L-NV6).
   The release notes of 3.6.5 state the weaker case (E-NV26). The rows
   LIT-b13 and LIT-c08 give both.
5. **Condition P-ATOM.** L-CU5 makes atomicity at system scope on the memory
   of another GPU depend on an attribute of the platform, and the front end
   names the same condition (L-TD5). D5, A1 and A2 do not mention it. The rows
   LIT-a03, LIT-a06, LIT-a07, LIT-b01 and LIT-b12 depend on it. Whether the
   two-GPU node has the attribute was not checked.
6. **`T.fence_gpu` and `T.fence_sys`** have docstrings with a scope and no
   order (L-TS3) and no row in `PrimitiveSources.md`. Kernels 5 and 21 of the
   TileScale census call `T.fence_sys`.

## 9. Rows to review first

The expected outcome of these rows is the least certain.

| Row | Why |
| --- | --- |
| LIT-b03, LIT-a08 | The outcome turns on whether a block barrier orders the stores of other threads. No source says so; the weakest treatment makes tutorial 01 and the DeepEP sender undefined. The event draft leaves the point to ticket S5.2 |
| LIT-b07 | It makes the inter-node path of tutorial 07 undefined, because `dl.wait` is read as CUDA atomics (D5) and the signal as an NVSHMEM operation. If `dl.wait` were documented as an NVSHMEM wait the outcome would be the payload |
| LIT-b12 | It makes the offset handshake of the DeepEP port undefined through the word `volatile`. L-TS1 can be read as documenting what `volatile` is for |
| LIT-c05 | It rests on decision D1 between sources that disagree; the other reading gives the payload |
| LIT-b13, LIT-c08 | Two outcome sets, by decision D2, which is open |
| LIT-a03, LIT-a06, LIT-a07, LIT-b01 | Stated "if P-ATOM", a condition that nothing registers yet |
| LIT-a05 | Marked Q on the docstring of base Triton alone; the scope string of the call is still under A1, clause d |
| LIT-b10 | Marked Q on a one-line description of `__syncthreads`; only the rendezvous is used |
| LIT-a09 | Marked Q on a docstring that explains its rule by an instruction; the rule itself is stated at the level of the front end |

## 10. Not checked

- Nothing was run: no row on the model, none on a GPU, no source file
  imported. The files were compiled with `python3 -m py_compile` only.
- The calls of every file were compared by a script with the definitions at
  the pins: the name exists, the keywords are parameters, the positional
  arguments fit, and each `libshmem_device` and `language_extra` name has a
  definition for the CUDA back end. Whether a call is accepted with the given
  argument values, for example the scope string `cta` in `st`, was not
  checked: that needs the body of the definition.
- Four calls of the files are absent from the call tables of
  `documents/census/MinimalSyntax.md`, that is, no target kernel uses them:
  `libshmem_device.quiet` (LIT-b06, LIT-c06 to LIT-c08), `T.fence_gpu`
  (LIT-a10), and the two host functions `set_signal` (LIT-b08, LIT-b09).
- The pages of `latest` were compared with 3.6.5 only for the passages of
  item 2 of section 8. The NVSHMEM pages on atomic memory operations and on
  teams were not read.
- The libcu++ pages are the `unstable` ones; no versioned page was found.
- Whether the project's node is a system "with only NVLink" (D2) and whether
  it has the attribute of P-ATOM.
