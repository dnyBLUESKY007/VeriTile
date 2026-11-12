# Real synchronization-defect corpus

E.3 ([#266](https://github.com/dnyBLUESKY007/VeriTile/issues/266)).
Access date for **every source and tracker observation below: 2026-11-12**.
This is a bounded, evidence-graded corpus of reports, not a count of defects
reproduced or detected by VeriTile. No external kernel or reproducer was run.

## Scope and selection

The [map](https://github.com/dnyBLUESKY007/VeriTile/issues/1) owns the proof
boundary and source pins. The front-end semantics and their limitations are
recorded in [PrimitiveSources](../dist/PrimitiveSources.md); the
[syntax census](../census/MinimalSyntax.md) is an inventory, not evidence that
every construct is already parsed or verified. These local documents were read
at VeriTile `a0ba977236322702b1d0d8b891a34a4e1c9b2074`.

Searched the four primary GitHub issue trackers, including open and closed
issues, with `is:issue` and the terms `race`, `synchronization`, `hang` or
`deadlock`; also inspected the TileScale issue list. Search results were
screened by affected code and then the selected issue bodies and all comments
were read. This is a seed corpus, not an exhaustive tracker export. Search
ranking, later edits, and reports using different terminology can change
coverage. Build failures and configuration-only reports are not automatically
synchronization defects.

The table separates three questions:

1. **Evidence:** what the reporter or a later correction actually establishes.
2. **Pin applicability:** whether the affected source was checked at this plan's
   pin, rather than assuming a report about another revision applies.
3. **Expressibility (inference):** whether the failure belongs to source-level
   kernel semantics. A source program can be expressible while its compiler's
   incorrect lowering is outside the verifier's trust boundary. No row claims
   an implemented VeriTile proof, negative test, or parser acceptance.

Issue state is only a dated observation. An open issue is not confirmation;
a closed issue is not proof of a fix. No instruction-set semantics or
instruction-set references are used as evidence here. Where a report also
contains lower-level analysis, only its front-end program, reported outcome,
and explicitly identified correction are used below.

## Corpus

### E3-TD-169 — Wrong zero-token notification address

- **Source:** [Triton-distributed #169](https://github.com/ByteDance-Seed/Triton-distributed/issues/169),
  open, no comments; accessed 2026-11-12.
- **Affected code:** `python/triton_dist/kernels/nvidia/ep_all2all_fused.py`,
  `tile_kernel_dispatch_token_intra_node`, zero-token expert branch.
- **Reported evidence:** the issue compares the zero-token notification slot
  with the nonzero-token path and proposes replacing division by remainder in
  the slot calculation. It describes possible missed or premature readiness;
  it does not supply a measured failing run or a fixed revision.
- **Pin check:** the reported expression is present at plan pin
  [`7908e4ea9010bb2238f9f05b28c904fd4b71c60a`, lines 153–162](https://github.com/ByteDance-Seed/Triton-distributed/blob/7908e4ea9010bb2238f9f05b28c904fd4b71c60a/python/triton_dist/kernels/nvidia/ep_all2all_fused.py#L153-L162).
  The [nonzero-token path](https://github.com/ByteDance-Seed/Triton-distributed/blob/7908e4ea9010bb2238f9f05b28c904fd4b71c60a/python/triton_dist/kernels/nvidia/ep_all2all_fused.py#L115-L148)
  uses the local expert index; the
  [consumer](https://github.com/ByteDance-Seed/Triton-distributed/blob/7908e4ea9010bb2238f9f05b28c904fd4b71c60a/python/triton_dist/kernels/nvidia/ep_all2all_fused.py#L667-L673)
  waits on `expert_id * world_size + thread_idx`. The source was read as text
  outside the checkout; blob `c4a09809cc6ce523828999dd35340f04d9ab0748`.
- **Inference / expressibility:** **source-level candidate**: rank arithmetic,
  distinct signal addresses and a receiver wait are within the intended
  distributed front-end problem. For `experts_per_rank=2`, `world_size=2`,
  source `rank=0`, global expert `i=1`, the reported zero-token expression
  addresses slot 0 on destination rank 0; the local-expert layout addresses
  slot 2. This arithmetic witness establishes different addresses, not an
  end-to-end hang. A negative test must retain the routing preconditions,
  zero-initialized signal state and consumer path, and audit all other writers.
  Signal semantics and missing helper coverage must follow PrimitiveSources;
  no new ordering guarantee is inferred from this report.

### E3-TS-63 — Reuse and initialization of a grid barrier

- **Source:** [TileScale #63, items 2 and 3](https://github.com/tile-ai/tilescale/issues/63),
  open; accessed 2026-11-12.
- **Affected code:** reported `src/tl_templates/cuda/distributed/sync.h`,
  `init_barrier_gpu` / `sync_barrier_gpu`, and
  `testing/python/distributed/primitives/sync/test_barrier_gpu.py`.
  The report names release-branch revision `42374143`, not the plan pin.
- **Reported evidence:** item 2 argues that a counter not reset after the
  first phase cannot implement a second barrier. The reporter explicitly
  failed to reproduce a runtime failure and invalidated one attempted delay.
  Item 3 identifies in-kernel initialization potentially racing with arrivals;
  the reporter observed no failure. These are **audit hypotheses**, not
  reproduced deadlocks.
- **Pin check:** at TileScale plan pin
  [`71e71457d40ed6e6a2dad297936bfdbff21d104b`, `sync.py:134–151`](https://github.com/tile-ai/tilescale/blob/71e71457d40ed6e6a2dad297936bfdbff21d104b/tilelang/language/distributed/sync.py#L134-L151),
  the public wrappers exist, but their short docstrings do not specify safe
  reinitialization, reuse or phase rules. Their presence does **not** establish
  that the reported implementation defects persist at the pin. The helper
  implementation and the release branch were not inspected for this corpus.
- **Inference / expressibility:** **conditional protocol candidates**. A
  client calling init and then waiting, or using two barrier phases, is
  expressible through the named APIs. Proving those APIs' implementation
  defects is outside a source-level verifier that trusts them. To retain a
  client misuse case, first establish its documented phase/initialization
  preconditions; to verify a counter protocol itself, explicitly model its
  front-end counter operations. Do not silently expand the wrapper into
  lower-level semantics or register a hypothetical failure as detected.

### E3-TL-3527 — Shared predicate read versus in-branch write

- **Source:** [TileLang #3527](https://github.com/tile-ai/tilelang/issues/3527)
  and [investigation comment](https://github.com/tile-ai/tilelang/issues/3527#issuecomment-6207534866),
  open; accessed 2026-11-12.
- **Affected code:** the issue's `fold` reproducer reads shared `present[0]`
  as its branch condition, then updates the same flag inside the branch;
  reported compiler site `src/transform/thread_storage_sync.cc`.
- **Reported evidence:** wrong results on `19da9245` and `e40e09e3`, with
  controls that bind the predicate separately or latch it between
  `T.sync_threads()` calls. A later comment promises investigation, not a
  completed fix. No fixed revision was verified.
- **Pin applicability:** neither reported revision was equated with TileLang
  plan pin `35850c0`; no compiler or runtime replay was made here.
- **Inference / expressibility:** the shared flag, conditional read/write and
  explicit synchronization pattern are **source-level material**, subject to
  actual front-end support and documented automatic-synchronization semantics.
  The reported omission by the compiler is **outside the lowering trust
  boundary**. A verifier must not declare a correct high-level program buggy
  merely because a compiler omitted required synchronization. A manually
  altered source with an explicit missing edge would be a derived test,
  not this original defect.

### E3-TL-2613 — Partial synchronization mistaken for whole-CTA synchronization

- **Source:** [TileLang #2613](https://github.com/tile-ai/tilelang/issues/2613)
  and its [corrected analysis](https://github.com/tile-ai/tilelang/issues/2613#issuecomment-5919222457),
  open; accessed 2026-11-12.
- **Affected code:**
  `examples/warp_specialize/example_warp_specialize_gemm_copy_gemm_0_1.py`,
  two `T.ws` groups reusing a shared operand across iterations with
  `num_stages=0`; reported compiler site `thread_storage_sync.cc`.
- **Reported evidence:** the original body reports unstable outputs and a
  successful added-barrier control. **The later comment supersedes its
  explanation:** a barrier covering one thread group incorrectly clears
  other groups' pending accesses. It retracts async execution as the cause
  and the earlier claim that the older GPU architecture was unaffected.
  The tested repair is an edit to generated CUDA, not a verified upstream
  compiler fix.
- **Pin applicability:** the report names `cd37ed5fc35ae7a60a1277c8eb49028174ac51e6`
  and a later release observation; occurrence at `35850c0` is unchecked.
- **Inference / expressibility:** **compiler defect outside the proof
  boundary**, with a useful source-level distinction between participant
  sets. A future explicit protocol test can examine whether a group barrier
  orders another group's shared accesses, but it is not a reproduction of
  the compiler defect. Do not generalize the report to all warp-specialized
  GEMM or all pipelined configurations.

### E3-DE-589 — Output initialization races with communication stream

- **Source:** [DeepEP #589](https://github.com/deepseek-ai/DeepEP/issues/589),
  open, no comments; accessed 2026-11-12.
- **Affected code:** `csrc/deep_ep.cpp`, especially `intranode_combine`;
  the report also identifies the dispatch and internode counterparts.
- **Reported evidence:** at `567632dd59810d77b3cc05553df953cc0f779799`,
  the reporter places `torch::empty` after a cross-stream wait, so
  deterministic-mode output filling is not ordered before communication
  writes. It supplies an intermittent-NaN reproducer and serialized/no-fill
  controls. No independent reproduction or maintainer confirmation was read.
- **Pin applicability:** this is a DeepEP host-binding revision, not proof
  that the TileScale version-1 port at `9ba8a605` has the same host behavior.
- **Inference / expressibility:** **host-stream integration case, outside
  current kernel front ends**. A kernel-only statement assumes its input and
  output storage are ready; it cannot establish correct allocation-stream
  ordering. Retain this as a launch-contract hazard, not a counted kernel
  counterexample or a reason to import host orchestration into the core.

### E3-DE-496 — Different cross-library collective order across ranks

- **Source:** [DeepEP #496](https://github.com/deepseek-ai/DeepEP/issues/496)
  and [discussion](https://github.com/deepseek-ai/DeepEP/issues/496#issuecomment-3540496211),
  open; accessed 2026-11-12.
- **Affected code:** the issue's host snippet interleaves NCCL
  `all_gather_into_tensor` and `buffer.low_latency_dispatch` in different
  orders on even and odd ranks, then waits across streams. No fixed affected
  source commit is given in that snippet.
- **Reported evidence:** a timeout and several competing explanations:
  collective ordering, nonblocking-call assumptions and resource contention.
  [The reporter disputes the first resource explanation](https://github.com/deepseek-ai/DeepEP/issues/496#issuecomment-3541753097);
  a [later comment links a possible downstream workaround](https://github.com/deepseek-ai/DeepEP/issues/496#issuecomment-4187563399)
  and asks whether it resolves this reproduction. That is not confirmation
  of one root cause or fix.
- **Pin applicability / expressibility:** **unresolved and out of current
  scope**: host orchestration plus low-latency mode, whereas the map targets
  DeepEP version-1 normal mode. It motivates checking progress assumptions,
  but must not become a claimed kernel deadlock or accepted source-level case.

## Exclusions and contract caveats

| Source (all accessed 2026-11-12) | Evidence and disposition |
| --- | --- |
| [DeepEP #423](https://github.com/deepseek-ai/DeepEP/issues/423), `notify_dispatch` / `barrier_block`; closed | The [follow-up](https://github.com/deepseek-ai/DeepEP/issues/423#issuecomment-3421085038) retracts the deadlock diagnosis: rank startup skew exceeded the timeout. Exclude as evidence of a synchronization defect; no source pin or independent replay established. |
| [TileScale #63 item 1](https://github.com/tile-ai/tilescale/issues/63), `barrier_blocks` / `sync_blocks`; open | At the plan pin, the [public docstrings](https://github.com/tile-ai/tilescale/blob/71e71457d40ed6e6a2dad297936bfdbff21d104b/tilelang/language/distributed/sync.py#L154-L174) expressly limit ordering: `barrier_blocks` still requires an explicit fence for subsequent peer-data loads; `sync_blocks` is only a rendezvous. Treat absent ordering as a documented contract limitation, not an undocumented full-barrier guarantee. A client violating it could be a future source-level case; this report does not supply one verified here. |
| [TileScale #62](https://github.com/tile-ai/tilescale/issues/62), `MergeSharedMemoryAllocations`, release revision `42374143`; open | Reported branch-dependent aliasing of separate shared allocations is a compiler/lifetime issue. The issue supplies transform output rather than an independently verified runtime synchronization failure. Exclude from the synchronization core; do not transfer it to plan pin `71e7145` or treat trusted lowering as a source-language theorem. |
| [Triton-distributed #180](https://github.com/ByteDance-Seed/Triton-distributed/issues/180), `_swiglu_backward_kernel` and persistent variant | Reported overlapping output rows arise from missing address offsets, not an identified synchronization protocol. Keep as a separate memory-addressing lead; source-pin applicability, comments and runtime outcome were not audited for this corpus. |

## Use in evaluation

The six report entries above cover all four requested trackers. Only E3-TD-169
has its reported faulty expression independently matched to a plan pin here.
E3-TS-63 supplies two explicitly unconfirmed protocol hypotheses. The two
TileLang entries are lowering failures, and the two DeepEP entries are host
integration leads (one also outside the selected mode). None is an accepted
VeriTile detection or a fresh GPU reproduction.

Before counting a case, retain the exact source revision and minimal legal
inputs, distinguish reporter evidence from local reproduction, establish the
front-end contract and supported syntax, and show the intended positive control
and failing negative example through the actual verification entrypoint.
Compiler and host failures must remain separate from source-kernel detections.

Unchecked: fixes and current affected-version ranges, GPU execution, numerical
or liveness reproduction, frontend ingestion, Lean proof/negative-example
acceptance, and exhaustive tracker coverage. These limitations are not passed
tests. The corpus adds no assumption, runtime code or vendored source.
