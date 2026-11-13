# Real synchronization-defect corpus

E.3 ([#266](https://github.com/dnyBLUESKY007/VeriTile/issues/266)).
Original six-entry evidence was accessed on **2026-11-12**; the source-pin and
coverage audit for E.3b ([#423](https://github.com/dnyBLUESKY007/VeriTile/issues/423))
was performed on **2026-11-13**. Individual dated observations remain dated.
This is a bounded, evidence-graded corpus of reports, not a count of defects
reproduced or detected by VeriTile. No external kernel or reproducer was run.

## Scope and selection

The [map](https://github.com/dnyBLUESKY007/VeriTile/issues/1) owns the proof
boundary and source pins. The front-end semantics and their limitations are
recorded in [PrimitiveSources](../dist/PrimitiveSources.md); the
[syntax census](../census/MinimalSyntax.md) is an inventory, not evidence that
every construct is already parsed or verified. The construct mapping below
uses these local documents at VeriTile
`31d3757257ce345d6ba8ba753f4229ba06d68cc7`. Missing rows are stated explicitly: a
census absence is not proof of parser support or a semantic guarantee.

Searched the four primary GitHub issue trackers, including open and closed
issues, with `is:issue` and the terms `race`, `synchronization`, `hang` or
`deadlock`; also inspected the TileScale issue list. The appendix records every
returned hit, including repeated hits across
queries, with its disposition. The original six entries remain a selected seed
corpus: complete search-result accounting does not make them an exhaustive
defect sample. The 21 additional reports named by #423 were body-screened before
disposition; the appendix distinguishes promising unaudited cases from reports
outside the source-kernel boundary. Search indexing, later edits, and reports
using different terminology can change coverage. Build failures and
configuration-only reports are not automatically
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

## Documentation premise for the TileLang grades

Read at TileLang `35850c06875bcc84751417982bf7b4d662bafdbc`, as documentation,
not as an interpretation of generated code. These are short verbatim excerpts;
the rest of each passage is summarized, rather than copied in full.

| Source passage | Excerpt and documented scope | Applicability to the two reports |
| --- | --- | --- |
| [`instructions.md:44–50`](https://github.com/tile-ai/tilelang/blob/35850c06875bcc84751417982bf7b4d662bafdbc/docs/programming_guides/instructions.md#L44-L50) | For synchronous `T.copy`, “it is safe to use `dst`” after the statement; the implementation must preserve this observable contract. | Orders consumption of that copy's destination. It says neither that any raw shared predicate read is latched across all threads before an in-branch write (#3527), nor that a later overwrite waits for other warp groups' earlier reads (#2613). |
| [`instructions.md:52–66`](https://github.com/tile-ai/tilelang/blob/35850c06875bcc84751417982bf7b4d662bafdbc/docs/programming_guides/instructions.md#L52-L66) | For explicitly asynchronous copying, “A barrier is still required” for cooperative production and cross-thread consumption; the passage describes automatic insertion before the first destination read in most programs, and explicit synchronization when needed. | A copy-destination read-after-write contract, not a general shared read-before-write contract. Neither reported hazard is established as covered. |
| [`instructions.md:124–128`](https://github.com/tile-ai/tilelang/blob/35850c06875bcc84751417982bf7b4d662bafdbc/docs/programming_guides/instructions.md#L124-L128) | In high-level pipelines, “you usually don’t need to write explicit barriers”; named passes arrange producer/consumer order and synchronization. | Qualified high-level-pipeline guidance. #3527's raw condition and #2613's explicit two-group, `num_stages=0` loop are not promised whole-block read-before-write order by this statement. |
| [`language_basics.md:209–214`](https://github.com/tile-ai/tilelang/blob/35850c06875bcc84751417982bf7b4d662bafdbc/docs/programming_guides/language_basics.md#L209-L214) | The asynchronous-copy discussion says “before the first read from `dst`”; it requires an explicit completion wait and describes the cross-thread barrier. | Reinforces the first-use copy guarantee; no guarantee for either disputed write-after-read pattern. Line 214 completes the sentence that begins in the requested 209–213 range. |

**Inference under the current rules:** neither report can be dismissed solely
as a compiler violation of a documented source guarantee. Both are retained as
source-level race candidates when no unstated automatic edge is granted. This
is a corpus grade, not a new front-end definition, an implemented detection, or
a proof that the compiler's claimed contract is complete. A stronger documented
contract would require reassessment.

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
- **Constructs:** the affected zero-token producer uses `language_extra.ld`
  ([PrimitiveSources §5.4](../dist/PrimitiveSources.md#54-language_extra), row
  `ld`: documented, with the default relaxed string covered by A1), then
  `libshmem_device.signal_op` ([§5.3](../dist/PrimitiveSources.md#53-libshmem_device),
  row `signal_op`: documented atomic signal). The consumer uses
  `language_extra.ld_acquire` (§5.4 row `ld_acquire`: W1, no acquire guarantee),
  `while`, and `language_extra.__syncthreads` (§5.4 row `__syncthreads`:
  documented block-local barrier). The corresponding fully qualified call rows
  are in [MinimalSyntax, Triton-distributed calls](../census/MinimalSyntax.md#triton-distributed-calls),
  with `While` in its [statements](../census/MinimalSyntax.md#triton-distributed-statements).
  [§6 rules I1/I3](../dist/PrimitiveSources.md#6-inferences-and-decisions-that-apply-to-many-rows)
  and [§9 item 10](../dist/PrimitiveSources.md#9-decisions-of-2026-11-12)
  govern the undocumented consumer, independently of the address correction.
- **Inference / expressibility:** **source-level candidate, currently masked by
  a shared modeling limitation**: rank arithmetic,
  distinct signal addresses and a receiver wait are within the intended
  distributed front-end problem. For `experts_per_rank=2`, `world_size=2`,
  source `rank=0`, global expert `i=1`, the reported zero-token expression
  addresses slot 0 on destination rank 0; the local-expert layout addresses
  slot 2. This arithmetic witness establishes different addresses, not an
  end-to-end hang. A negative test must retain the routing preconditions,
  zero-initialized signal state and consumer path, and audit all other writers.
  Signal semantics and missing helper coverage must follow PrimitiveSources;
  no new ordering guarantee is inferred from this report. Under the current W1
  treatment, both the reported program and its address-corrected control admit
  the same conflicting signal/poll race (I1); the nominal `gpu` argument also
  cannot order a remote writer (I3). Thus the table does not establish a passing
  corrected control that isolates the address bug. A missed notification can
  produce spin-wait non-termination, rather than a demonstrated cyclic
  deadlock; the project's [forward-progress decision #20](https://github.com/dnyBLUESKY007/VeriTile/issues/20)
  is a separate assumption and cannot repair a wrong address or grant acquire
  semantics. No claim is made that a run actually hung.

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
  implementation and the release branch were not used to assign semantics.
  The pinned [client test, lines 21–33](https://github.com/tile-ai/tilescale/blob/71e71457d40ed6e6a2dad297936bfdbff21d104b/testing/python/distributed/primitives/sync/test_barrier_gpu.py#L21-L33)
  initializes inside every launched block, atomically updates `A`, calls
  `T.sync_barrier_gpu` **once**, then copies `A` to shared storage and to the
  output. The [host setup, lines 39–49](https://github.com/tile-ai/tilescale/blob/71e71457d40ed6e6a2dad297936bfdbff21d104b/testing/python/distributed/primitives/sync/test_barrier_gpu.py#L39-L49)
  zero-initializes `bar` before launch. Thus the in-kernel-init pattern is
  present, but this test does not exercise a second barrier phase and does not
  establish the reported counter-reuse defect at the pin.
- **Constructs:** [PrimitiveSources §5.5](../dist/PrimitiveSources.md#55-tilescale-kernel-side)
  rows `init_barrier_gpu` and `sync_barrier_gpu` are **W6**: neither counter
  protocol nor memory ordering is specified. Both calls are **absent** from
  [MinimalSyntax's TileScale call table](../census/MinimalSyntax.md#tilescale-calls),
  whose scanned files do not include this test. Its local `T.atomic_add` and
  `T.copy` do have census rows `tilelang.language.atomic_add` and
  `tilelang.language.copy`; §5.5's corresponding primitive rows cover remote
  arguments, **not** these local calls. Local-operation semantics need base
  TileLang documentation; no remote guarantee is imported into them.
- **Inference / expressibility:** **conditional protocol candidates with an
  explicit coverage gap**. The present model grants these grid-barrier wrappers
  no ordering or counter-protocol guarantee, so it cannot use them as trusted
  full barriers or distinguish a supposedly repaired implementation. A future
  client case needs a documented contract and census/parser coverage; an
  explicit front-end counter protocol would be a separate modeled program.
  Do not silently expand wrappers into lower-level semantics or register an
  audit hypothesis as a detected or reproduced failure.

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
- **Pin applicability:** GitHub compare at the access date places
  [`19da9245`](https://github.com/tile-ai/tilelang/compare/19da9245fe8c2deb810dfb6c777d0363d24b9cf8...35850c06875bcc84751417982bf7b4d662bafdbc)
  and [`e40e09e3`](https://github.com/tile-ai/tilelang/compare/e40e09e3e61595083d63a38bcf67f7ccb976b2ea...35850c06875bcc84751417982bf7b4d662bafdbc)
  before the pin (136 and 222 commits ahead respectively, zero behind; each
  merge base equals the reported revision). Ancestry is not evidence the
  defect survives intervening changes; no compiler or runtime replay was made.
- **Constructs:** shared allocation and raw scalar read/write inside `if` map
  to [MinimalSyntax's TileLang calls](../census/MinimalSyntax.md#tilelang-calls)
  row `tilelang.language.alloc_shared` and [statements](../census/MinimalSyntax.md#tilelang-statements)
  rows `If` / `Assign`. Raw subscript expressions are not separately counted
  (see its counting rules). The control's `T.sync_threads` is listed as
  `tilelang.language.sync_threads` in the **TileScale** call table, not the
  TileLang table. [PrimitiveSources §2](../dist/PrimitiveSources.md#2-enumeration)
  explicitly excludes base TileLang, including `T.sync_threads`; there is no
  primitive row to cite for these local operations. The documentation premise
  above supplies the available source guarantees and identifies what is missing.
- **Inference / expressibility:** **source-level race candidate under the
  current weakest semantics**. The raw `present[0]` read in the condition and
  the in-branch overwrite require a read-before-write edge across threads.
  None of the quoted documentation promises that edge for this pattern. The
  original blanket compiler-only grade is withdrawn; no automatic predicate
  latch is assumed. The explicit-barrier control remains reported evidence,
  not a locally accepted VeriTile positive control.

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
- **Pin applicability:** the reported
  [`cd37ed5fc35ae7a60a1277c8eb49028174ac51e6`](https://github.com/tile-ai/tilelang/compare/cd37ed5fc35ae7a60a1277c8eb49028174ac51e6...35850c06875bcc84751417982bf7b4d662bafdbc)
  is an ancestor (636 commits ahead, zero behind; merge base equals the
  reported revision). At the pin, the [example, lines 30–37](https://github.com/tile-ai/tilelang/blob/35850c06875bcc84751417982bf7b4d662bafdbc/examples/warp_specialize/example_warp_specialize_gemm_copy_gemm_0_1.py#L30-L37)
  still has `num_stages=0`, one all-thread copy of `A_shared`, and two `T.ws`
  groups reading it. Its [default sizes, lines 47–53](https://github.com/tile-ai/tilelang/blob/35850c06875bcc84751417982bf7b4d662bafdbc/examples/warp_specialize/example_warp_specialize_gemm_copy_gemm_0_1.py#L47-L53)
  have `K=block_K=64`, so the shipped invocation has no second iteration.
  Source-pattern persistence is not evidence of a surviving compiler defect.
- **Constructs:** [MinimalSyntax, TileLang calls](../census/MinimalSyntax.md#tilelang-calls)
  rows `tilelang.language.copy`, `.gemm`, `.ws`, `.Pipelined` and
  `.alloc_shared` (each abbreviation has prefix `tilelang.language`) describe
  the relevant operations; `For` and `With` appear in its statements table.
  [PrimitiveSources §2](../dist/PrimitiveSources.md#2-enumeration) excludes
  base TileLang: there are **no primitive rows** granting these local calls
  a whole-block barrier or read-completion edge. The premise table above
  states the documentation that can actually be applied.
- **Inference / expressibility:** **source-level race candidate under the
  current weakest semantics**, specifically the unpipelined, two-group reuse
  across iterations. Synchronous `T.copy` makes its destination consumable;
  it does not promise that both other groups have finished an earlier read
  before the next copy overwrites that storage. The qualified high-level
  pipeline statement also does not establish this edge for `num_stages=0`.
  The prior unconditional compiler-only classification is withdrawn. This
  does not generalize to ordinary pipelined GEMM or prove a runtime failure.

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
- **Pin applicability:** the version-1 port at
  [`9ba8a605d347f600a95d0275e2639cb00122c084`, `intranode/combine.py:378–397`](https://github.com/tile-ai/tilescale/blob/9ba8a605d347f600a95d0275e2639cb00122c084/examples/distributed/deepseek_deepep/intranode/combine.py#L378-L397)
  first enters a communication-stream block for notification (378–380), then
  allocates `recv_x` and `recv_topk_weights` outside that block (383–384), and
  re-enters the communication stream to launch combine (395–397). There is
  no communication-stream wait for these allocations between them and launch.
  The [wait at 413–415](https://github.com/tile-ai/tilescale/blob/9ba8a605d347f600a95d0275e2639cb00122c084/examples/distributed/deepseek_deepep/intranode/combine.py#L413-L415)
  is in the opposite direction, compute waiting for communication after launch.
  **Observation:** allocation after initial communication work and outside its
  stream is present. **Inference:** with distinct streams and an asynchronous
  initialization of the outputs, this function does not itself supply the
  required allocation-to-consumer edge. The original C++ wait sequence is not
  identical, and neither external caller ordering nor a failure was tested.
- **Constructs:** `torch.empty`, `torch.cuda.stream`,
  `torch.cuda.current_stream` and `Stream.wait_stream` are host orchestration.
  They have **no rows** in [PrimitiveSources §5.6](../dist/PrimitiveSources.md#56-tilescale-host-side)
  (which lists only TileScale signal utilities), or in [MinimalSyntax's kernel-only call tables](../census/MinimalSyntax.md#what-the-report-counts).
  Kernel invocation is the host/kernel boundary, not an implicit modeled wait.
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
- **Constructs:** the affected `all_gather_into_tensor`,
  `buffer.low_latency_dispatch` and host stream/event waits have **no rows** in
  [PrimitiveSources §5.6](../dist/PrimitiveSources.md#56-tilescale-host-side) or
  [MinimalSyntax](../census/MinimalSyntax.md#what-the-report-counts). They are
  NCCL/DeepEP host calls, not the enumerated front-end kernel primitives;
  no barrier, completion or progress guarantee is imported from their names.
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

The six report entries above cover all four requested trackers. E3-TD-169's
faulty expression is at the pin, but W1 prevents a valid address-corrected
control. E3-TS-63 remains an unconfirmed protocol audit; the pinned test has
in-kernel initialization and only one barrier phase. The two TileLang reports
are now source-level race candidates under the documented weakest guarantees,
with ancestry established and runtime persistence untested. The two DeepEP
entries remain host integration leads (one also outside the selected mode).
None is an accepted VeriTile detection or a fresh GPU reproduction.

**These entries are not a rate.** The appendix is complete for the sixteen
specified searches on the access date, not for the trackers or all defects.
Included means represented in this six-entry corpus, not verified or in-scope
for kernel detection. Excluded means not promoted to a graded entry, with the
reason stated; an unaudited synchronization lead remains a lead. A search hit
is neither an independent defect nor a valid denominator: one report can match
several terms, several reports can share a cause, and keyword searches miss
reports with other wording. Do not compute detection or defect rates here.

Before counting a case, retain the exact source revision and minimal legal
inputs, distinguish reporter evidence from local reproduction, establish the
front-end contract and supported syntax, and show the intended positive control
and failing negative example through the actual verification entrypoint.
Compiler and host failures must remain separate from source-kernel detections.

Still unverified: fixes and current affected-version ranges, GPU execution,
numerical or liveness reproduction, front-end ingestion, Lean proof/negative
example acceptance, and coverage outside these searches. Missing primitive or
census rows are coverage gaps, not passed checks. The corpus adds no assumption,
runtime code or vendored source.

## Appendix: complete accounting of the specified searches

Access date: **2026-11-13**. For each repository below, request
`GET /search/issues?q=repo:<repository>+is:issue+<term>&per_page=100&page=<page>`
for each of `race`, `synchronization`, `hang`, `deadlock`, without state or
field restrictions. All pages were collected until the returned item count
matched `total_count`; every response had `incomplete_results=false`.
Each row is one **query/report hit**, so a report matching three queries has
three rows. The sixteen totals sum to **268 rows**, representing **211 distinct
reports**. They match the ticket's 2026-11-12 totals. A zero-hit query contributes
zero rows. These are search totals, not reproduced-defect counts.

The tables list report numbers rather than copied titles or logs to avoid
personal details and lower-level material. Each linked body supports the
screening disposition; no reporter identities, host details, source attachments
or generated code are reproduced. The six main entries retain their deeper
evidence grades. Other relevant synchronization reports are explicitly
excluded from that graded seed pending source-contract/pin audit, rather than
called non-defects. The 21 reports singled out in #423 were body-screened;
exclusion of them is not an assertion that their alleged defect is false.

### Triton-distributed

Query counts (`race` / `synchronization` / `hang` / `deadlock`): **1 / 2 / 8 / 0**;
11 rows below.

| Query term | Report | Disposition |
| --- | --- | --- |
| `race` | [#180](https://github.com/ByteDance-Seed/Triton-distributed/issues/180) | Excluded: overlapping gradient stores are an addressing lead; no pin/contract audit for this corpus. |
| `synchronization` | [#129](https://github.com/ByteDance-Seed/Triton-distributed/issues/129) | Not a synchronization defect report: asks whether benchmark timing needs an inter-rank barrier. |
| `synchronization` | [#169](https://github.com/ByteDance-Seed/Triton-distributed/issues/169) | Included: E3-TD-169; wrong signal address, with the W1 consumer limitation recorded above. |
| `hang` | [#40](https://github.com/ByteDance-Seed/Triton-distributed/issues/40) | Excluded: performance complaint also reports a shape-dependent hang; no isolated synchronization cause at the pin. |
| `hang` | [#51](https://github.com/ByteDance-Seed/Triton-distributed/issues/51) | Not a synchronization report: installation stalls while building dependencies. |
| `hang` | [#79](https://github.com/ByteDance-Seed/Triton-distributed/issues/79) | Excluded: dual-GPU execution hang; source protocol and applicability at the plan pin remain unestablished. |
| `hang` | [#81](https://github.com/ByteDance-Seed/Triton-distributed/issues/81) | Excluded: dual-GPU execution hang; source protocol and applicability at the plan pin remain unestablished. |
| `hang` | [#99](https://github.com/ByteDance-Seed/Triton-distributed/issues/99) | Excluded: environment/NCCL or NVSHMEM failure logs do not isolate a source-level synchronization defect. |
| `hang` | [#107](https://github.com/ByteDance-Seed/Triton-distributed/issues/107) | Excluded: environment/NCCL or NVSHMEM failure logs do not isolate a source-level synchronization defect. |
| `hang` | [#124](https://github.com/ByteDance-Seed/Triton-distributed/issues/124) | Excluded: resource-failure report; no independently established synchronization counterexample. |
| `hang` | [#169](https://github.com/ByteDance-Seed/Triton-distributed/issues/169) | Included: E3-TD-169; wrong signal address, with the W1 consumer limitation recorded above. |

### TileLang

Query counts (`race` / `synchronization` / `hang` / `deadlock`): **50 / 52 / 40 / 19**;
161 rows below.

| Query term | Report | Disposition |
| --- | --- | --- |
| `race` | [#96](https://github.com/tile-ai/tilelang/issues/96) | Excluded: pipelined tutorial reports wrong results; a source-contract/pin audit is needed before grading this additional case. |
| `race` | [#666](https://github.com/tile-ai/tilelang/issues/666) | Excluded: pre-loop shared clear changes pipelined results; source-contract/pin applicability is not established here. |
| `race` | [#808](https://github.com/tile-ai/tilelang/issues/808) | Not a kernel synchronization report: compilation, tooling, indexing, arithmetic or runtime-support failure. |
| `race` | [#1671](https://github.com/tile-ai/tilelang/issues/1671) | Excluded: correctness/hang lead without an isolated source synchronization cause or established pin applicability. |
| `race` | [#1822](https://github.com/tile-ai/tilelang/issues/1822) | Excluded: explicit barrier fixes the reported mask; base-copy contract and pin applicability need a separate audit. |
| `race` | [#1935](https://github.com/tile-ai/tilelang/issues/1935) | Excluded: parallel compiler host-thread race, outside GPU kernel semantics. |
| `race` | [#2396](https://github.com/tile-ai/tilelang/issues/2396) | Excluded: serial fragment stores collide; layout/ownership contract and source-pin applicability need a separate audit. |
| `race` | [#2410](https://github.com/tile-ai/tilelang/issues/2410) | Excluded: correctness/hang lead without an isolated source synchronization cause or established pin applicability. |
| `race` | [#2426](https://github.com/tile-ai/tilelang/issues/2426) | Excluded: addressing, layout, dtype or instruction-selection correctness issue; no source synchronization case established. |
| `race` | [#2458](https://github.com/tile-ai/tilelang/issues/2458) | Not a synchronization defect report: dynamic indexing performance and thread utilization question. |
| `race` | [#2524](https://github.com/tile-ai/tilelang/issues/2524) | Excluded: addressing, layout, dtype or instruction-selection correctness issue; no source synchronization case established. |
| `race` | [#2542](https://github.com/tile-ai/tilelang/issues/2542) | Excluded: addressing, layout, dtype or instruction-selection correctness issue; no source synchronization case established. |
| `race` | [#2556](https://github.com/tile-ai/tilelang/issues/2556) | Excluded: partial active-thread region loses ordering; contract for that participant set and pin applicability need audit. |
| `race` | [#2566](https://github.com/tile-ai/tilelang/issues/2566) | Excluded: single-iteration automatic pipeline hang with controls; no independently audited source defect at the pin. |
| `race` | [#2593](https://github.com/tile-ai/tilelang/issues/2593) | Excluded: addressing, layout, dtype or instruction-selection correctness issue; no source synchronization case established. |
| `race` | [#2594](https://github.com/tile-ai/tilelang/issues/2594) | Excluded: compiler read/write-order or packed-storage ownership allegation; source-level contract/pin audit remains open. |
| `race` | [#2595](https://github.com/tile-ai/tilelang/issues/2595) | Excluded: compiler read/write-order or packed-storage ownership allegation; source-level contract/pin audit remains open. |
| `race` | [#2603](https://github.com/tile-ai/tilelang/issues/2603) | Excluded: addressing, layout, dtype or instruction-selection correctness issue; no source synchronization case established. |
| `race` | [#2604](https://github.com/tile-ai/tilelang/issues/2604) | Excluded: addressing, layout, dtype or instruction-selection correctness issue; no source synchronization case established. |
| `race` | [#2605](https://github.com/tile-ai/tilelang/issues/2605) | Excluded: addressing, layout, dtype or instruction-selection correctness issue; no source synchronization case established. |
| `race` | [#2612](https://github.com/tile-ai/tilelang/issues/2612) | Excluded: guarded-loop shared accesses lose ordering; source-contract and participant-set audit remains open. |
| `race` | [#2613](https://github.com/tile-ai/tilelang/issues/2613) | Included: E3-TL-2613; unpipelined cross-group reuse candidate, with source-contract limits recorded above. |
| `race` | [#2623](https://github.com/tile-ai/tilelang/issues/2623) | Excluded: addressing, layout, dtype or instruction-selection correctness issue; no source synchronization case established. |
| `race` | [#2627](https://github.com/tile-ai/tilelang/issues/2627) | Excluded: addressing, layout, dtype or instruction-selection correctness issue; no source synchronization case established. |
| `race` | [#2628](https://github.com/tile-ai/tilelang/issues/2628) | Excluded: compiler read/write-order or packed-storage ownership allegation; source-level contract/pin audit remains open. |
| `race` | [#2634](https://github.com/tile-ai/tilelang/issues/2634) | Excluded: lowering-level or architecture-specific allegation; no instruction semantics imported into this source corpus. |
| `race` | [#2648](https://github.com/tile-ai/tilelang/issues/2648) | Not a kernel synchronization report: compilation, tooling, indexing, arithmetic or runtime-support failure. |
| `race` | [#2699](https://github.com/tile-ai/tilelang/issues/2699) | Excluded: report attributes overlap to compiler reuse of distinct buffers; no source-level aliasing established. |
| `race` | [#2703](https://github.com/tile-ai/tilelang/issues/2703) | Excluded: shared clear/fill report is a separate relevant lead; front-end contract and pin applicability need audit. |
| `race` | [#2897](https://github.com/tile-ai/tilelang/issues/2897) | Not a synchronization defect report: usage, feature or contract-design discussion; no accepted counterexample here. |
| `race` | [#2963](https://github.com/tile-ai/tilelang/issues/2963) | Excluded: compiler read/write-order or packed-storage ownership allegation; source-level contract/pin audit remains open. |
| `race` | [#3037](https://github.com/tile-ai/tilelang/issues/3037) | Not a kernel synchronization report: compilation, tooling, indexing, arithmetic or runtime-support failure. |
| `race` | [#3220](https://github.com/tile-ai/tilelang/issues/3220) | Not a runtime synchronization defect report: compiler analysis, optimization or compilation-performance discussion. |
| `race` | [#3287](https://github.com/tile-ai/tilelang/issues/3287) | Excluded: automatic pipeline releases a stage too early; source-contract/pin audit is needed before grading. |
| `race` | [#3422](https://github.com/tile-ai/tilelang/issues/3422) | Not a synchronization defect report: usage, feature or contract-design discussion; no accepted counterexample here. |
| `race` | [#3424](https://github.com/tile-ai/tilelang/issues/3424) | Excluded: asynchronous shared-stage reuse is a source/code-generation audit hypothesis without a reported device failure. |
| `race` | [#3446](https://github.com/tile-ai/tilelang/issues/3446) | Excluded: lowering-level or architecture-specific allegation; no instruction semantics imported into this source corpus. |
| `race` | [#3451](https://github.com/tile-ai/tilelang/issues/3451) | Excluded: premature release is inferred from generated code without a device run; front-end contract/pin audit remains open. |
| `race` | [#3470](https://github.com/tile-ai/tilelang/issues/3470) | Excluded: training-only producer-phase hang; reported controls exist but no minimal source counterexample accepted here. |
| `race` | [#3519](https://github.com/tile-ai/tilelang/issues/3519) | Not a kernel synchronization report: compilation, tooling, indexing, arithmetic or runtime-support failure. |
| `race` | [#3525](https://github.com/tile-ai/tilelang/issues/3525) | Excluded: additional synchronization/ownership lead; source contract, minimal control and applicability at the pin need audit. |
| `race` | [#3527](https://github.com/tile-ai/tilelang/issues/3527) | Included: E3-TL-3527; raw shared predicate/write race candidate under the documented weakest guarantees. |
| `race` | [#3529](https://github.com/tile-ai/tilelang/issues/3529) | Excluded: additional synchronization/ownership lead; source contract, minimal control and applicability at the pin need audit. |
| `race` | [#3531](https://github.com/tile-ai/tilelang/issues/3531) | Not a kernel synchronization report: compilation, tooling, indexing, arithmetic or runtime-support failure. |
| `race` | [#3533](https://github.com/tile-ai/tilelang/issues/3533) | Not a kernel synchronization report: compilation, tooling, indexing, arithmetic or runtime-support failure. |
| `race` | [#3541](https://github.com/tile-ai/tilelang/issues/3541) | Not a kernel synchronization report: compilation, tooling, indexing, arithmetic or runtime-support failure. |
| `race` | [#3645](https://github.com/tile-ai/tilelang/issues/3645) | Excluded: lowering-level or architecture-specific allegation; no instruction semantics imported into this source corpus. |
| `race` | [#3808](https://github.com/tile-ai/tilelang/issues/3808) | Excluded: reported scalarization regression has a revert control; source-contract/pin applicability remains unaudited. |
| `race` | [#3870](https://github.com/tile-ai/tilelang/issues/3870) | Excluded: additional synchronization/ownership lead; source contract, minimal control and applicability at the pin need audit. |
| `race` | [#3938](https://github.com/tile-ai/tilelang/issues/3938) | Excluded: double-buffer loop has barrier controls; current-pin behavior and the source contract remain unaudited. |
| `synchronization` | [#386](https://github.com/tile-ai/tilelang/issues/386) | Not a runtime synchronization defect report: compiler analysis, optimization or compilation-performance discussion. |
| `synchronization` | [#666](https://github.com/tile-ai/tilelang/issues/666) | Excluded: pre-loop shared clear changes pipelined results; source-contract/pin applicability is not established here. |
| `synchronization` | [#1106](https://github.com/tile-ai/tilelang/issues/1106) | Excluded: additional synchronization/ownership lead; source contract, minimal control and applicability at the pin need audit. |
| `synchronization` | [#1257](https://github.com/tile-ai/tilelang/issues/1257) | Excluded: additional synchronization/ownership lead; source contract, minimal control and applicability at the pin need audit. |
| `synchronization` | [#1263](https://github.com/tile-ai/tilelang/issues/1263) | Not a kernel synchronization report: compilation, tooling, indexing, arithmetic or runtime-support failure. |
| `synchronization` | [#1463](https://github.com/tile-ai/tilelang/issues/1463) | Not a synchronization defect report: usage, feature or contract-design discussion; no accepted counterexample here. |
| `synchronization` | [#1493](https://github.com/tile-ai/tilelang/issues/1493) | Not a synchronization defect report: usage, feature or contract-design discussion; no accepted counterexample here. |
| `synchronization` | [#1535](https://github.com/tile-ai/tilelang/issues/1535) | Not a kernel synchronization report: compilation, tooling, indexing, arithmetic or runtime-support failure. |
| `synchronization` | [#1552](https://github.com/tile-ai/tilelang/issues/1552) | Excluded: correctness/hang lead without an isolated source synchronization cause or established pin applicability. |
| `synchronization` | [#1604](https://github.com/tile-ai/tilelang/issues/1604) | Excluded: additional synchronization/ownership lead; source contract, minimal control and applicability at the pin need audit. |
| `synchronization` | [#1919](https://github.com/tile-ai/tilelang/issues/1919) | Not a synchronization defect report: usage, feature or contract-design discussion; no accepted counterexample here. |
| `synchronization` | [#2056](https://github.com/tile-ai/tilelang/issues/2056) | Not a synchronization defect report: usage, feature or contract-design discussion; no accepted counterexample here. |
| `synchronization` | [#2123](https://github.com/tile-ai/tilelang/issues/2123) | Not a kernel synchronization report: compilation, tooling, indexing, arithmetic or runtime-support failure. |
| `synchronization` | [#2181](https://github.com/tile-ai/tilelang/issues/2181) | Not a synchronization defect report: usage, feature or contract-design discussion; no accepted counterexample here. |
| `synchronization` | [#2192](https://github.com/tile-ai/tilelang/issues/2192) | Excluded: additional synchronization/ownership lead; source contract, minimal control and applicability at the pin need audit. |
| `synchronization` | [#2376](https://github.com/tile-ai/tilelang/issues/2376) | Not a synchronization defect report: usage, feature or contract-design discussion; no accepted counterexample here. |
| `synchronization` | [#2524](https://github.com/tile-ai/tilelang/issues/2524) | Excluded: addressing, layout, dtype or instruction-selection correctness issue; no source synchronization case established. |
| `synchronization` | [#2539](https://github.com/tile-ai/tilelang/issues/2539) | Excluded: addressing, layout, dtype or instruction-selection correctness issue; no source synchronization case established. |
| `synchronization` | [#2556](https://github.com/tile-ai/tilelang/issues/2556) | Excluded: partial active-thread region loses ordering; contract for that participant set and pin applicability need audit. |
| `synchronization` | [#2566](https://github.com/tile-ai/tilelang/issues/2566) | Excluded: single-iteration automatic pipeline hang with controls; no independently audited source defect at the pin. |
| `synchronization` | [#2612](https://github.com/tile-ai/tilelang/issues/2612) | Excluded: guarded-loop shared accesses lose ordering; source-contract and participant-set audit remains open. |
| `synchronization` | [#2613](https://github.com/tile-ai/tilelang/issues/2613) | Included: E3-TL-2613; unpipelined cross-group reuse candidate, with source-contract limits recorded above. |
| `synchronization` | [#2628](https://github.com/tile-ai/tilelang/issues/2628) | Excluded: compiler read/write-order or packed-storage ownership allegation; source-level contract/pin audit remains open. |
| `synchronization` | [#2668](https://github.com/tile-ai/tilelang/issues/2668) | Excluded: additional synchronization/ownership lead; source contract, minimal control and applicability at the pin need audit. |
| `synchronization` | [#2707](https://github.com/tile-ai/tilelang/issues/2707) | Excluded: lowering-level or architecture-specific allegation; no instruction semantics imported into this source corpus. |
| `synchronization` | [#2708](https://github.com/tile-ai/tilelang/issues/2708) | Not a synchronization defect report: usage, feature or contract-design discussion; no accepted counterexample here. |
| `synchronization` | [#2992](https://github.com/tile-ai/tilelang/issues/2992) | Not a kernel synchronization report: compilation, tooling, indexing, arithmetic or runtime-support failure. |
| `synchronization` | [#3175](https://github.com/tile-ai/tilelang/issues/3175) | Not a runtime synchronization defect report: compiler analysis, optimization or compilation-performance discussion. |
| `synchronization` | [#3301](https://github.com/tile-ai/tilelang/issues/3301) | Excluded: aggregated failures need per-case isolation; a bundle is not one synchronization defect. |
| `synchronization` | [#3409](https://github.com/tile-ai/tilelang/issues/3409) | Not a kernel synchronization report: compilation, tooling, indexing, arithmetic or runtime-support failure. |
| `synchronization` | [#3410](https://github.com/tile-ai/tilelang/issues/3410) | Not a kernel synchronization report: compilation, tooling, indexing, arithmetic or runtime-support failure. |
| `synchronization` | [#3422](https://github.com/tile-ai/tilelang/issues/3422) | Not a synchronization defect report: usage, feature or contract-design discussion; no accepted counterexample here. |
| `synchronization` | [#3446](https://github.com/tile-ai/tilelang/issues/3446) | Excluded: lowering-level or architecture-specific allegation; no instruction semantics imported into this source corpus. |
| `synchronization` | [#3459](https://github.com/tile-ai/tilelang/issues/3459) | Not a synchronization defect report: usage, feature or contract-design discussion; no accepted counterexample here. |
| `synchronization` | [#3527](https://github.com/tile-ai/tilelang/issues/3527) | Included: E3-TL-3527; raw shared predicate/write race candidate under the documented weakest guarantees. |
| `synchronization` | [#3584](https://github.com/tile-ai/tilelang/issues/3584) | Not a synchronization defect report: usage, feature or contract-design discussion; no accepted counterexample here. |
| `synchronization` | [#3616](https://github.com/tile-ai/tilelang/issues/3616) | Excluded: Ascend/PTO backend synchronization or compilation report; outside the selected CUDA front ends. |
| `synchronization` | [#3641](https://github.com/tile-ai/tilelang/issues/3641) | Excluded: additional synchronization/ownership lead; source contract, minimal control and applicability at the pin need audit. |
| `synchronization` | [#3645](https://github.com/tile-ai/tilelang/issues/3645) | Excluded: lowering-level or architecture-specific allegation; no instruction semantics imported into this source corpus. |
| `synchronization` | [#3732](https://github.com/tile-ai/tilelang/issues/3732) | Not a runtime synchronization defect report: compiler analysis, optimization or compilation-performance discussion. |
| `synchronization` | [#3743](https://github.com/tile-ai/tilelang/issues/3743) | Not a kernel synchronization report: compilation, tooling, indexing, arithmetic or runtime-support failure. |
| `synchronization` | [#3745](https://github.com/tile-ai/tilelang/issues/3745) | Excluded: Ascend/PTO backend synchronization or compilation report; outside the selected CUDA front ends. |
| `synchronization` | [#3746](https://github.com/tile-ai/tilelang/issues/3746) | Not a kernel synchronization report: compilation, tooling, indexing, arithmetic or runtime-support failure. |
| `synchronization` | [#3747](https://github.com/tile-ai/tilelang/issues/3747) | Excluded: Ascend/PTO backend synchronization or compilation report; outside the selected CUDA front ends. |
| `synchronization` | [#3749](https://github.com/tile-ai/tilelang/issues/3749) | Excluded: Ascend/PTO backend synchronization or compilation report; outside the selected CUDA front ends. |
| `synchronization` | [#3773](https://github.com/tile-ai/tilelang/issues/3773) | Excluded: additional synchronization/ownership lead; source contract, minimal control and applicability at the pin need audit. |
| `synchronization` | [#3808](https://github.com/tile-ai/tilelang/issues/3808) | Excluded: reported scalarization regression has a revert control; source-contract/pin applicability remains unaudited. |
| `synchronization` | [#3866](https://github.com/tile-ai/tilelang/issues/3866) | Not a kernel synchronization report: compilation, tooling, indexing, arithmetic or runtime-support failure. |
| `synchronization` | [#3907](https://github.com/tile-ai/tilelang/issues/3907) | Not a kernel synchronization report: compilation, tooling, indexing, arithmetic or runtime-support failure. |
| `synchronization` | [#3918](https://github.com/tile-ai/tilelang/issues/3918) | Not a kernel synchronization report: compilation, tooling, indexing, arithmetic or runtime-support failure. |
| `synchronization` | [#3938](https://github.com/tile-ai/tilelang/issues/3938) | Excluded: double-buffer loop has barrier controls; current-pin behavior and the source contract remain unaudited. |
| `synchronization` | [#3976](https://github.com/tile-ai/tilelang/issues/3976) | Excluded: Ascend/PTO backend synchronization or compilation report; outside the selected CUDA front ends. |
| `hang` | [#284](https://github.com/tile-ai/tilelang/issues/284) | Excluded: additional synchronization/ownership lead; source contract, minimal control and applicability at the pin need audit. |
| `hang` | [#479](https://github.com/tile-ai/tilelang/issues/479) | Excluded: aggregated failures need per-case isolation; a bundle is not one synchronization defect. |
| `hang` | [#506](https://github.com/tile-ai/tilelang/issues/506) | Excluded: aggregated failures need per-case isolation; a bundle is not one synchronization defect. |
| `hang` | [#1026](https://github.com/tile-ai/tilelang/issues/1026) | Excluded: additional synchronization/ownership lead; source contract, minimal control and applicability at the pin need audit. |
| `hang` | [#1069](https://github.com/tile-ai/tilelang/issues/1069) | Not a kernel synchronization report: compilation, tooling, indexing, arithmetic or runtime-support failure. |
| `hang` | [#1132](https://github.com/tile-ai/tilelang/issues/1132) | Not a synchronization defect report: usage, feature or contract-design discussion; no accepted counterexample here. |
| `hang` | [#1150](https://github.com/tile-ai/tilelang/issues/1150) | Excluded: correctness/hang lead without an isolated source synchronization cause or established pin applicability. |
| `hang` | [#1370](https://github.com/tile-ai/tilelang/issues/1370) | Excluded: correctness/hang lead without an isolated source synchronization cause or established pin applicability. |
| `hang` | [#1498](https://github.com/tile-ai/tilelang/issues/1498) | Excluded: correctness/hang lead without an isolated source synchronization cause or established pin applicability. |
| `hang` | [#1534](https://github.com/tile-ai/tilelang/issues/1534) | Excluded: correctness/hang lead without an isolated source synchronization cause or established pin applicability. |
| `hang` | [#1535](https://github.com/tile-ai/tilelang/issues/1535) | Not a kernel synchronization report: compilation, tooling, indexing, arithmetic or runtime-support failure. |
| `hang` | [#2066](https://github.com/tile-ai/tilelang/issues/2066) | Not a kernel synchronization report: compilation, tooling, indexing, arithmetic or runtime-support failure. |
| `hang` | [#2200](https://github.com/tile-ai/tilelang/issues/2200) | Excluded: correctness/hang lead without an isolated source synchronization cause or established pin applicability. |
| `hang` | [#2301](https://github.com/tile-ai/tilelang/issues/2301) | Excluded: correctness/hang lead without an isolated source synchronization cause or established pin applicability. |
| `hang` | [#2328](https://github.com/tile-ai/tilelang/issues/2328) | Not a kernel synchronization report: compilation, tooling, indexing, arithmetic or runtime-support failure. |
| `hang` | [#2346](https://github.com/tile-ai/tilelang/issues/2346) | Excluded: correctness/hang lead without an isolated source synchronization cause or established pin applicability. |
| `hang` | [#2397](https://github.com/tile-ai/tilelang/issues/2397) | Not a kernel synchronization report: compilation, tooling, indexing, arithmetic or runtime-support failure. |
| `hang` | [#2410](https://github.com/tile-ai/tilelang/issues/2410) | Excluded: correctness/hang lead without an isolated source synchronization cause or established pin applicability. |
| `hang` | [#2506](https://github.com/tile-ai/tilelang/issues/2506) | Excluded: lowering-level or architecture-specific allegation; no instruction semantics imported into this source corpus. |
| `hang` | [#2525](https://github.com/tile-ai/tilelang/issues/2525) | Excluded: addressing, layout, dtype or instruction-selection correctness issue; no source synchronization case established. |
| `hang` | [#2527](https://github.com/tile-ai/tilelang/issues/2527) | Excluded: addressing, layout, dtype or instruction-selection correctness issue; no source synchronization case established. |
| `hang` | [#2528](https://github.com/tile-ai/tilelang/issues/2528) | Excluded: addressing, layout, dtype or instruction-selection correctness issue; no source synchronization case established. |
| `hang` | [#2555](https://github.com/tile-ai/tilelang/issues/2555) | Not a kernel synchronization report: compilation, tooling, indexing, arithmetic or runtime-support failure. |
| `hang` | [#2559](https://github.com/tile-ai/tilelang/issues/2559) | Excluded: additional synchronization/ownership lead; source contract, minimal control and applicability at the pin need audit. |
| `hang` | [#2564](https://github.com/tile-ai/tilelang/issues/2564) | Excluded: addressing, layout, dtype or instruction-selection correctness issue; no source synchronization case established. |
| `hang` | [#2566](https://github.com/tile-ai/tilelang/issues/2566) | Excluded: single-iteration automatic pipeline hang with controls; no independently audited source defect at the pin. |
| `hang` | [#2567](https://github.com/tile-ai/tilelang/issues/2567) | Excluded: additional synchronization/ownership lead; source contract, minimal control and applicability at the pin need audit. |
| `hang` | [#2572](https://github.com/tile-ai/tilelang/issues/2572) | Not a kernel synchronization report: compilation, tooling, indexing, arithmetic or runtime-support failure. |
| `hang` | [#2584](https://github.com/tile-ai/tilelang/issues/2584) | Excluded: addressing, layout, dtype or instruction-selection correctness issue; no source synchronization case established. |
| `hang` | [#2603](https://github.com/tile-ai/tilelang/issues/2603) | Excluded: addressing, layout, dtype or instruction-selection correctness issue; no source synchronization case established. |
| `hang` | [#2644](https://github.com/tile-ai/tilelang/issues/2644) | Excluded: lowering-level or architecture-specific allegation; no instruction semantics imported into this source corpus. |
| `hang` | [#3301](https://github.com/tile-ai/tilelang/issues/3301) | Excluded: aggregated failures need per-case isolation; a bundle is not one synchronization defect. |
| `hang` | [#3380](https://github.com/tile-ai/tilelang/issues/3380) | Not a kernel synchronization report: compilation, tooling, indexing, arithmetic or runtime-support failure. |
| `hang` | [#3395](https://github.com/tile-ai/tilelang/issues/3395) | Excluded: lowering-level or architecture-specific allegation; no instruction semantics imported into this source corpus. |
| `hang` | [#3470](https://github.com/tile-ai/tilelang/issues/3470) | Excluded: training-only producer-phase hang; reported controls exist but no minimal source counterexample accepted here. |
| `hang` | [#3524](https://github.com/tile-ai/tilelang/issues/3524) | Excluded: additional synchronization/ownership lead; source contract, minimal control and applicability at the pin need audit. |
| `hang` | [#3642](https://github.com/tile-ai/tilelang/issues/3642) | Excluded: additional synchronization/ownership lead; source contract, minimal control and applicability at the pin need audit. |
| `hang` | [#3680](https://github.com/tile-ai/tilelang/issues/3680) | Excluded: additional synchronization/ownership lead; source contract, minimal control and applicability at the pin need audit. |
| `hang` | [#3732](https://github.com/tile-ai/tilelang/issues/3732) | Not a runtime synchronization defect report: compiler analysis, optimization or compilation-performance discussion. |
| `hang` | [#3941](https://github.com/tile-ai/tilelang/issues/3941) | Not a kernel synchronization report: compilation, tooling, indexing, arithmetic or runtime-support failure. |
| `deadlock` | [#1069](https://github.com/tile-ai/tilelang/issues/1069) | Not a kernel synchronization report: compilation, tooling, indexing, arithmetic or runtime-support failure. |
| `deadlock` | [#1106](https://github.com/tile-ai/tilelang/issues/1106) | Excluded: additional synchronization/ownership lead; source contract, minimal control and applicability at the pin need audit. |
| `deadlock` | [#1535](https://github.com/tile-ai/tilelang/issues/1535) | Not a kernel synchronization report: compilation, tooling, indexing, arithmetic or runtime-support failure. |
| `deadlock` | [#2328](https://github.com/tile-ai/tilelang/issues/2328) | Not a kernel synchronization report: compilation, tooling, indexing, arithmetic or runtime-support failure. |
| `deadlock` | [#2522](https://github.com/tile-ai/tilelang/issues/2522) | Excluded: addressing, layout, dtype or instruction-selection correctness issue; no source synchronization case established. |
| `deadlock` | [#2559](https://github.com/tile-ai/tilelang/issues/2559) | Excluded: additional synchronization/ownership lead; source contract, minimal control and applicability at the pin need audit. |
| `deadlock` | [#2566](https://github.com/tile-ai/tilelang/issues/2566) | Excluded: single-iteration automatic pipeline hang with controls; no independently audited source defect at the pin. |
| `deadlock` | [#2567](https://github.com/tile-ai/tilelang/issues/2567) | Excluded: additional synchronization/ownership lead; source contract, minimal control and applicability at the pin need audit. |
| `deadlock` | [#2593](https://github.com/tile-ai/tilelang/issues/2593) | Excluded: addressing, layout, dtype or instruction-selection correctness issue; no source synchronization case established. |
| `deadlock` | [#2603](https://github.com/tile-ai/tilelang/issues/2603) | Excluded: addressing, layout, dtype or instruction-selection correctness issue; no source synchronization case established. |
| `deadlock` | [#2612](https://github.com/tile-ai/tilelang/issues/2612) | Excluded: guarded-loop shared accesses lose ordering; source-contract and participant-set audit remains open. |
| `deadlock` | [#2628](https://github.com/tile-ai/tilelang/issues/2628) | Excluded: compiler read/write-order or packed-storage ownership allegation; source-level contract/pin audit remains open. |
| `deadlock` | [#2644](https://github.com/tile-ai/tilelang/issues/2644) | Excluded: lowering-level or architecture-specific allegation; no instruction semantics imported into this source corpus. |
| `deadlock` | [#3030](https://github.com/tile-ai/tilelang/issues/3030) | Excluded: barrier-count input validation/contract gap; no validated synchronization counterexample. |
| `deadlock` | [#3380](https://github.com/tile-ai/tilelang/issues/3380) | Not a kernel synchronization report: compilation, tooling, indexing, arithmetic or runtime-support failure. |
| `deadlock` | [#3470](https://github.com/tile-ai/tilelang/issues/3470) | Excluded: training-only producer-phase hang; reported controls exist but no minimal source counterexample accepted here. |
| `deadlock` | [#3524](https://github.com/tile-ai/tilelang/issues/3524) | Excluded: additional synchronization/ownership lead; source contract, minimal control and applicability at the pin need audit. |
| `deadlock` | [#3642](https://github.com/tile-ai/tilelang/issues/3642) | Excluded: additional synchronization/ownership lead; source contract, minimal control and applicability at the pin need audit. |
| `deadlock` | [#3732](https://github.com/tile-ai/tilelang/issues/3732) | Not a runtime synchronization defect report: compiler analysis, optimization or compilation-performance discussion. |

### TileScale

Query counts (`race` / `synchronization` / `hang` / `deadlock`): **0 / 4 / 0 / 0**;
4 rows below.

| Query term | Report | Disposition |
| --- | --- | --- |
| `synchronization` | [#6](https://github.com/tile-ai/tilescale/issues/6) | Not a synchronization defect report: requests new synchronization/resource-control APIs. |
| `synchronization` | [#7](https://github.com/tile-ai/tilescale/issues/7) | Not a synchronization defect report: requests new synchronization/resource-control APIs. |
| `synchronization` | [#46](https://github.com/tile-ai/tilescale/issues/46) | Not a synchronization report: synchronization means updating the fork from upstream. |
| `synchronization` | [#63](https://github.com/tile-ai/tilescale/issues/63) | Included: E3-TS-63; unconfirmed barrier audit, with item 1 retained as a contract caveat. |

### DeepEP

Query counts (`race` / `synchronization` / `hang` / `deadlock`): **8 / 32 / 29 / 23**;
92 rows below.

| Query term | Report | Disposition |
| --- | --- | --- |
| `race` | [#289](https://github.com/deepseek-ai/DeepEP/issues/289) | Not a kernel synchronization report: dependency initialization or import/link failure. |
| `race` | [#491](https://github.com/deepseek-ai/DeepEP/issues/491) | Not a synchronization defect report: API/protocol/design question rather than an established failing program. |
| `race` | [#589](https://github.com/deepseek-ai/DeepEP/issues/589) | Included: E3-DE-589; host allocation/communication-stream race lead, not a kernel detection. |
| `race` | [#621](https://github.com/deepseek-ai/DeepEP/issues/621) | Excluded: low-latency-mode report or concern; the selected proof target is version-1 normal mode. |
| `race` | [#657](https://github.com/deepseek-ai/DeepEP/issues/657) | Excluded: version-2 pipeline stage-reuse allegation; outside the version-1 target, no instruction semantics adopted. |
| `race` | [#683](https://github.com/deepseek-ai/DeepEP/issues/683) | Excluded: host JIT-cache filesystem race, not GPU kernel synchronization. |
| `race` | [#756](https://github.com/deepseek-ai/DeepEP/issues/756) | Excluded: HybridEP configuration/lifetime/collective-order report; outside the selected version-1 port. |
| `race` | [#772](https://github.com/deepseek-ai/DeepEP/issues/772) | Excluded: version-2 low-latency network-visibility report; both the selected mode and front-end boundary exclude it. |
| `synchronization` | [#17](https://github.com/deepseek-ai/DeepEP/issues/17) | Excluded: low-latency-mode report or concern; the selected proof target is version-1 normal mode. |
| `synchronization` | [#28](https://github.com/deepseek-ai/DeepEP/issues/28) | Not a synchronization defect report: performance, timing or bandwidth question without an isolated ordering failure. |
| `synchronization` | [#51](https://github.com/deepseek-ai/DeepEP/issues/51) | Not a synchronization defect report: performance, timing or bandwidth question without an isolated ordering failure. |
| `synchronization` | [#71](https://github.com/deepseek-ai/DeepEP/issues/71) | Not a synchronization defect report: performance, timing or bandwidth question without an isolated ordering failure. |
| `synchronization` | [#93](https://github.com/deepseek-ai/DeepEP/issues/93) | Excluded: low-latency-mode report or concern; the selected proof target is version-1 normal mode. |
| `synchronization` | [#94](https://github.com/deepseek-ai/DeepEP/issues/94) | Not a synchronization defect report: API/protocol/design question rather than an established failing program. |
| `synchronization` | [#113](https://github.com/deepseek-ai/DeepEP/issues/113) | Not a synchronization defect report: API/protocol/design question rather than an established failing program. |
| `synchronization` | [#126](https://github.com/deepseek-ai/DeepEP/issues/126) | Excluded: transport/environment or timeout lead; no source-level defect in the pinned intranode port established here. |
| `synchronization` | [#128](https://github.com/deepseek-ai/DeepEP/issues/128) | Excluded: low-latency-mode report or concern; the selected proof target is version-1 normal mode. |
| `synchronization` | [#136](https://github.com/deepseek-ai/DeepEP/issues/136) | Excluded: instruction-level cache argument; below the front-end proof boundary and not used as semantic evidence. |
| `synchronization` | [#138](https://github.com/deepseek-ai/DeepEP/issues/138) | Not a synchronization defect report: performance, timing or bandwidth question without an isolated ordering failure. |
| `synchronization` | [#145](https://github.com/deepseek-ai/DeepEP/issues/145) | Excluded: transport/environment or timeout lead; no source-level defect in the pinned intranode port established here. |
| `synchronization` | [#160](https://github.com/deepseek-ai/DeepEP/issues/160) | Excluded: low-latency-mode report or concern; the selected proof target is version-1 normal mode. |
| `synchronization` | [#202](https://github.com/deepseek-ai/DeepEP/issues/202) | Excluded: low-latency-mode report or concern; the selected proof target is version-1 normal mode. |
| `synchronization` | [#208](https://github.com/deepseek-ai/DeepEP/issues/208) | Excluded: low-latency-mode report or concern; the selected proof target is version-1 normal mode. |
| `synchronization` | [#209](https://github.com/deepseek-ai/DeepEP/issues/209) | Excluded: low-latency-mode report or concern; the selected proof target is version-1 normal mode. |
| `synchronization` | [#220](https://github.com/deepseek-ai/DeepEP/issues/220) | Not a synchronization defect report: API/protocol/design question rather than an established failing program. |
| `synchronization` | [#223](https://github.com/deepseek-ai/DeepEP/issues/223) | Excluded: transport/environment or timeout lead; no source-level defect in the pinned intranode port established here. |
| `synchronization` | [#250](https://github.com/deepseek-ai/DeepEP/issues/250) | Not a synchronization defect report: performance, timing or bandwidth question without an isolated ordering failure. |
| `synchronization` | [#252](https://github.com/deepseek-ai/DeepEP/issues/252) | Not a synchronization defect report: API/protocol/design question rather than an established failing program. |
| `synchronization` | [#254](https://github.com/deepseek-ai/DeepEP/issues/254) | Not a synchronization defect report: performance, timing or bandwidth question without an isolated ordering failure. |
| `synchronization` | [#261](https://github.com/deepseek-ai/DeepEP/issues/261) | Not a synchronization defect report: performance, timing or bandwidth question without an isolated ordering failure. |
| `synchronization` | [#289](https://github.com/deepseek-ai/DeepEP/issues/289) | Not a kernel synchronization report: dependency initialization or import/link failure. |
| `synchronization` | [#290](https://github.com/deepseek-ai/DeepEP/issues/290) | Excluded: transport/environment or timeout lead; no source-level defect in the pinned intranode port established here. |
| `synchronization` | [#423](https://github.com/deepseek-ai/DeepEP/issues/423) | Excluded: follow-up retracts the deadlock diagnosis in favor of startup skew exceeding a timeout. |
| `synchronization` | [#429](https://github.com/deepseek-ai/DeepEP/issues/429) | Not a kernel synchronization report: dependency initialization or import/link failure. |
| `synchronization` | [#436](https://github.com/deepseek-ai/DeepEP/issues/436) | Not a synchronization defect report: API/protocol/design question rather than an established failing program. |
| `synchronization` | [#560](https://github.com/deepseek-ai/DeepEP/issues/560) | Not a synchronization defect report: API/protocol/design question rather than an established failing program. |
| `synchronization` | [#577](https://github.com/deepseek-ai/DeepEP/issues/577) | Excluded: low-latency-mode report or concern; the selected proof target is version-1 normal mode. |
| `synchronization` | [#636](https://github.com/deepseek-ai/DeepEP/issues/636) | Not a synchronization defect report: performance, timing or bandwidth question without an isolated ordering failure. |
| `synchronization` | [#668](https://github.com/deepseek-ai/DeepEP/issues/668) | Not a synchronization defect report: API/protocol/design question rather than an established failing program. |
| `synchronization` | [#756](https://github.com/deepseek-ai/DeepEP/issues/756) | Excluded: HybridEP configuration/lifetime/collective-order report; outside the selected version-1 port. |
| `hang` | [#54](https://github.com/deepseek-ai/DeepEP/issues/54) | Excluded: low-latency-mode report or concern; the selected proof target is version-1 normal mode. |
| `hang` | [#74](https://github.com/deepseek-ai/DeepEP/issues/74) | Excluded: transport/environment or timeout lead; no source-level defect in the pinned intranode port established here. |
| `hang` | [#82](https://github.com/deepseek-ai/DeepEP/issues/82) | Not a synchronization defect report: performance, timing or bandwidth question without an isolated ordering failure. |
| `hang` | [#103](https://github.com/deepseek-ai/DeepEP/issues/103) | Excluded: low-latency-mode report or concern; the selected proof target is version-1 normal mode. |
| `hang` | [#202](https://github.com/deepseek-ai/DeepEP/issues/202) | Excluded: low-latency-mode report or concern; the selected proof target is version-1 normal mode. |
| `hang` | [#240](https://github.com/deepseek-ai/DeepEP/issues/240) | Excluded: low-latency-mode report or concern; the selected proof target is version-1 normal mode. |
| `hang` | [#326](https://github.com/deepseek-ai/DeepEP/issues/326) | Excluded: transport/environment or timeout lead; no source-level defect in the pinned intranode port established here. |
| `hang` | [#330](https://github.com/deepseek-ai/DeepEP/issues/330) | Excluded: low-latency-mode report or concern; the selected proof target is version-1 normal mode. |
| `hang` | [#364](https://github.com/deepseek-ai/DeepEP/issues/364) | Excluded: transport/environment or timeout lead; no source-level defect in the pinned intranode port established here. |
| `hang` | [#366](https://github.com/deepseek-ai/DeepEP/issues/366) | Excluded: C++ shift-width correctness proposal; not an audited front-end synchronization case. |
| `hang` | [#368](https://github.com/deepseek-ai/DeepEP/issues/368) | Excluded: low-latency-mode report or concern; the selected proof target is version-1 normal mode. |
| `hang` | [#378](https://github.com/deepseek-ai/DeepEP/issues/378) | Excluded: low-latency-mode report or concern; the selected proof target is version-1 normal mode. |
| `hang` | [#398](https://github.com/deepseek-ai/DeepEP/issues/398) | Excluded: low-latency-mode report or concern; the selected proof target is version-1 normal mode. |
| `hang` | [#406](https://github.com/deepseek-ai/DeepEP/issues/406) | Excluded: transport/environment or timeout lead; no source-level defect in the pinned intranode port established here. |
| `hang` | [#411](https://github.com/deepseek-ai/DeepEP/issues/411) | Excluded: body reports an NCCL ALLGATHER watchdog timeout; it does not isolate a DeepEP kernel synchronization cause. |
| `hang` | [#417](https://github.com/deepseek-ai/DeepEP/issues/417) | Excluded: transport/environment or timeout lead; no source-level defect in the pinned intranode port established here. |
| `hang` | [#427](https://github.com/deepseek-ai/DeepEP/issues/427) | Excluded: low-latency-mode report or concern; the selected proof target is version-1 normal mode. |
| `hang` | [#461](https://github.com/deepseek-ai/DeepEP/issues/461) | Not a synchronization defect report: API/protocol/design question rather than an established failing program. |
| `hang` | [#496](https://github.com/deepseek-ai/DeepEP/issues/496) | Included: E3-DE-496; unresolved cross-library ordering report, also outside the selected mode. |
| `hang` | [#509](https://github.com/deepseek-ai/DeepEP/issues/509) | Excluded: transport/environment or timeout lead; no source-level defect in the pinned intranode port established here. |
| `hang` | [#513](https://github.com/deepseek-ai/DeepEP/issues/513) | Excluded: low-latency-mode report or concern; the selected proof target is version-1 normal mode. |
| `hang` | [#548](https://github.com/deepseek-ai/DeepEP/issues/548) | Excluded: low-latency-mode report or concern; the selected proof target is version-1 normal mode. |
| `hang` | [#553](https://github.com/deepseek-ai/DeepEP/issues/553) | Excluded: transport/environment or timeout lead; no source-level defect in the pinned intranode port established here. |
| `hang` | [#561](https://github.com/deepseek-ai/DeepEP/issues/561) | Excluded: HybridEP configuration/lifetime/collective-order report; outside the selected version-1 port. |
| `hang` | [#570](https://github.com/deepseek-ai/DeepEP/issues/570) | Excluded: transport/environment or timeout lead; no source-level defect in the pinned intranode port established here. |
| `hang` | [#596](https://github.com/deepseek-ai/DeepEP/issues/596) | Excluded: HybridEP configuration/lifetime/collective-order report; outside the selected version-1 port. |
| `hang` | [#604](https://github.com/deepseek-ai/DeepEP/issues/604) | Excluded: HybridEP configuration/lifetime/collective-order report; outside the selected version-1 port. |
| `hang` | [#658](https://github.com/deepseek-ai/DeepEP/issues/658) | Excluded: transport/environment or timeout lead; no source-level defect in the pinned intranode port established here. |
| `hang` | [#707](https://github.com/deepseek-ai/DeepEP/issues/707) | Excluded: transport/environment or timeout lead; no source-level defect in the pinned intranode port established here. |
| `deadlock` | [#17](https://github.com/deepseek-ai/DeepEP/issues/17) | Excluded: low-latency-mode report or concern; the selected proof target is version-1 normal mode. |
| `deadlock` | [#20](https://github.com/deepseek-ai/DeepEP/issues/20) | Not a synchronization defect report: asks why adaptive routing has documented restrictions; inter-host transport is outside scope. |
| `deadlock` | [#33](https://github.com/deepseek-ai/DeepEP/issues/33) | Not a synchronization defect report: asks why a subset of threads may call a collective; no failing execution supplied. |
| `deadlock` | [#39](https://github.com/deepseek-ai/DeepEP/issues/39) | Not a synchronization defect report: proposes an alternative buffering architecture. |
| `deadlock` | [#44](https://github.com/deepseek-ai/DeepEP/issues/44) | Not a synchronization defect report: asks about block/SM placement and residency. |
| `deadlock` | [#126](https://github.com/deepseek-ai/DeepEP/issues/126) | Excluded: transport/environment or timeout lead; no source-level defect in the pinned intranode port established here. |
| `deadlock` | [#128](https://github.com/deepseek-ai/DeepEP/issues/128) | Excluded: low-latency-mode report or concern; the selected proof target is version-1 normal mode. |
| `deadlock` | [#145](https://github.com/deepseek-ai/DeepEP/issues/145) | Excluded: transport/environment or timeout lead; no source-level defect in the pinned intranode port established here. |
| `deadlock` | [#159](https://github.com/deepseek-ai/DeepEP/issues/159) | Excluded: host-language lock/progress question; outside the kernel front ends. |
| `deadlock` | [#160](https://github.com/deepseek-ai/DeepEP/issues/160) | Excluded: low-latency-mode report or concern; the selected proof target is version-1 normal mode. |
| `deadlock` | [#189](https://github.com/deepseek-ai/DeepEP/issues/189) | Excluded: low-latency-mode report or concern; the selected proof target is version-1 normal mode. |
| `deadlock` | [#202](https://github.com/deepseek-ai/DeepEP/issues/202) | Excluded: low-latency-mode report or concern; the selected proof target is version-1 normal mode. |
| `deadlock` | [#209](https://github.com/deepseek-ai/DeepEP/issues/209) | Excluded: low-latency-mode report or concern; the selected proof target is version-1 normal mode. |
| `deadlock` | [#223](https://github.com/deepseek-ai/DeepEP/issues/223) | Excluded: transport/environment or timeout lead; no source-level defect in the pinned intranode port established here. |
| `deadlock` | [#289](https://github.com/deepseek-ai/DeepEP/issues/289) | Not a kernel synchronization report: dependency initialization or import/link failure. |
| `deadlock` | [#290](https://github.com/deepseek-ai/DeepEP/issues/290) | Excluded: transport/environment or timeout lead; no source-level defect in the pinned intranode port established here. |
| `deadlock` | [#346](https://github.com/deepseek-ai/DeepEP/issues/346) | Not a synchronization defect report: API/protocol/design question rather than an established failing program. |
| `deadlock` | [#411](https://github.com/deepseek-ai/DeepEP/issues/411) | Excluded: body reports an NCCL ALLGATHER watchdog timeout; it does not isolate a DeepEP kernel synchronization cause. |
| `deadlock` | [#423](https://github.com/deepseek-ai/DeepEP/issues/423) | Excluded: follow-up retracts the deadlock diagnosis in favor of startup skew exceeding a timeout. |
| `deadlock` | [#429](https://github.com/deepseek-ai/DeepEP/issues/429) | Not a kernel synchronization report: dependency initialization or import/link failure. |
| `deadlock` | [#496](https://github.com/deepseek-ai/DeepEP/issues/496) | Included: E3-DE-496; unresolved cross-library ordering report, also outside the selected mode. |
| `deadlock` | [#596](https://github.com/deepseek-ai/DeepEP/issues/596) | Excluded: HybridEP configuration/lifetime/collective-order report; outside the selected version-1 port. |
| `deadlock` | [#668](https://github.com/deepseek-ai/DeepEP/issues/668) | Not a synchronization defect report: API/protocol/design question rather than an established failing program. |
