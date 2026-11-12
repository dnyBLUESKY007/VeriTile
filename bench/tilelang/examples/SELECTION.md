# K-L.1: proposed TileLang example selection

Status: **proposal; user confirmation pending**. Access date for all sources and
tracker observations: **2026-11-12**. This is the document deliverable of
[#228](https://github.com/dnyBLUESKY007/VeriTile/issues/228), not implementation,
a proof result, or acceptance of the list.

## Sources and interpretation

| Source | Fixed revision and role |
| --- | --- |
| VeriTile upstream | `3a0d07615a31b28e46ceff9660ce44351c425a73`: the map's upstream pin; the nine worked-example Python counterparts below still exist under `bench/examples/` at this revision |
| TileLang | `35850c06875bcc84751417982bf7b4d662bafdbc`: the map's TileLang pin; related upstream examples read as text only |
| VeriTileBench | `39f6669cd2d714c31940256a326b8b0373f5bb7f`: proposed benchmark-source pin, resolved from its `main` on the access date; the map names this repository but does not pin its source revision |
| Fork | `a0ba977236322702b1d0d8b891a34a4e1c9b2074`: inspection baseline; `experiments/interp_gpu/specs/ASSIGNMENT.tsv` is absent |

The [map](https://github.com/dnyBLUESKY007/VeriTile/issues/1) owns the scope and
hard constraints. A benchmark package's VeriTile dependency revision is not a
pin for the benchmark's own source. The benchmark pin here is an explicit
proposal, not a previously approved map decision.

**Observed** below means source syntax or an existing issue requirement.
**Proposed** means a selection or intended implementation decision. Global,
shared and fragment describe source-level storage, not synchronization scope,
physical register allocation, or a compiler guarantee. Triton `tl.load` and
`tl.store` address global input/output buffers in the selected harnesses;
intermediate SSA tensors are not explicit shared-memory allocations. The
TileLang storage column is planned unless a named upstream example is reused.
No execution, importing, lowering inspection or low-level semantic evidence was
used. Python files with top-level tests were read, never imported.

## Nine worked examples

All nine names are retained from #228. The proposal targets each example's
original and comparison variants together, including explicit dtype conversions,
masks, multi-launch intermediates and guards. Reusing an upstream example means
adapting a starting point, not claiming identical arithmetic or an existing proof.
“Written by us” is an implementation choice; it does not claim an exhaustive
search proved that no upstream implementation exists.

| Target and observed Triton counterpart | TileLang source choice (proposed) | Source-level storage exercised and preservation requirement |
| --- | --- | --- |
| [VectorAdd](https://github.com/VeriInfra-Lab/VeriTile/blob/3a0d07615a31b28e46ceff9660ce44351c425a73/bench/examples/VectorAdd/VectorAdd.py) — `add_kernel`, `add_commuted_kernel` | Adapt [elementwise add](https://github.com/tile-ai/tilelang/blob/35850c06875bcc84751417982bf7b4d662bafdbc/examples/elementwise/example_elementwise_add.py) (`elementwise_add`, lines 11–33). | Triton: global inputs/output and SSA values. TileLang example: global tensors, CTA shared A/B/C and fragment C; adapt its 2-D tiling to the original complete 1-D FP32 tile and keep both operand orders. |
| [RowWiseSum](https://github.com/VeriInfra-Lab/VeriTile/blob/3a0d07615a31b28e46ceff9660ce44351c425a73/bench/examples/RowWiseSum/RowWiseSum.py) — `row_wise_sum`, `row_wise_sum_reversed` | Written by us. | Global row input and scalar output; planned fragment row/reduction accumulator, no shared storage required. Preserve complete-tile addressing, row stride, FP32 reduction and reversed-lane comparison; do not silently add padding/masks. |
| [RowWiseMax](https://github.com/VeriInfra-Lab/VeriTile/blob/3a0d07615a31b28e46ceff9660ce44351c425a73/bench/examples/RowWiseMax/RowWiseMax.py) — `row_wise_max`, `row_wise_max_inlined` | Written by us. | Global row input/scalar output and planned fragment values/reduction. Preserve nonempty complete-tile contract and inlined comparison; no explicit shared buffer required. |
| [SoftmaxStable](https://github.com/VeriInfra-Lab/VeriTile/blob/3a0d07615a31b28e46ceff9660ce44351c425a73/bench/examples/SoftmaxStable/SoftmaxStable.py) — `naive_softmax`, `stable_softmax` | Written by us. | Global FP32 input/BF16 output and planned fragment intermediates. Preserve naive versus max-subtracted exponentiation and output rounding; the related online example below is not a drop-in counterpart. |
| [OnlineSoftmax](https://github.com/VeriInfra-Lab/VeriTile/blob/3a0d07615a31b28e46ceff9660ce44351c425a73/bench/examples/OnlineSoftmax/OnlineSoftmax.py) — `stableSoftmaxKernel`, `onlineSoftmaxKernel` | Written by us; related upstream online example is compared below, not directly reused. | Global FP32 input/output; planned fragment row plus local scalar running max/sum. Preserve scalar input traversal, both passes and the separate arithmetic graph of the online variant. |
| [FusedSiLU](https://github.com/VeriInfra-Lab/VeriTile/blob/3a0d07615a31b28e46ceff9660ce44351c425a73/bench/examples/FusedSiLU/FusedSiLU.py) — `fused_silu`; `silu_gate` → `silu_activation` → `silu_residual` | Written by us. | Global inputs/output and global intermediate buffers for the three-launch comparison; planned fragment arithmetic. Preserve residual + silu(x * gate), FP64 input/scratch and BF16 output, not ordinary silu(x) alone. |
| [FusedSwiglu](https://github.com/VeriInfra-Lab/VeriTile/blob/3a0d07615a31b28e46ceff9660ce44351c425a73/bench/examples/FusedSwiglu/FusedSwiglu.py) — `swiglu_fused`; `silu_step` → `mul_step` | Written by us. | Global FP32 inputs, BF16 intermediate scratch/output and planned fragment arithmetic. Preserve masks, BF16 intermediate rounding, re-load conversion and ordered two-launch comparison. |
| [StableLogSumExp](https://github.com/VeriInfra-Lab/VeriTile/blob/3a0d07615a31b28e46ceff9660ce44351c425a73/bench/examples/StableLogSumExp/StableLogSumExp.py) — `direct_log_sum_exp`, `guarded_log_sum_exp` | Written by us. | Global FP32 input/BF16 scalar output; planned fragment reductions/scalars. Retain both guarded fallback branches and conversions, not just a textbook unguarded log-sum-exp formula. |
| [Welford](https://github.com/VeriInfra-Lab/VeriTile/blob/3a0d07615a31b28e46ceff9660ce44351c425a73/bench/examples/Welford/Welford.py) — `twopassWelfordKernel`, `onlineWelfordKernel` | Written by us. | Global FP32 row and two BF16 scalar outputs at address zero; planned fragment row and scalar statistics. Preserve count conversions and per-program contract; do not reinterpret this as an arbitrary multi-row launch writing distinct outputs. |

### Why not reuse similarly named examples verbatim?

- Observed: TileLang [online softmax](https://github.com/tile-ai/tilelang/blob/35850c06875bcc84751417982bf7b4d662bafdbc/examples/online_softmax/online_softmax.py)
  (`softmax_kernel`, lines 7–46) uses fragment tiles, a chunked log-sum-exp
  recurrence and `exp2`/`log2`, with FP16 defaults and FP32 accumulation. The
  selected VeriTile online variant uses scalar running max/sum with its own
  arithmetic order. Inference: shared purpose does not establish numerical or
  program equivalence; the proposal writes the matching counterpart explicitly.
- Observed: TileLang [layer normalization](https://github.com/tile-ai/tilelang/blob/35850c06875bcc84751417982bf7b4d662bafdbc/examples/norm/layernorm.py)
  (`_layernorm_fwd`, lines 6–66) combines sum and sum-of-squares, shared/fragment
  buffers, affine parameters and reciprocal standard deviation. Inference: it
  is not the selected Welford pair. The proposal does not substitute it.

## Six provisional benchmark representatives

[#130 (I10.1)](https://github.com/dnyBLUESKY007/VeriTile/issues/130) requires every
one of the 183 tasks to be assigned to exactly one of **A1, A2, B1, B2, C1, C2**.
[#233 (K-L.6)](https://github.com/dnyBLUESKY007/VeriTile/issues/233) explicitly
uses those families. This document therefore uses those six families, not a
new taxonomy inferred from filenames. The individual family issues below define
the intended categories. Their assignment file is not delivered at the inspection
baseline, so the selections are provisional and must be reconciled with #130.
The six representatives select the named original kernel entrypoints, not every
optimization candidate in their source files (for example, A2 also contains a
`guarded_reciprocal_kernel` comparison). Required branches of each selected
original remain in scope. The nine worked examples above retain paired variants.
One representative is not coverage of every kernel, operation, dtype or branch
of every task in that family.

| Family authority | Proposed task and observed Triton entry | TileLang source choice (proposed) | Storage and required correspondence |
| --- | --- | --- | --- |
| [A1: elementwise, activation, sampling](https://github.com/dnyBLUESKY007/VeriTile/issues/131) | [vector_addition](https://github.com/VeriInfra-Lab/VeriTileBench/blob/39f6669cd2d714c31940256a326b8b0373f5bb7f/bench/tritonbench_g/vector_addition/vector_addition.py) — `add_kernel` (lines 15–28) | Adapt the same [elementwise add](https://github.com/tile-ai/tilelang/blob/35850c06875bcc84751417982bf7b4d662bafdbc/examples/elementwise/example_elementwise_add.py) starting point. | Global input/output; shared+fragment TileLang staging. Preserve the FP32 contiguous host contract and masked tail, which distinguishes this task from unmasked VectorAdd. |
| [A2: normalization, softmax, reductions, losses](https://github.com/dnyBLUESKY007/VeriTile/issues/132) | [softmax_triton3](https://github.com/VeriInfra-Lab/VeriTileBench/blob/39f6669cd2d714c31940256a326b8b0373f5bb7f/bench/tritonbench_g/softmax_triton3/softmax_triton3.py) — `softmax_kernel` (lines 8–31) | Written by us. | Global FP32 input/output and optional mask; planned fragment reduction intermediates. Preserve optional additive mask after row-max subtraction, masked tail and last-axis host contract. |
| [B1: matmul and quantization](https://github.com/dnyBLUESKY007/VeriTile/issues/133) | [matmul_triton1](https://github.com/VeriInfra-Lab/VeriTileBench/blob/39f6669cd2d714c31940256a326b8b0373f5bb7f/bench/tritonbench_g/matmul_triton1/matmul_triton1.py) — `matmul_kernel` (lines 6–49) | Written by us; related TileLang GEMM is not directly reused. | Global FP32 A/B/C; planned shared operand tiles and fragment accumulator. Preserve strided indexing, masked tails, K-loop and arithmetic contract; default FP16 upstream GEMM is not a numerical substitute. |
| [B2: copy, indexing, cumsum, cache](https://github.com/dnyBLUESKY007/VeriTile/issues/134) | [matrix_transpose](https://github.com/VeriInfra-Lab/VeriTileBench/blob/39f6669cd2d714c31940256a326b8b0373f5bb7f/bench/tritonbench_g/matrix_transpose/matrix_transpose.py) — `kernel` (lines 9–24) | Written by us. | Global FP16 matrix/output; planned fragment tile (shared staging optional, not required by the selection). Preserve the single-tile power-of-two shape contract and explicit transpose. |
| [C1: attention](https://github.com/dnyBLUESKY007/VeriTile/issues/135) | [attention_fwd_triton2](https://github.com/VeriInfra-Lab/VeriTileBench/blob/39f6669cd2d714c31940256a326b8b0373f5bb7f/bench/tritonbench_g/attention_fwd_triton2/attention_fwd_triton2.py) — `_attn_fwd_inner` (lines 8–38), `_attn_fwd` (lines 42–86) | Written by us; related flash-attention example is not directly reused. | Global Q/K/V, scaling tensors and output; planned shared Q/K/V staging and fragment online accumulators. Preserve per-block scales, masked fixed-width dimensions, accumulation/conversion order and launch envelope. |
| [C2: recurrent, chunked sequence, rotary, concurrent tasks](https://github.com/dnyBLUESKY007/VeriTile/issues/136) | [rotary_transform](https://github.com/VeriInfra-Lab/VeriTileBench/blob/39f6669cd2d714c31940256a326b8b0373f5bb7f/bench/tritonbench_g/rotary_transform/rotary_transform.py) — `rotary_kernel` (lines 7–70) | Written by us. | Global input/output, cosine/sine and optional sequence metadata; planned fragment arithmetic. Preserve variable-length versus fixed-length, offset modes, interleaved layout and conjugate branches; select the task rather than silently only its simplest branch. |

Observed comparison sources at the TileLang pin:
[GEMM](https://github.com/tile-ai/tilelang/blob/35850c06875bcc84751417982bf7b4d662bafdbc/examples/gemm/example_gemm.py) (`matmul`, lines 6–26) stages shared
operands and a fragment accumulator but defaults to FP16 input;
[flash attention](https://github.com/tile-ai/tilelang/blob/35850c06875bcc84751417982bf7b4d662bafdbc/examples/flash_attention/example_mha_fwd_bhsd.py)
(`flashattn`, beginning at line 23) uses FP16 tensors, shared Q/K/V/output,
fragment scores/accumulators and online reductions. These support the proposed
storage design only. Neither establishes exact correspondence to the selected
FP32 GEMM or scaled-attention task, so neither is presented as a ready-made port.

## Confirmation and downstream gates

The concrete question for the user is: **approve these nine worked-example
pairs plus the six named provisional representatives, the proposed benchmark
pin, and the source choices above?** In particular, the proposal includes the
comparison variants, retains Welford's per-program output restriction, and does
not treat a similar TileLang example as the same floating-point program.

- User confirmation: **pending**. This document does not sign on the user's behalf.
- I10.1 assignment reconciliation: **pending #130**. Recheck all six selections
  against the delivered assignment; replace a misassigned representative through
  review, or record why TileLang cannot express a family under #233. No claim
  that all 183 tasks are assigned or covered is made here.
- Porting, frontend expressibility of every required construct, reduction-order
  agreement, dtype/API adaptation, proof and GPU comparison: **not performed**.
  Shared storage does not itself establish any synchronization guarantee.
- For an implementation ticket, read the selected `Contract.lean`, `IO.lean`,
  `RealCorrect.lean` and `FPEquiv.lean` alongside its Python source at the same
  pin. Selecting source files is not an audit of those theorem surfaces.
- No source file was vendored, imported or executed; no GPU, Lean build or
  numerical verdict belongs to this document-only selection.
