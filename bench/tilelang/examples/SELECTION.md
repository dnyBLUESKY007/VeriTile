# K-L.1: proposed TileLang example selection

Status: **list and sources confirmed by the user on 2026-11-12; #130 reconciliation pending**. Access date for all sources and
tracker observations in the original selection: **2026-11-12**; citation and
candidate-name audit for [#424](https://github.com/dnyBLUESKY007/VeriTile/issues/424):
**2026-11-13**. This is the document deliverable of
[#228](https://github.com/dnyBLUESKY007/VeriTile/issues/228), not implementation,
a proof result, or completion of the #130 reconciliation gate.

## Sources and interpretation

| Source | Fixed revision and role |
| --- | --- |
| VeriTile upstream | `3a0d07615a31b28e46ceff9660ce44351c425a73`: the map's upstream pin; the nine worked-example Python counterparts below still exist under `bench/examples/` at this revision |
| TileLang | `35850c06875bcc84751417982bf7b4d662bafdbc`: the map's TileLang pin; related upstream examples read as text only |
| VeriTileBench | `39f6669cd2d714c31940256a326b8b0373f5bb7f`: benchmark-source pin proposed in the original selection and confirmed by the user on 2026-11-12; explicitly pinned by the map at the 2026-11-13 readback |
| Fork | `a0ba977236322702b1d0d8b891a34a4e1c9b2074`: inspection baseline; `experiments/interp_gpu/specs/ASSIGNMENT.tsv` is absent |

The [map](https://github.com/dnyBLUESKY007/VeriTile/issues/1) owns the scope and
hard constraints. A benchmark package's VeriTile dependency revision is not a
pin for the benchmark's own source. The original selection proposed a source pin;
the map now records that same benchmark revision (read on 2026-11-13).

**Observed** below means source syntax, an explicitly identified module-docstring
contract, or an existing issue requirement. **Proposed** means a selection or
intended implementation decision. Storage terms describe front-end allocations,
not synchronization scope, physical register placement, or a compiler guarantee.
The TileLang storage column is planned unless a named upstream example is reused.
Triton intermediate SSA values are expressions, not a fifth storage category.

| Storage term | Pinned TileLang API evidence and meaning here |
| --- | --- |
| global | Input/output tensors or intermediate buffers passed between launches; [alloc_global, lines 171–190](https://github.com/tile-ai/tilelang/blob/35850c06875bcc84751417982bf7b4d662bafdbc/tilelang/language/allocate.py#L171-L190) also exposes explicit global workspace allocation. This selection does not require that allocator for host-provided tensors. |
| shared | [alloc_shared, lines 49–64](https://github.com/tile-ai/tilelang/blob/35850c06875bcc84751417982bf7b4d662bafdbc/tilelang/language/allocate.py#L49-L64), with default scope `shared.dyn`; used for tiles shared within a CTA. |
| fragment | [alloc_fragment, lines 81–92](https://github.com/tile-ai/tilelang/blob/35850c06875bcc84751417982bf7b4d662bafdbc/tilelang/language/allocate.py#L81-L92), with default scope `local.fragment`; used for tile arithmetic/reductions. |
| local | [alloc_local, lines 67–78](https://github.com/tile-ai/tilelang/blob/35850c06875bcc84751417982bf7b4d662bafdbc/tilelang/language/allocate.py#L67-L78), default scope `local`, and scalar [alloc_var signature/docstring, lines 103–126](https://github.com/tile-ai/tilelang/blob/35850c06875bcc84751417982bf7b4d662bafdbc/tilelang/language/allocate.py#L103-L126), default scope `local.var`. Both OnlineSoftmax and Welford use this term for planned loop-carried scalars. |

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
| [VectorAdd](https://github.com/VeriInfra-Lab/VeriTile/blob/3a0d07615a31b28e46ceff9660ce44351c425a73/bench/examples/VectorAdd/VectorAdd.py) — `add_kernel`, `add_commuted_kernel` | Adapt [elementwise add](https://github.com/tile-ai/tilelang/blob/35850c06875bcc84751417982bf7b4d662bafdbc/examples/elementwise/example_elementwise_add.py) (`elementwise_add`, lines 11–32). | Triton: global inputs/output. TileLang example: global tensors, CTA shared A/B/C and fragment C; adapt its 2-D tiling to the original complete 1-D FP32 tile and keep both operand orders. Dtype contract: [module docstring, line 1](https://github.com/VeriInfra-Lab/VeriTile/blob/3a0d07615a31b28e46ceff9660ce44351c425a73/bench/examples/VectorAdd/VectorAdd.py#L1). |
| [RowWiseSum](https://github.com/VeriInfra-Lab/VeriTile/blob/3a0d07615a31b28e46ceff9660ce44351c425a73/bench/examples/RowWiseSum/RowWiseSum.py) — `row_wise_sum`, `row_wise_sum_reversed` | Written by us. | Global row input and scalar output; planned fragment row/reduction accumulator, no shared storage required. Preserve complete-tile addressing, row stride, FP32 reduction and reversed-lane comparison; do not silently add padding/masks. Dtype contract: [module docstring, lines 1–7](https://github.com/VeriInfra-Lab/VeriTile/blob/3a0d07615a31b28e46ceff9660ce44351c425a73/bench/examples/RowWiseSum/RowWiseSum.py#L1-L7). |
| [RowWiseMax](https://github.com/VeriInfra-Lab/VeriTile/blob/3a0d07615a31b28e46ceff9660ce44351c425a73/bench/examples/RowWiseMax/RowWiseMax.py) — `row_wise_max`, `row_wise_max_inlined` | Written by us. | Global row input/scalar output and planned fragment values/reduction. Preserve nonempty complete-tile contract and inlined comparison; no explicit shared buffer required. Dtype contract: [module docstring, lines 1–7](https://github.com/VeriInfra-Lab/VeriTile/blob/3a0d07615a31b28e46ceff9660ce44351c425a73/bench/examples/RowWiseMax/RowWiseMax.py#L1-L7). |
| [SoftmaxStable](https://github.com/VeriInfra-Lab/VeriTile/blob/3a0d07615a31b28e46ceff9660ce44351c425a73/bench/examples/SoftmaxStable/SoftmaxStable.py) — `naive_softmax`, `stable_softmax` | Written by us. | Global FP32 input/BF16 output and planned fragment intermediates. Preserve naive versus max-subtracted exponentiation and output rounding; the related online example below is not a drop-in counterpart. Dtype contract: [module docstring, lines 1–7](https://github.com/VeriInfra-Lab/VeriTile/blob/3a0d07615a31b28e46ceff9660ce44351c425a73/bench/examples/SoftmaxStable/SoftmaxStable.py#L1-L7). |
| [OnlineSoftmax](https://github.com/VeriInfra-Lab/VeriTile/blob/3a0d07615a31b28e46ceff9660ce44351c425a73/bench/examples/OnlineSoftmax/OnlineSoftmax.py) — `stableSoftmaxKernel`, `onlineSoftmaxKernel` | Written by us; related upstream online example is compared below, not directly reused. | Global FP32 input/output; planned fragment row plus local scalar running max/sum. Preserve scalar input traversal, both passes and the separate arithmetic graph of the online variant. Dtype contract: [module docstring, lines 1–8](https://github.com/VeriInfra-Lab/VeriTile/blob/3a0d07615a31b28e46ceff9660ce44351c425a73/bench/examples/OnlineSoftmax/OnlineSoftmax.py#L1-L8). |
| [FusedSiLU](https://github.com/VeriInfra-Lab/VeriTile/blob/3a0d07615a31b28e46ceff9660ce44351c425a73/bench/examples/FusedSiLU/FusedSiLU.py) — `fused_silu`; `silu_gate` → `silu_activation` → `silu_residual` | Written by us. | Global inputs/output and global intermediate buffers for the three-launch comparison; planned fragment arithmetic. Preserve residual + silu(x * gate), FP64 input/scratch and BF16 output, not ordinary silu(x) alone. Dtype contract: [module docstring, lines 1–7](https://github.com/VeriInfra-Lab/VeriTile/blob/3a0d07615a31b28e46ceff9660ce44351c425a73/bench/examples/FusedSiLU/FusedSiLU.py#L1-L7). |
| [FusedSwiglu](https://github.com/VeriInfra-Lab/VeriTile/blob/3a0d07615a31b28e46ceff9660ce44351c425a73/bench/examples/FusedSwiglu/FusedSwiglu.py) — `swiglu_fused`; `silu_step` → `mul_step` | Written by us. | Global FP32 inputs, BF16 intermediate scratch/output and planned fragment arithmetic. Preserve masks, BF16 intermediate rounding, re-load conversion and ordered two-launch comparison. Dtype contract: [module docstring, lines 1–5](https://github.com/VeriInfra-Lab/VeriTile/blob/3a0d07615a31b28e46ceff9660ce44351c425a73/bench/examples/FusedSwiglu/FusedSwiglu.py#L1-L5). |
| [StableLogSumExp](https://github.com/VeriInfra-Lab/VeriTile/blob/3a0d07615a31b28e46ceff9660ce44351c425a73/bench/examples/StableLogSumExp/StableLogSumExp.py) — `direct_log_sum_exp`, `guarded_log_sum_exp` | Written by us. | Global FP32 input/BF16 scalar output; planned fragment reductions/scalars. Retain both guarded fallback branches and conversions, not just a textbook unguarded log-sum-exp formula. Dtype contract: [module docstring, lines 1–7](https://github.com/VeriInfra-Lab/VeriTile/blob/3a0d07615a31b28e46ceff9660ce44351c425a73/bench/examples/StableLogSumExp/StableLogSumExp.py#L1-L7). |
| [Welford](https://github.com/VeriInfra-Lab/VeriTile/blob/3a0d07615a31b28e46ceff9660ce44351c425a73/bench/examples/Welford/Welford.py) — `twopassWelfordKernel`, `onlineWelfordKernel` | Written by us. | Global FP32 row and two BF16 scalar outputs at address zero; planned fragment row and local scalar statistics. Preserve count conversions and per-program contract; do not reinterpret this as an arbitrary multi-row launch writing distinct outputs. Dtype contract: [module docstring, lines 1–9](https://github.com/VeriInfra-Lab/VeriTile/blob/3a0d07615a31b28e46ceff9660ce44351c425a73/bench/examples/Welford/Welford.py#L1-L9). |

### One reuse test, applied to five upstream examples

**Decision rule:** adapt when the source already has the target's arithmetic and
control structure, so edits can be confined to tensor rank/shape, dtype arguments,
indexing/boundary predicates, storage staging and the named operand-order variant.
Write the counterpart when matching the target requires replacing a reduction or
loop-carried recurrence, introducing a different contraction operation, or changing
its scale/branch/conversion sequence. A dtype default alone never decides this.
This is a source-construction criterion, not a numerical equivalence theorem;
even an adapted source still needs its correspondence and proof.

| Upstream example at the TileLang pin | Application of the same test |
| --- | --- |
| [elementwise add](https://github.com/tile-ai/tilelang/blob/35850c06875bcc84751417982bf7b4d662bafdbc/examples/elementwise/example_elementwise_add.py#L11-L32), `elementwise_add`, lines 11–32 | **Adapt** for VectorAdd and A1. Lines 27–28 already perform one independent addition per element. Rank/tiling, explicit dtype arguments, boundary predicates and operand order can change without introducing a recurrence or replacing arithmetic. Shared/fragment staging at lines 20–30 is a starting point, not a proof that boundary handling is already identical. |
| [online softmax](https://github.com/tile-ai/tilelang/blob/35850c06875bcc84751417982bf7b4d662bafdbc/examples/online_softmax/online_softmax.py#L7-L45), `softmax_kernel`, lines 7–45 | **Write** OnlineSoftmax, SoftmaxStable and A2. The chunked log-sum-exp recurrence with `exp2`/`log2` is neither the worked example's scalar two-pass running max/sum graph nor A2's single-row max/exp/sum graph with an optional additive mask. Changing that recurrence exceeds adaptation; FP16 defaults are not the reason. |
| [layer normalization](https://github.com/tile-ai/tilelang/blob/35850c06875bcc84751417982bf7b4d662bafdbc/examples/norm/layernorm.py#L6-L58), `_layernorm_fwd`, lines 6–58 | **Write** Welford. Sum and sum-of-squares, affine parameters and reciprocal standard deviation do not preserve the selected two-pass/online mean-and-variance pair. Replacing these operations and the online recurrence fails the reuse test. |
| [GEMM](https://github.com/tile-ai/tilelang/blob/35850c06875bcc84751417982bf7b4d662bafdbc/examples/gemm/example_gemm.py#L6-L26), `matmul`, lines 6–26 | **Write** B1. Its pipelined tile copies and `T.gemm` (lines 19–22) differ from [matmul_triton1, lines 6–49](https://github.com/VeriInfra-Lab/VeriTileBench/blob/39f6669cd2d714c31940256a326b8b0373f5bb7f/bench/tritonbench_g/matmul_triton1/matmul_triton1.py#L6-L49): explicit stride arithmetic, loads masked on M/N/K tails, and accumulation of `tl.dot(..., allow_tf32=False)` in a K loop. Strides/tails alone could be adaptation, but preserving the contraction and accumulation contract requires an explicit new correspondence; setting dtype to FP32 does not supply it. |
| [flash attention](https://github.com/tile-ai/tilelang/blob/35850c06875bcc84751417982bf7b4d662bafdbc/examples/flash_attention/example_mha_fwd_bhsd.py#L23-L102), `flashattn`, lines 23–102 | **Write** C1. Its dimension-derived scale (line 24), `T.gemm` operations (75, 95) and online update (77–98) do not implement the selected task's Q/K scale loads and products, fixed-width dimension masks, FP16 conversions and second dot's explicit FP16 result (selected helper lines 17–33). These alter the arithmetic/conversion sequence, not just tensor parameters. |

All five applications retain the confirmed source choices; no selection conflict
was found. The shared/fragment allocations in GEMM and flash attention inform the
planned storage only; they do not supply matching arithmetic or synchronization.

## Six provisional benchmark representatives

[#130 (I10.1)](https://github.com/dnyBLUESKY007/VeriTile/issues/130) requires every
one of the 183 tasks to be assigned to exactly one of **A1, A2, B1, B2, C1, C2**.
[#233 (K-L.6)](https://github.com/dnyBLUESKY007/VeriTile/issues/233) explicitly
uses those families. This document therefore uses those six families, not a
new taxonomy inferred from filenames. The individual family issues below define
the intended categories. Their assignment file is not delivered at the inspection
baseline, so the selections are provisional and must be reconciled with #130.
The six representatives select the named original entrypoints and all their
required branches. They do **not** select every comparison in the source file.
The nine worked examples above still retain both sides of each pair. The four
family files with comparisons leave these additional entrypoints outside this
selection (all links use the benchmark pin):

| Selected task | Excluded comparison entrypoints |
| --- | --- |
| vector_addition | [add_commuted_kernel, lines 32–45](https://github.com/VeriInfra-Lab/VeriTileBench/blob/39f6669cd2d714c31940256a326b8b0373f5bb7f/bench/tritonbench_g/vector_addition/vector_addition.py#L32-L45) |
| softmax_triton3 | [guarded_reciprocal_kernel, lines 86–115](https://github.com/VeriInfra-Lab/VeriTileBench/blob/39f6669cd2d714c31940256a326b8b0373f5bb7f/bench/tritonbench_g/softmax_triton3/softmax_triton3.py#L86-L115) |
| matrix_transpose | [kernel_index_commuted, lines 28–44](https://github.com/VeriInfra-Lab/VeriTileBench/blob/39f6669cd2d714c31940256a326b8b0373f5bb7f/bench/tritonbench_g/matrix_transpose/matrix_transpose.py#L28-L44) |
| attention_fwd_triton2 | [_attn_fwd_inner_index_shift, lines 160–190, and its _attn_fwd_index_shift caller, lines 194–238](https://github.com/VeriInfra-Lab/VeriTileBench/blob/39f6669cd2d714c31940256a326b8b0373f5bb7f/bench/tritonbench_g/attention_fwd_triton2/attention_fwd_triton2.py#L160-L238) |

For #233, “proved against the same specification” therefore remains an obligation
to connect each TileLang original to the selected task's existing independent
specification, IO and contract, including its masks, dtypes and full branch domain.
It is not permission to weaken that specification to fit a similar algorithm.
This selection alone proves neither original-to-TileLang correspondence nor the
excluded comparison kernels' equivalence. Porting those extra candidates and
re-establishing their `FPEquiv` results is outside the six-original selection;
#233 review must check the specification connection separately. One representative
is not coverage of every kernel, operation, dtype or branch in a family.

| Family authority | Proposed task and observed Triton entry | TileLang source choice (proposed) | Storage and required correspondence |
| --- | --- | --- | --- |
| [A1: elementwise, activation, sampling](https://github.com/dnyBLUESKY007/VeriTile/issues/131) | [vector_addition](https://github.com/VeriInfra-Lab/VeriTileBench/blob/39f6669cd2d714c31940256a326b8b0373f5bb7f/bench/tritonbench_g/vector_addition/vector_addition.py) — `add_kernel` (lines 15–28) | Adapt the same [elementwise add](https://github.com/tile-ai/tilelang/blob/35850c06875bcc84751417982bf7b4d662bafdbc/examples/elementwise/example_elementwise_add.py) starting point. | Global input/output; shared+fragment TileLang staging. Preserve the FP32 contiguous host contract and masked tail, which distinguishes this task from unmasked VectorAdd. |
| [A2: normalization, softmax, reductions, losses](https://github.com/dnyBLUESKY007/VeriTile/issues/132) | [softmax_triton3](https://github.com/VeriInfra-Lab/VeriTileBench/blob/39f6669cd2d714c31940256a326b8b0373f5bb7f/bench/tritonbench_g/softmax_triton3/softmax_triton3.py) — `softmax_kernel` (lines 8–31) | Written by us. | Global FP32 input/output and optional mask; planned fragment reduction intermediates. Preserve optional additive mask after row-max subtraction, masked tail and last-axis host contract. |
| [B1: matmul and quantization](https://github.com/dnyBLUESKY007/VeriTile/issues/133) | [matmul_triton1](https://github.com/VeriInfra-Lab/VeriTileBench/blob/39f6669cd2d714c31940256a326b8b0373f5bb7f/bench/tritonbench_g/matmul_triton1/matmul_triton1.py) — `matmul_kernel` (lines 6–49) | Written by us; related TileLang GEMM is not directly reused. | Global FP32 A/B/C; planned shared operand tiles and fragment accumulator. Preserve strided indexing, masked tails, K-loop and arithmetic contract; the intrinsic/loop correspondence must be written explicitly (see reuse test above). |
| [B2: copy, indexing, cumsum, cache](https://github.com/dnyBLUESKY007/VeriTile/issues/134) | [matrix_transpose](https://github.com/VeriInfra-Lab/VeriTileBench/blob/39f6669cd2d714c31940256a326b8b0373f5bb7f/bench/tritonbench_g/matrix_transpose/matrix_transpose.py) — `kernel` (lines 9–24) | Written by us. | Global FP16 matrix/output; planned fragment tile, no shared allocation. Preserve the single-tile power-of-two shape contract and explicit transpose. |
| [C1: attention](https://github.com/dnyBLUESKY007/VeriTile/issues/135) | [attention_fwd_triton2](https://github.com/VeriInfra-Lab/VeriTileBench/blob/39f6669cd2d714c31940256a326b8b0373f5bb7f/bench/tritonbench_g/attention_fwd_triton2/attention_fwd_triton2.py) — `_attn_fwd_inner` (lines 8–38), `_attn_fwd` (lines 42–86) | Written by us; related flash-attention example is not directly reused. | Global Q/K/V, scaling tensors and output; planned shared Q/K/V staging and fragment online accumulators. Preserve per-block scales, masked fixed-width dimensions, accumulation/conversion order and launch envelope. |
| [C2: recurrent, chunked sequence, rotary, concurrent tasks](https://github.com/dnyBLUESKY007/VeriTile/issues/136) | [rotary_transform](https://github.com/VeriInfra-Lab/VeriTileBench/blob/39f6669cd2d714c31940256a326b8b0373f5bb7f/bench/tritonbench_g/rotary_transform/rotary_transform.py) — `rotary_kernel` (lines 7–70) | Written by us. | Global input/output, cosine/sine and optional sequence metadata; planned fragment arithmetic. Preserve variable-length versus fixed-length, offset modes, interleaved layout and conjugate branches; select the task rather than silently only its simplest branch. |

### Candidate-name audit and rejection reasons

The source of every name below is the [183-directory benchmark tree at the fixed
pin](https://github.com/VeriInfra-Lab/VeriTileBench/tree/39f6669cd2d714c31940256a326b8b0373f5bb7f/bench/tritonbench_g),
read on 2026-11-13. Apply each family issue's literal glob or exact name to these
directories. For its prose terms, this audit makes the name expansion explicit:
`*dropout*`, `*sampling*`, `*reduction*`, `*argmax*`, `*transpose*`,
`*sparse*att*` and `*parallel*att*` (the latter two include both `attn` and
`attention` spellings). These are exhaustive **name matches under those rules**,
not a semantic classification of all 183 tasks. Matches can overlap, for example
`masked_select`, `spinning_lock_reduction` and `chunk_cumsum_kernel`; only #130
will assign each task to one family. Counts include the selected task; tables
list every rejected match once within each family. Rejection is a selection
tradeoff, not an unsupported-frontend or failed-proof verdict. Criteria refer to
the selected source ranges above; no ranking of uninspected theorem surfaces is
claimed.

#### A1: 20 matches, select `vector_addition`

**Criterion:** Choose a two-input addition with a masked tail: it gives a direct contrast with the unmasked VectorAdd worked example and shares the elementwise starting point.

| Rejected name group | All rejected candidates | Reason |
| --- | --- | --- |
| `add_*`, `vector_addition*` | `add_example`, `add_value`, `vector_addition_custom` | One addition representative is enough; vector_addition supplies the masked contiguous FP32 contract already cited above, so additional addition siblings are not selected. |
| `masked_*` | `masked_add_cuda`, `masked_select` | Keep the representative focused on two-input addition with an extent mask; masked-name tasks do not add a separate selection slot. |
| `relu_*`, `fused_activation`, `sin_*`, `pow_*`, `swiglu_*`, `geglu_*` | `fused_activation`, `geglu_tanh_triton`, `pow_scalar_tensor`, `relu_strided_buffer`, `relu_triton_kernel`, `sin_computation`, `sin_kernel`, `swiglu_backward`, `swiglu_fwd`, `swiglu_triton` | Activation and other arithmetic subgroups would broaden the first representative beyond the chosen addition graph. |
| `*dropout*`, `*sampling*` | `dropout_triton`, `multinomial_sampling`, `seeded_dropout`, `uniform_sampling` | Randomized/sampling subgroups are not needed for the chosen deterministic addition contrast. |

#### A2: 32 matches, select `softmax_triton3`

**Criterion:** Choose one last-axis row softmax that exercises both reduction and the optional additive mask, retaining the mask placement after max subtraction.

| Rejected name group | All rejected candidates | Reason |
| --- | --- | --- |
| `softmax_triton1`, `softmax_triton2` | `softmax_triton1`, `softmax_triton2` | Their original row softmax kernels lack the optional additive mask: see softmax_triton1 lines 7–27 and softmax_triton2 lines 8–28 below. The third sibling exercises that additional branch. |
| `softmax*` | `softmax_flaggems`, `softmax_optimize`, `softmax_reducev` | Other softmax siblings receive no second slot; the selected third sibling already supplies the reduction-plus-optional-mask contrast. |
| `layer_norm*`, `rms*norm*`, `l2_norm*` | `l2_norm_bwd`, `l2_norm_triton1`, `l2_norm_triton2`, `layer_norm_fwd`, `layer_norm_liger`, `layer_norm_ops`, `layer_norm_triton`, `layer_norm_welfold`, `rms_norm_triton`, `rmsnorm_fused`, `rmsnorm_fused_llama`, `rmsnorm_implementation`, `rmsnorm_triton` | Normalization subgroups are deferred because this representative targets softmax reduction and its optional mask, not a normalization family census. |
| `log_softmax`, `logsumexp*`, `cross_entropy*`, `kldiv*` | `cross_entropy1`, `cross_entropy2`, `cross_entropy_ops`, `kldiv_compute`, `kldiv_ops`, `kldiv_triton`, `log_softmax`, `logsumexp_fwd` | Log-domain and loss subgroups would require a different output specification than the selected softmax. |
| `*reduction*`, `*argmax*` | `matrix_reduction`, `max_reduction`, `mean_reduction`, `spinning_lock_reduction`, `triton_argmax` | Standalone reduction/argmax subgroups are not selected because the chosen softmax already combines reductions with normalization and a mask. |

#### B1: 30 matches, select `matmul_triton1`

**Criterion:** Choose the explicit two-input matrix product with stride arithmetic, masked M/N/K tails and an inspectable K loop, before adding quantization, batching or fused operations.

| Rejected name group | All rejected candidates | Reason |
| --- | --- | --- |
| `matmul_*` | `matmul_dequant_int4`, `matmul_dequantize`, `matmul_dequantize_int4`, `matmul_kernel`, `matmul_leakyrelu`, `matmul_leakyrelu_fp8`, `matmul_persistent_triton`, `matmul_tma`, `matmul_triton2`, `matmul_triton_autotune` | Keep the cited explicit K-loop matmul_triton1 as the single matrix-product representative; other matmul-named alternatives are not claimed inferior or equivalent. |
| `bmm_*` | `bmm_chunk_bwd`, `bmm_chunk_fwd`, `bmm_optimized` | Batched-matmul candidates are outside the selected two-matrix contrast. |
| `int4_*`, `int8_*`, `dequantize_*`, `quantize_*` | `dequantize_matmul`, `dequantize_rowwise`, `int4_matmul`, `int8_dequant_matmul`, `int8_matmul_kernel`, `int8_matmul_quantization`, `int8_quantization`, `quantize_copy_kv`, `quantize_global`, `quantize_kv_copy`, `quantize_kv_transform` | Quantization/dequantization subgroups add a different precision or conversion objective to the chosen matrix-product representative. |
| `lora_*`, `bgmv_*`, `sgmv_*`, `triton_conv2d_fwd` | `bgmv_expand_slice`, `bgmv_shrink_kernel`, `lora_expand_gemv`, `sgmv_expand_slice`, `triton_conv2d_fwd` | Adapter/grouped-vector and convolution subgroups are deferred in favor of the selected plain matrix-product structure. |

#### B2: 19 matches, select `matrix_transpose`

**Criterion:** Choose an explicit two-dimensional transpose under a single-tile power-of-two contract, isolating an indexing permutation; plan fragment storage with no shared allocation.

| Rejected name group | All rejected candidates | Reason |
| --- | --- | --- |
| `*transpose*` | `quant_transpose_kernel` | The other transpose-named candidate is not needed for this single permutation slot; matrix_transpose supplies the cited single-tile contract. |
| `kv_cache_*`, `destindex_*`, `embedding_*`, `index_select_*`, `masked_select`, `var_len_copy` | `destindex_copy`, `destindex_copy_kv1`, `destindex_copy_kv2`, `embedding_triton_kernel`, `index_select_bwd`, `index_select_cat`, `kv_cache_copy`, `kv_cache_filling`, `masked_select`, `var_len_copy` | Cache/copy/gather and variable-length subgroups broaden indexing beyond the selected fixed-shape permutation. |
| `*cumsum*`, `adam_update_triton` | `adam_update_triton`, `chunk_cumsum_kernel`, `chunk_cumsum_vector`, `chunked_cumsum_fwd`, `decay_cumsum`, `reversed_cumsum`, `reversed_cumsum_scalar` | Scan/update subgroups change values as well as addresses; the representative deliberately isolates transpose indexing. |

#### C1: 25 matches, select `attention_fwd_triton2`

**Criterion:** Choose an attention original whose helper exposes Q/K block scales, fixed-width dimension masks, online state and explicit precision conversions in one inspected pair of entrypoints.

| Rejected name group | All rejected candidates | Reason |
| --- | --- | --- |
| `attention_*`, `attn_*` | `attention_forward_triton`, `attention_fwd_triton1`, `attention_fwd_triton3`, `attention_kernel`, `attention_kernel_aligned`, `attention_llama`, `attention_score`, `attn_fwd_causal`, `attn_fwd_triton` | Keep attention_fwd_triton2 for the cited scales/masks/conversions combination; the remaining attention/attn siblings receive no additional slot, without asserting that they are equivalent. |
| `context_attn_*`, `flash_*`, `token_attn_*` | `context_attn_bloom`, `context_attn_fwd`, `context_attn_llama`, `context_attn_mistral`, `context_attn_nopad`, `flash_attn`, `flash_decode2_llama`, `flash_decode2_phi`, `token_attn_llama2`, `token_attn_mistral`, `token_attn_reduceV` | Context/flash/token variants are deferred; the selected scaled-forward pair supplies the chosen attention contrast. |
| `*sparse*att*`, `*parallel*att*` | `block_sparse_attn`, `mixed_sparse_attention`, `parallel_attention`, `parallel_retention_attention` | Sparse and parallel attention subgroups introduce alternative scheduling/attention structures beyond the selected representative. |

#### C2: 25 matches, select `rotary_transform`

**Criterion:** Choose a rotary original that exposes fixed/variable lengths, offset modes, interleaving and conjugation branches; retain every branch rather than selecting only its simplest mode.

| Rejected name group | All rejected candidates | Reason |
| --- | --- | --- |
| `rope_*`, `rotary_*` | `rope_backward_transform`, `rope_embedding`, `rope_transform`, `rotary_emb`, `rotary_emb_nopad`, `rotary_transform_ops` | The selected rotary_transform provides the cited branch-rich entrypoint; other rope/rotary siblings receive no second representative slot. |
| `chunk_*`, `fused_recurrent_*`, `fused_rwkv6_kernel`, `diag_ssm_triton` | `chunk_bwd_dqkg`, `chunk_cumsum_kernel`, `chunk_cumsum_vector`, `chunk_delta_fwd`, `chunk_gate_recurrence`, `chunk_gated_attention`, `chunk_gla_fwd`, `chunk_gla_simple`, `chunk_linear_attn`, `chunk_retention`, `chunk_retention_ops`, `diag_ssm_triton`, `fused_recurrent_delta`, `fused_recurrent_hgrn`, `fused_recurrent_retention`, `fused_rwkv6_kernel` | Chunked/recurrent/state-space subgroups require different state recurrences from the selected rotary transformation. |
| `streamk_matmul`, `spinning_lock_reduction` | `spinning_lock_reduction`, `streamk_matmul` | The family issue explicitly flags these for the concurrent line; avoid conflating their single-schedule specs with this rotary representative. |

A2 source comparison: [softmax_triton1, lines 7–27](https://github.com/VeriInfra-Lab/VeriTileBench/blob/39f6669cd2d714c31940256a326b8b0373f5bb7f/bench/tritonbench_g/softmax_triton1/softmax_triton1.py#L7-L27)
and [softmax_triton2, lines 8–28](https://github.com/VeriInfra-Lab/VeriTileBench/blob/39f6669cd2d714c31940256a326b8b0373f5bb7f/bench/tritonbench_g/softmax_triton2/softmax_triton2.py#L8-L28)
normalize the max-subtracted row directly; [softmax_triton3, lines 8–31](https://github.com/VeriInfra-Lab/VeriTileBench/blob/39f6669cd2d714c31940256a326b8b0373f5bb7f/bench/tritonbench_g/softmax_triton3/softmax_triton3.py#L8-L31)
adds the optional mask before exponentiation. That distinction, not the suffix
number or a claimed proof advantage, picks the third sibling.

## Confirmation and downstream gates

On **2026-11-12**, the user confirmed the nine worked-example pairs, six named
provisional representatives, benchmark pin and source choices above, while
explicitly retaining the #130 reconciliation gate. This includes the nine worked
examples' comparison variants and Welford's per-program output restriction; the six family rows select
originals only, as explicitly delimited above; a similar TileLang example
is still not treated as the same floating-point program.

- User confirmation: **recorded, 2026-11-12**. The user confirmed the list and
  sources and explicitly retained the #130 gate; this records their decision.
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
