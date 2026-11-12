# Supported Triton Subset and Semantic Gaps

This document records the Triton-like surface syntax currently embedded in
Lean by VeriTile, the semantic model behind it, and the gaps that are not yet
modeled. It is an artifact-facing contract: if a kernel uses syntax outside
this subset, either the DSL rejects it or the proof is not claiming Triton
semantic fidelity for that feature.

## Scope Summary

VeriTile models a typed Triton-style kernel language, not Python execution and
not full Triton IR. Kernels are written inside Lean using `triton { ... }` and
lower to typed AST nodes:

```lean
Op   : TileDType → TileShape → Type
Stmt : Type
```

The current shape model is ND tiles with outermost-first shape lists. Scalar
values have shape `[]`; a matrix `[M, D]` has index shape
`TileIndex [M, D]`.

## Supported Surface Syntax

### Control and Program IDs

- `tl.program_id(axis)` and `tl.program_id(axis=axis)` where `axis` is a
  numeric literal or `$(n)`. The runtime state stores `pids : Nat → Nat`,
  so every axis is total.
- `tl.num_programs(axis)` and `tl.num_programs(axis=axis)`: the
  launch-grid dimension along `axis`. The runtime state stores
  `numPids : Nat → Nat` (default `1` on every axis — one program per
  unspanned axis); `BlockState.withGridIndex` sets it to the actual grid
  dimensions when instantiating per-program states, so under an ND launch
  (#1 / `Launch.Grid`) `tl.num_programs` reads the true grid extent.
- `tl.for i in $(n) { ... }` and `tl.for i in N { ... }`.
  The loop is operationally modeled and proved through `forLoop_inv`.
- `tl.static_range i in $(n) { ... }` and `tl.static_range i in N { ... }`
  are surface aliases for the same bounded-loop AST. Unroll and pipeline
  attributes are not modeled.
- Block conditional, in two equivalent surface spellings:
  - `if cond { ... }` and `if cond { ... } else { ... }` (Python-style; the
    `if` token here is the DSL conditional, distinct from Lean's term-level `if`).
  - `tl.if cond { ... }` and `tl.if cond { ... } else { ... }` (explicit
    `tl.` prefix, also accepted).

  The condition must be a scalar boolean (`Op .bool []`). There is no
  `break` or `continue`; Triton block-skipping patterns should be written
  by negating the skip condition and wrapping the useful body.
  Elementwise conditional selection remains `tl.where`.
- Register assignment:
  - `x = expr` and `x := expr` (single binding)
  - `x, y = e1, e2` and `x, y := e1, e2` (parallel multi-assign);
    the comma-list lengths must match
  - `value, index = tl.max(..., return_indices=True)`
    (tuple-returning op binding)

### Constants and Dtypes

- Real literals: `0`, `1`, `3.5`, etc. lower to the `.real` channel.
- Lean antiquotation: `$(x)` is context-sensitive. Address/shape/index contexts
  lower it to `.nat`; data/scalar contexts lower it to the algorithm data channel.
- `-inf` and the Lean-source spelling `-float("inf")` lower to `Op.negInf`,
  represented internally by `⊥ : WithBot ℝ`.
- `tl.toReal(x)` converts a `.nat` scalar/tile to `.real`.
- `tl.cast(x, tl.float64|tl.float32|tl.float16|tl.bfloat16)` changes the
  floating dtype index. In the current semantic model this preserves the
  underlying `WithBot ℝ` value; it does not model rounding.
- `(x).to(tl.float64|tl.float32|tl.float16|tl.bfloat16)` is accepted as the
  method-style cast spelling. Parentheses around bare identifiers avoid
  Lean's hierarchical-name parser treating `x.to` as one identifier.
- `tl.bitcast(x, tl.uint32|tl.int32|tl.float32)` is modeled in the compute AST
  for 32-bit payload reinterpretation. Constant uint32 bit patterns can project
  to the algorithm layer (`.nat`, `.int`, or finite `.real`, including zeros
  and subnormals); runtime
  bitcasts remain compute-only and make `ComputeKernel.toAlgorithm?` fail.

Supported channels:

- `.real`: modeled as `WithBot ℝ`, mainly to represent `-inf`.
- `.fp32`, `.fp16`, `.bf16`: explicit floating dtype channels, currently
  backed by the same `WithBot ℝ` mathematical carrier as `.real`.
- `.int`: AST-level signed mathematical-integer channel with
  arithmetic/comparison semantics. `tl.int8`, `tl.int16`, `tl.int32`, and
  `tl.int64` all map here. Ordinary arithmetic does not imply fixed-width
  overflow semantics. Explicit signed narrowing to `tl.int32` lowers through
  `Op.intToSigned 32`, whose result is `(BitVec.ofInt 32 value).toInt` in the
  mathematical, rounded and structural evaluators. Same-width i32 casts and
  int8 widening retain their source representability contracts. Bool-to-i32
  conversion produces 0/1. Other signed narrowing and float-to-integer cases
  require their own supported paths and contracts.
- `.nat`: used for offsets, loop counters, sizes, and address arithmetic.
- `.bool`: produced by comparisons and consumed by masks / `tl.where`.

### Tile Construction and Shape Operations

- `tl.arange(n)` and `tl.arange(start, end)`.
  The two-argument form lowers to `start + tl.arange(end - start)`;
  `tl.arange(0, end)` collapses to `tl.arange(end)`.
  Numeric literals and `$(...)` Lean `Nat` meta-expressions are accepted for
  both bounds.
- `tl.full([dims...], value)`.
- `tl.zeros([dims...])`, sugar for `tl.full([dims...], 0)`.
- `e[:, None]` and `e[None, :]` for rank-1 inputs only.
  These lower to `Op.expandDim`.
- `tl.expand_dims(e, axis=N)` and `tl.expand_dims(e, N)` insert a unit axis
  at a literal position for any macro-known rank.
- `tl.trans(e)` swaps the trailing two axes, with any leading axes treated as
  a batch prefix.

### Arithmetic, Comparisons, and Broadcasting

- Arithmetic: `+`, `-`, `*`, `/` on numeric values.
  Mixed-channel arithmetic is rejected by the DSL.
- Integer floor division and remainder: `//` and `%` on `.nat` / `.int`.
- `tl.cdiv(x, y)` on `.nat`, lowered as `(x + y - 1) / y` with the current
  mathematical `Nat` semantics.
- Pointwise comparisons: `<`, `<=`, `==`, `>`, `>=`, `!=` on `.real` or
  `.nat`, producing `.bool`.
- Boolean ops: `tl.logical_and`, `tl.logical_or`, `tl.logical_not`, plus mask
  operator spellings `a & b`, `a | b`, and `~a` on `.bool` values.
- Nat bitwise ops: `&`, `|`, `^`, `<<`, `>>` on the `.nat` channel. `~` is
  not modeled for `.nat`, because mathematical naturals do not have a finite
  width; signed/fixed-width bitwise semantics is deferred to the fixed-width
  integer model.
- Two-argument `tl.max(a, b)` as pointwise max on `.real`.
- `tl.maximum(a, b)` and `tl.minimum(a, b)` as pointwise select-based sugar
  over comparable channels. Branch broadcasting is currently limited to
  scalar-to-tile lifting, matching `tl.where`.
  Signed integer operands stay signed through min/max, `tl.multiple_of`,
  `tl.max_contiguous` and row/column broadcast views. Compiler-hint arguments
  do not change the value's signedness.
- `tl.minimum(a, float("inf"))` retains a distinct `Op.minPosInf` operation.
  The operand must use fp32/fp64 computation; lower-precision operands need
  an explicit cast. The DSL rejects an uncast operand instead of silently
  widening the minimum operation.
  Its real projection is identity on `WithBot ℝ`; FP execution calls the
  backend's opaque `minPosInf` unary operation at the expression's compute
  precision. This spelling does not introduce a standalone positive-infinity
  value into the real carrier, or assume an FP identity for the operation.
- Directed scans: `tl.cumsum`, `tl.cumprod`, and `tl.associative_scan(x, op,
  axis=N)` on `.real` tiles, each accepting `reverse=True/False`:
  `forward` is the prefix fold, `reverse` the suffix fold along the scanned
  axis. The supported associative op names are the closed enum `sum`, `prod`,
  `max`, `min`; arbitrary user functions are not embedded in the AST.
- Index/order ops: `tl.argmax`, `tl.argmin`, and `tl.sort` on `.real` tiles
  with static `axis=N`. Arg ties return the smallest axis index; sort is
  ascending along the selected axis.
- Shape/view ops: `tl.reshape`, `tl.view`, `tl.ravel`, `tl.permute`,
  `tl.flip`, `tl.join`, and projection-form `tl.split(x, 0|1)`. Views are
  modeled as row-major reshape or typed index remaps. `tl.split` is exposed
  as two projections because the Lean DSL does not yet have tuple
  destructuring syntax for `a, b = tl.split(x)`.
- Broadcasting is ND and follows the current `Broadcast` witness:
  same dimension, scalar-to-tile, or dimension `1` expanded to the other side.
  The DSL constructs the broadcast proof syntactically, so equivalent but
  non-identical dimension expressions may still need to be written in a
  matching form.
- Elementwise selection: `tl.where(cond, a, b)`.
  The condition must be `.bool`, the branches must have the same dtype, and
  scalar-to-tile lifting is accepted. Non-scalar operands must already have
  the same surface shape; use the supported unit-axis slicers to make shapes
  agree before calling `tl.where`.

### Unary Math

- `tl.exp`
- `tl.log`
- `tl.sigmoid`
- `tl.sqrt`
- `tl.tanh`
- `tl.sin`, `tl.math.sin`
- `tl.cos`
- `tl.tan`
- `tl.atan`
- `tl.cosh`
- `tl.sinh`
- `tl.erf`
- `tl.extra.cuda.libdevice.erf`
- `tl.log2` (Real semantics: `tl.log(x) / log(2)`)
- `tl.exp2` (Real semantics: `tl.exp(x * log(2))`)
- `tl.abs`

These operate on the `.real` channel. `tl.abs(x)` is desugared to
`tl.where(x < 0, 0 - x, x)` in the current real-valued semantics.
The `tl.math.*` namespace and `tl.extra.cuda.libdevice.*` aliases lower to
the same mathematical operator at the algorithm layer; they are not proofs
of CUDA libdevice's bit-level approximation, which belongs to the
ComputeCorrect gap contract path (#1).

### Reductions

- `tl.sum(x)`
- `tl.max(x)`
- `tl.sum(x, axis=N)` and the positional spelling `tl.sum(x, N)`
- `tl.max(x, axis=N)` and the positional spelling `tl.max(x, N)`
- `keep_dims=true|false`
- `tl.max(x, axis=N, return_indices=True)` returns a (value, index) tuple
  consumable only via the multi-assign binding form
  `vmax, imax = tl.max(x, axis=N, return_indices=True)`. Use
  [`ComputeCorrect.OutputPairWhere`](CorrectnessSurfaces.md) for the
  paired-channel correctness surface.

Omitted `axis` follows Triton's `axis=None` behavior: reduce over all
dimensions. Explicit `axis=N` reduces one axis. Reductions are currently over
`.real` tiles.

### Matrix Operations

- `tl.dot(a, b)`.
  It operates on the trailing two dimensions:
  `[..., M, K] × [..., K, N] → [..., M, N]`.
- `tl.dot(a, b, acc)` is accepted as the Triton accumulator form and lowers to
  `acc + tl.dot(a, b)`.
- `tl.trans(e)` is the trailing-two-axis transpose used by `tl.dot(Q, Kᵀ)`.

The current `tl.dot` model is mathematical real matrix multiplication over the
`.real` abstraction, not Triton's hardware-specific dot instruction semantics.

### Memory Operations

Supported loads:

- `tl.load($(region))`
- `tl.load($(region) + offset)`
- `ptrs := $(region) + offset; tl.load(ptrs)`
- `tl.load(ptr, mask=mask)`
- `tl.load(ptr, mask=mask, other=other)`
- `tl.load(ptr, other=other, mask=mask)`
- `tl.load(ptr, dtype=tl.float32|tl.float16|tl.bfloat16|tl.int8|tl.int16|tl.int32|tl.int64|tl.uint8|tl.uint16|tl.uint32|tl.uint64)`
- `tl.load(ptr, mask=mask, other=other, dtype=...)`
- `bp := tl.make_block_ptr($(region), base=$(base), shape=[...],
  strides=[...], offsets=[...], block_shape=[...])`
- `bp2 := tl.advance(bp, [deltas...])`
- `tl.load(bp, boundary_check=([axes...] : List Nat), padding_option="zero")`

Supported stores:

- `tl.store($(region), value)`
- `tl.store($(region) + offset, value)`
- `ptrs := $(region) + offset; tl.store(ptrs, value)`
- `tl.store(ptr, value, mask=mask)`
- `tl.store(ptr, value, dtype=...)`, where `dtype` must match the value dtype.
  Without `dtype=`, stores infer the dtype from `value`.
- `tl.store(bp, value, boundary_check=([axes...] : List Nat))`

Unknown kwargs are rejected. For block pointers, `boundary_check` is supported
only on block-pointer `tl.load` / `tl.store`; it cannot be mixed with `mask` or
`other`. The only modeled `padding_option` is `"zero"`.

The executable checker rejects block-pointer metadata rank mismatches
(`parentShape`, `blockShape`, `strides`, and `offsets` must have the same
rank) and statically visible `tl.advance` underflow. Runtime execution remains
total for unchecked terms; theorem statements should use `BlockPtr.WellFormed`,
`BlockPtr.CheckedAxesValid`, and `BlockPtr.AdvanceNonnegative` when they rely
on Triton-style well-formed block pointers.

Masked load semantics:

- If `mask` is true, read memory.
- If `mask` is false and `other` is supplied, return `other`.
- If `mask` is false and `other` is omitted, Triton leaves the value
  undefined; VeriTile models this through `BlockState.undef`.

Masked store semantics:

- If `mask` is true, write the lane.
- If `mask` is false, leave memory unchanged.

## Memory Model

VeriTile models a limited first-class pointer value as:

```lean
RegionName × Nat
```

The runtime memory model is typed at the cell boundary:

```lean
RegionName → Nat → MemCell
```

The proof-facing Real compatibility API remains `BlockState.readMem` /
`BlockState.writeMem`, so existing mathematical correctness theorems keep their
Real-valued observation surface. In other words, the old theorem-facing
`RegionName → Nat → ℝ` view still exists as an API layer, but it is no longer
the runtime storage representation.

Region dtype contracts are modeled separately in `VeriTile.Triton.Memory.Typing`:

```lean
RegionTyping := RegionName → TileDType
Kernel.RespectsRegionTyping Γ k
```

This is a lightweight static layer. It checks that named-region loads/stores
use the dtype declared by `Γ`, while execution stores typed `MemCell`s. For
pointer-valued loads/stores, statically visible `$(region)` pointer bases are
checked against `Γ`; pointer registers are treated as dynamic/external values
in this layer.

The optional executable checker adds diagnostics and pointer provenance
tracking without replacing the proof-side contract:

```lean
RegionEnv := RegionName → Option TileDType
Kernel.check Γ k
Kernel.checkStrict Γ k
```

`Kernel.check` runs in lax mode and skips undeclared regions; `checkStrict`
reports undeclared regions. The checker tracks register dtype/shape
consistency, pointer and block-pointer provenance through assignments, direct
and pointer-derived load/store dtype mismatches, and basic block-pointer
metadata sanity, including rank equality and static block-pointer advance
underflow. A unified `BlockPtrSummary` carries block-pointer region, optional
parent rank, and optional static offsets through simple expressions and
registers, so checks continue across simple assignment and `tl.advance`
chains. It deliberately does not prove bounds, aliasing,
launch coverage, page ownership, or IEEE/hardware dtype fidelity.
The first proof-facing bridges are available as
`checkBlockPtrMetadata_ok`, `checkBoundaryAxes_ok`, and
`checkStaticAdvanceNonnegative_ok`; summary-level bridges
`BlockPtrSummary.ofStaticChecked_ok`,
`BlockPtrSummary.ofDynamicOffsetsChecked_ok`,
`BlockPtrSummary.checkedAdvance_ok`, and
`BlockPtrSummary.checkBoundary_ok` cover construction and propagation. These
lemmas turn successful local checker calls into the corresponding theorem-side
block-pointer contracts.
The convention for these local obligations is: checker code and decidable
contracts share executable `BlockPtr.*Valid` / axis-level Bool helpers,
theorem statements expose Prop wrappers such as `BlockPtr.WellFormed` and
`BlockPtr.AdvanceNonnegative`, and `_ok` lemmas bridge successful checker
results to those Prop contracts.

Pointer values can be used inline, assigned, and reused:

```lean
ptrs := $(xReg) + offs
x := tl.load(ptrs)
ptrs2 := ptrs + stride
```

This is intentionally narrower than CUDA/Triton pointers: VeriTile supports
pointer base creation from a `RegionName`, pointer plus `.nat` offsets, and
load/store through pointer-valued registers. It also models block pointers as
first-class `.blockPtr` tile values carrying base region, base offset, parent
shape, block shape, strides, and logical offsets. Block-pointer load/store
computes each lane address from that layout; out-of-bounds checked load lanes
return zero, and out-of-bounds checked store lanes leave memory unchanged. It
does not yet model pointer casts, pointer comparison, hardware/TMA block-pointer
behavior, or a typed address space. For proof-facing contracts, use
`BlockPtr.WellFormed` for metadata rank equality, `BlockPtr.CheckedAxesValid`
for `boundary_check` axes, and `BlockPtr.AdvanceNonnegative` for signed
`tl.advance` deltas.

Offsets are explicit `.nat` expressions. For higher-dimensional tensors, the
user supplies strided offset formulas such as:

```lean
b * stride_b + h * stride_h + i * stride_s + d * stride_d
```

Public theorem surfaces use `TensorView.loaded` / `TensorView.observe` to
connect those formulas to mathematical tensor slices. Internally, proofs may
still use the lower-level `InputAt` escape hatch for arbitrary offset maps and
then package the result as a `TensorView`. Aliasing is represented by choosing
equal or distinct `RegionName`s; arbitrary pointer alias analysis beyond those
named regions is not modeled. See
[`GpuMemoryModel.md`](GpuMemoryModel.md) for the GPU memory hierarchy scope
and the sequential-consistency assumptions. See
[`SemanticCaveats.md`](SemanticCaveats.md) for the semantic assumptions that
affect theorem interpretation.

## Floating-Point Model

Arithmetic is currently an `ℝ` abstraction:

- Real data is modeled as `WithBot ℝ`.
- `-inf` is modeled exactly as `⊥`.
- `exp ⊥ = 0` and `sigmoid ⊥ = 0` are built into the semantics.
- Memory stores demote `⊥` with a default value; well-formed kernels should not
  store `⊥`.

What this means: theorems prove real-valued mathematical correctness, not
bit-level IEEE-754 equivalence. Concrete IEEE rounding, NaNs, signed zeros,
overflow, underflow, denormals, exception flags, hardware dot precision, and
fast-math rewrites are not fully modeled. The separate `RoundingModel` layer
applies abstract rounding at explicit float casts and stores, with real-channel
identity (`round_real`) and idempotence (`round_idem`). These contracts do not
establish hardware numerical error bounds. See
[`SemanticCaveats.md`](SemanticCaveats.md) for the review checklist around
partial math functions, fixed-width integers, pointer offsets, and total
memory reads.

The core AST uses one dtype-indexed memory form:

```lean
Op.load    : TileDType → MemAccess shape → MaskOpt dtype shape → Op ...
Stmt.store : TileDType → MemAccess shape → Op ... → MaskOpt dtype shape → Stmt
```

The public DSL defaults `tl.load` to `.real`, but accepts `dtype=...` with
`tl.float32`, `tl.float16`, `tl.bfloat16`, `tl.int32`, `tl.int64`, `tl.uint32`,
or `tl.uint64` to produce typed memory nodes. The narrower spellings
`tl.int8`, `tl.int16`, `tl.uint8`, and `tl.uint16` are also accepted.
All `tl.uint*` spellings map to VeriTile's `.nat` channel for nonnegative
index/block-table values. All `tl.int*` spellings map to VeriTile's `.int`
channel, a mathematical signed-integer abstraction with no bit-width or
overflow semantics. `tl.store`
infers its dtype from the value being stored, with optional matching `dtype=`
syntax for Triton-like surface spelling.

Float theorem policy: exact algorithm contracts use mathematical values.
Rounding contracts use `execR` and retain representable cast/store events in
the projected kernel. The compute-facing DSL surface is
`ComputeKernel`; `ComputeKernel.ComputeCorrect` and
`ComputeKernel.ComputeRefine` project successful compute kernels to the
algorithm layer through `toAlgorithm?`, with an optional `GapPolicy` for
recording externally checked compute-to-algorithm gap contracts. Existing
float-facing theorems can still use erasure equations such as
`k.eraseDType = realK` to reuse Real proofs for dtype-annotated algorithm
kernels. Numeric compute correctness/refinement is represented separately by
external gap contracts and differential tests rather than IEEE-754 proof. These
definitions live in `VeriTile.Triton.Correctness` and `VeriTile.Triton.Float`; see `documents/EraseDType.md` for
the compute/algorithm split and bitcast
policy.

## Deterministic random values

`tl.rand(seed, offsets)` lowers to an explicit `Op.rand` carrying the source
seed/offset integer types and the Philox round count. Supported source types
are int32, int64, uint32 and uint64. The default is ten rounds; a positional
third argument or `n_rounds=` can select a nonnegative constexpr, including
zero. Both operands are evaluated and checked against their declared source
ranges, including the seed when the offset tile is empty.

The primitive computes the actual Philox bits and the canonical binary32
conversion, retaining both the integer-to-float rounding and the subsequent
rounded multiplication. Its finite real projection lies in `[0, 1-2^-24]`.
Both `evalOp` and `evalOpR` use that projection; ambient rounding does not
replace the generator's intrinsic binary32 conversion. Structural FP
evaluation injects the same result bits through `fp32Bits`. A surrounding
cast to fp64 retains the inner binary32 random value.

External scalar seeds require a direct typed binding, for example
`seed_arg = $((seed : Int))`, followed by
`tl.source_integer seed_arg = tl.int32`. This metadata records source
provenance; it does not prove native overflow safety. Symbolic block sizes
use the fully rooted `_root_.VeriTile.Triton.Random.i32ArangeSize` adapter
with an `ArangeSizeDomain` proof: positive, power of two and at most 2^20.
The corresponding rooted `i32NatConstexpr` and `roundsConstexpr` adapters
preserve explicit constexpr values. Unknown source widths or unproved
symbolic arange sizes are rejected. Intermediate source-index arithmetic,
native launch validity and source-to-Lean correspondence require separate
proofs or evidence.

This deterministic primitive supplies the values needed for seeded dropout.
It does not establish dropout's complete execution, IO/frame, source fidelity
or native numerical behavior, and it does not certify probability
distributions, mutable RNG state or a general stochastic program model.

## Operator and Syntax Coverage Checklist

This table is the current operator-coverage contract for GitHub issue #1.
`Supported` means the syntax has a Lean AST constructor or accepted DSL
lowering, operational semantics, and at least the proof surface needed by the
current examples. `Limited` means VeriTile has a deliberately narrow version
of the Triton feature. `Gap` means kernels using the feature are outside the
current semantic contract.

| Area | Status | Coverage |
| --- | --- | --- |
| Scalar/tile constants | Supported | Real literals, context-sensitive `$(x)`, `-inf` / `-float("inf")`, register refs |
| Program IDs | Limited | `tl.program_id(axis)` and `tl.program_id(axis=axis)` for literal or antiquoted `Nat` axes; ND grid quantification is available through `GridIndex` / `Kernel.ForAllPrograms`, but no launch executor is modeled |
| Grid dims | Limited | `tl.num_programs(axis)` / `tl.num_programs(axis=axis)` reading `BlockState.numPids` (default `1`); `withGridIndex` sets it from the launch grid |
| Loops | Supported | Bounded `tl.for`; `tl.static_range` alias backed by the same loop AST |
| Conditionals | Limited | Scalar `if cond { ... }` and `tl.if cond { ... }` (and `... else { ... }`); no `break` or `continue` |
| Multi-assign | Supported | `x, y = e1, e2` parallel binding; `value, index = tl.max(..., return_indices=True)` tuple-op binding |
| Arithmetic | Supported | `+`, `-`, `*`, `/` on same-channel numeric operands; `//`, `%` on integer channels; `tl.cdiv` on `.nat`; `ptr + nat` for pointer offsets |
| Comparisons | Supported | `<`, `<=`, `==`, `>`, `>=`, `!=` on `.real` or `.nat` |
| Boolean ops | Supported | `tl.logical_and`, `tl.logical_or`, `tl.logical_not`, plus `&`, `|`, `~` mask spellings |
| Nat bitwise ops | Limited | `&`, `|`, `^`, `<<`, `>>` on `.nat`; no `.nat ~` and no signed fixed-width bitwise semantics yet |
| Pointwise select | Supported | `tl.where(cond, a, b)` with scalar lifting and matching non-scalar shapes |
| Unary math | Supported | `tl.exp`, `tl.exp2`, `tl.log`, `tl.log2`, `tl.sigmoid`, `tl.sqrt`, `tl.tanh`, `tl.sin`, `tl.cos`, `tl.tan`, `tl.atan`, `tl.cosh`, `tl.sinh`, `tl.erf`, `tl.extra.cuda.libdevice.erf`; floating dtype tags project to `.real` for algorithm proofs |
| Reductions | Supported | `tl.sum`, `tl.max` with optional `axis=` (or positional axis) and `keep_dims` over `.real` or floating-tagged tiles; floating tags project to `.real`; `tl.max(..., return_indices=True)` returns a value/index tuple via multi-assign |
| Directed scans | Limited | `tl.cumsum`, `tl.cumprod`, `tl.associative_scan(x, sum/prod/max/min, axis=N)` with optional `reverse=True/False` (prefix/suffix fold) over `.real` or floating-tagged tiles; no arbitrary combine functions |
| Index/order ops | Limited | `tl.argmax`, `tl.argmin`, `tl.sort` over `.real` or floating-tagged tiles with static `axis=N`; arg ties return the smallest axis index, sort is ascending |
| Broadcast | Supported | ND same-dim, scalar-to-tile, and dimension-`1` expansion |
| Shape construction | Limited | `tl.arange`, `tl.full`, `tl.zeros`, rank-1 `[:, None]` / `[None, :]`, literal-axis `tl.expand_dims` |
| Shape/view ops | Limited | `tl.reshape`, `tl.view`, `tl.ravel`, `tl.permute`, `tl.flip`, `tl.join`, projection-form `tl.split(x, 0|1)` |
| Transpose | Supported | `tl.trans(e)` swaps trailing two axes; arbitrary static permutations use `tl.permute(e, [axes])` |
| Matrix multiply | Supported | `tl.dot(a, b)` and accumulator form `tl.dot(a, b, acc)` over mathematical `ℝ` |
| Bitcast | Limited | 32-bit compute payloads `tl.uint32` / `tl.int32` / `tl.float32`; constant uint32 patterns project to algorithm values, runtime bitcast is expressible but compute-only |
| Loads | Limited | Pointer-expression load, optional `mask`, optional `other`, optional `dtype=` for float/int*/uint* spelling-only integer channels; block-pointer load with `boundary_check` and `padding_option="zero"` |
| Stores | Limited | Pointer-expression store, optional `mask`, dtype inferred from value with optional matching `dtype=`; block-pointer store with `boundary_check` |
| Memory bounds safety | Limited | `Kernel.MemorySafe` / `ComputeKernel.MemorySafe` prove active-lane region bounds for direct, pointer, and block-pointer memory operations; no alias, race, frame, or permission model |
| Memory frame contracts | Limited | Predicate-level `WriteFootprint`, `BlockState.WriteWithin`, `Stmt.StoreAddressesWithin`, and `Kernel.ExecFrame` surface for single-program write-frame reasoning; `tileImage` / `activeTileImage` helpers cover common store footprints, and unrelated-region preservation helpers cover common frame readback goals |
| Disjoint grid composition | Limited | `Kernel.GridFrames`, `GridWritesDisjoint`, and `mergeFrames` merge explicit per-program frames with pairwise-disjoint write footprints; no overlapping writes or scheduling semantics |
| Tensor views | Supported | Strided `TensorView.loaded` / `TensorView.observe` wrappers for theorem statements |
| Integer memory | Limited | Typed cells plus typed load/store support Nat/index and mathematical signed-Int HBM values; no richer signed/unsigned width lattice yet |
| Randomness | Limited | Deterministic `tl.rand` with explicit source integer profiles, Philox rounds and canonical binary32 conversion; no mutable RNG or probabilistic model |
| Indirection | Limited | Typed index loads can feed pointer arithmetic for gather/paged-KV style data-dependent addresses (#1); no alias/bounds/page-ownership proof layer yet |
| Block pointers | Limited | `tl.make_block_ptr`, `tl.advance`, block-pointer load/store with checked-axis zero padding / store skip; no hardware/TMA behavior |
| Atomics / async / barriers | Limited | `tl.atomic_add` has an AlgKernel `Stmt.atomicAdd` marker, sequential single-program semantics, trace payload vocabulary, and a Real grid-merge sum theorem. `tl.atomic_xchg` / `tl.atomic_cas` now project to return-valued `Stmt.atomicRMW` markers and have single-cell RMW fold semantics, executable statement semantics, stateful trace emission, and a single-cell grid launcher relation with an explicit linearization witness. Other `tl.atomic_*`, `tl.async_copy`, `tl.async_wait`, and `tl.debug_barrier` lower to compute-facing failure markers only; async/TMA discipline remains documented but not implemented; no full scheduler, executable barriers, executable async copy semantics, TMA AST, or IEEE atomic semantics (#1) |
| Floating point fidelity | Limited | Mathematical values and abstract cast/store rounding models; no full IEEE-754 hardware semantics (#1) |

## Expressiveness Matrix

This matrix answers a different question from the operator checklist: if a
user starts with a real Triton kernel, what kind of gap would block writing it
faithfully in the current Lean DSL?

| Pattern | Status | Gap type | Practical impact |
| --- | --- | --- | --- |
| Dense elementwise kernels | Mostly supported | Proof/theorem surface | Pointwise arithmetic, masks, casts, pointer values, and TensorView observation are available; richer dtype/memory claims still use the real abstraction. |
| Softmax / reductions / LayerNorm / Welford | Supported for current examples | Proof engineering | Core reductions, loops, masks, and TensorView wrappers exist; new kernels mainly need invariants and theorem packaging. |
| FlashAttention-style dense tiled kernels | Supported for FA-1 forward and FA-1 backward core surfaces | Proof engineering + limited semantics | Dot, transpose, causal/boundary masks, D-tail, 4D views, Real backward math, atomic-`dQ` launcher composition, strided stripped backward, and a 4D full-sequence backward wrapper are covered; async/shared-memory/hardware dot fidelity remain out of scope. |
| First-class pointer expressions | Limited | Surface + lightweight semantics | `ptrs := $(r) + offs`, pointer registers, pointer load/store, and pointer offset updates work for `RegionName × Nat`; no pointer casts/comparison/alias analysis. |
| Block pointers / `boundary_check` | Limited | Surface + sequential semantics | `tl.make_block_ptr`, `tl.advance`, zero-padded checked loads, and checked store-skip work; no `order`, non-zero padding, TMA, or hardware behavior. |
| Typed floating memory | Limited | Semantic abstraction | `dtype=tl.float32/fp16/bf16` creates typed floating nodes and erases to real for algorithm proofs; IEEE rounding is not modeled. |
| Integer / bool tensor memory | Limited | Dtype coverage | Typed cells plus typed load/store support Nat/index and mathematical signed-Int HBM values; no complete Triton integer-width lattice yet. |
| Indirect / gather addressing | Limited | Surface + view semantics (#1) | Typed index tensor loads can drive pointer arithmetic and ordinary masked loads; alias analysis, bounds proof, page ownership, and paged FA-1 equivalence are not modeled yet. |
| Active-lane memory bounds | Limited | Lean proof predicate (#1) | `Kernel.MemorySafe` checks direct region offsets, dynamic pointer addresses, mask activeness, and `boundary_check` block-pointer lanes against `RegionBounds`; no race freedom, frame theorem, or permission accounting. |
| Single-program write footprint/frame | Limited | Predicate-level frame contract + extraction helpers | `WriteFootprint := (RegionName × Nat) → Prop` and `BlockState.WriteWithin` state that an execution only modified cells inside a supplied footprint; `tileImage` / `activeTileImage` helpers extract direct, masked, and checked block-pointer store footprints. |
| RNG / dropout | Limited | Deterministic value semantics | Profiled `tl.rand` is modeled; complete dropout source/IO/FP proofs and state/probabilistic semantics remain separate obligations. |
| Atomics / async / shared memory / barriers | Limited | Atomic-add slice + concurrency boundary (#1) | `tl.atomic_add` has a proof-facing marker and Real trace/grid-sum theorem; `tl.atomic_xchg` / `tl.atomic_cas` have return-valued algorithm markers, single-cell RMW fold semantics, executable statement/trace integration, and a single-cell grid launcher relation; remaining unsupported atomic family members and async/barrier surfaces fail projection explicitly; async/TMA, shared memory, barriers, full scheduling, and IEEE atomic behavior remain gaps. |
| Whole-grid launch semantics | Limited | ND grid theorem surface + disjoint/atomic merge (#1) | `GridIndex`, `BlockState.withGridIndex`, `Kernel.ForAllPrograms`, and `ForAllProgramsSome` quantify per-program correctness; `Kernel.mergeFrames` handles pairwise-disjoint footprints, `Kernel.mergeFramesWithAtomic` handles selected Real atomic-add contributions, and `Kernel.GridLaunchedRMW` handles one order-sensitive RMW cell with an explicit linearization witness. No full race/scheduler/interleaved executor. |
| Python/Triton source ingestion | Gap | Front-end/lifter (#1) | Users must write Lean `triton { ... }`; decorators, Python-side constexpr execution, and general Python control flow are not parsed. |
| Type checking / pointer provenance | Limited | Optional checker (#1) | `Kernel.check` / `checkStrict` track register dtype/shape, pointer and block-pointer provenance, dtype mismatches, and basic block-pointer metadata; no bounds, alias, launch, or page-ownership proof. |

Recommended near-term priority for expressiveness is to remove core semantic
gaps before building a full Python lifter: stateful/probabilistic RNG and dropout integration (#1),
atomics/async/concurrency (#1), and bounds/memory-safety assumptions (#1).
A lifter is only useful for kernels whose operations are already representable.

## Unsupported or Not Yet Faithfully Modeled

- Full IEEE-754 floating-point semantics.
- Block-pointer hardware/TMA behavior and unsupported padding options beyond
  `"zero"`.
- Full CUDA/Triton pointer semantics beyond `RegionName × Nat` pointer values.
- Arbitrary pointer alias analysis beyond named-region equality.
- General Python/Triton JIT semantics, decorators, meta-parameter execution,
  and Python control flow outside the embedded `triton { ... }` block.
- Atomic-family members beyond the current limited `atomic_add` /
  `atomic_xchg` / `atomic_cas` algorithm slices.
- Async copy / TMA / shared-memory staging.
- Barriers and inter-program or inter-warp synchronization.
- Full grid launch execution. `GridIndex` / `Kernel.ForAllPrograms` provide a
  theorem surface for quantifying over every program instance in an ND grid,
  but VeriTile still does not model a sequential or concurrent launch executor,
  overlapping ordinary writes, races, full CUDA atomic memory ordering, or
  scheduling.
- Caches and performance hints such as `cache_modifier`, `eviction_policy`,
  `volatile`, or `is_volatile`.
- Higher-rank bracket slicing beyond the currently supported rank-1
  `[:, None]` / `[None, :]` forms. Use `tl.expand_dims(e, axis=N)` for
  explicit unit-axis insertion.
- Integer widths, overflow, signedness, and signed bitwise behavior. The `.nat`
  channel is mathematical `Nat`.

## Documentation Generation

There is no automatic documentation generator for this subset yet. A useful
future tool would extract the raw DSL syntax / `Op` constructors into a table,
then compare it against this document. That would catch drift such as
"implemented but undocumented" or "documented but no longer accepted".

For now this document is manually maintained because the important artifact
claim is semantic, not just syntactic: the gaps above require human judgment.

<!-- generated by scripts/census/triton_coverage.py: begin; do not edit up to the end marker -->

## Triton 3.5.1 Coverage Matrix (generated, unreviewed)

Ticket A1.1 (issue #75 of the fork `dnyBLUESKY007/VeriTile`; every issue number in this section is an issue of that fork). **This section is generated and has not been reviewed.** It is written by `scripts/census/triton_coverage.py`; edit the generator, not the text between the two marker comments. The sections above it are maintained by hand and are not changed by the generator.

The matrix compares the public surface of Triton 3.5.1 with the `syntax` declarations of the `triton { }` macro. It is a comparison of source text: the macro was not run, and a rule says that a spelling parses, not that its meaning is modelled.

### Regenerating

The generator reads one checkout per pin, in a directory of your choice, each named `<repository>@<full commit hash>`: the pins of `scripts/census/targets.tsv` (the loop in `documents/census/MinimalSyntax.md` fetches them) and the two below. From the root of this repository:

```sh
sources=/path/to/the/directory/of/checkouts
for pin in \
  https://github.com/triton-lang/triton@0add68262ab0a2e33b84524346cb27cbb2787356 \
  https://github.com/VeriInfra-Lab/VeriTileBench@750a79190dfb33539e4868a51629f6db5303d91d; do
  url=${pin%@*}; commit=${pin##*@}; d="$sources/$(basename "$url")@$commit"
  git init -q "$d" && git -C "$d" fetch -q --depth 1 "$url" "$commit" && git -C "$d" checkout -q FETCH_HEAD
done
python3 scripts/census/triton_coverage.py scripts/census/targets.tsv "$sources" --update documents/TritonSubset.md
python3 site/scripts/sync-docs.py
```

The second command rewrites this section; the third refreshes the copy of this page under `site/`, which `scripts/check_docs.py` compares. The checkouts are data: Python files are parsed with `ast`, Lean files are read as text, and nothing is imported or run. Equal inputs give equal bytes.

### Inputs

- Triton: `0add68262ab0a2e33b84524346cb27cbb2787356`, the tag `v3.5.1`. The rows are the names of `triton.language.__all__` in `python/triton/language/__init__.py`. A function has one more row per optional parameter of its signature (a class: of its `__init__`; names starting with `_` are left out) and per keyword that a scanned kernel passes.
- The macro: the `syntax` declarations in the 143 Lean files under `VeriTile/Triton/DSL/` of this repository, 454 declarations. File names in the table are relative to that directory.
- Target kernels: the files that `scripts/census/targets.tsv` lists for the front ends Triton (23 kernels) and Triton-distributed (119 kernels), scanned by `scripts/census/scan.py` as in `documents/census/MinimalSyntax.md`. A kernel is charged with the constructs of the functions of its own file that it reaches.
- Benchmark kernels: the 183 files `bench/tritonbench_g/*/*.py` of `VeriInfra-Lab/VeriTileBench` at `750a79190dfb33539e4868a51629f6db5303d91d`; a file counts when a kernel in it uses the construct. The map (#1) names no commit of that repository. It says the benchmark requires VeriTile at `3bf30ea`, and this is the one commit whose `lake-manifest.json` pins that revision.

### How a row is decided

Each declaration is read as a form: its atoms as written, with a placeholder for each non-terminal. Only declarations of the categories `tritonExpr`, `tritonStmt`, `tritonDType` count, which are the ones a `triton { }` block is parsed with.

- **rule**: a form spells the construct. For a call, the form contains `tl.name(`; for a dtype or another value, the bare `tl.name`; for a statement kind, the form has the shape of the statement. The basis column gives the category and the lines of the declarations.
- **no rule**: no form spells it.
- **partial**: a rule exists and the generator found a limit: the expander lowers the call to a failure marker (`effectMarker "tl.name"`), a keyword row of the call has no rule, or only some spellings of a statement kind have a declaration.
- A keyword row `tl.name(k=)` has a rule when a form of the call spells `k=`. When the form has a keyword position that takes any name (`<ident>=`, directly or through a keyword category), the syntax accepts the keyword and the expander decides. Then: if the word `k` occurs in no file of the directory outside comments, nothing can recognise it, and the row is **no rule**; otherwise the row is decided **by reading** or is **not determined**.
- **by reading**: the basis contains "by reading" and names the file and line relied on. The generator holds these entries and checks that the line still contains the text that was read; if it does not, the row becomes "not determined".
- **never (PTX boundary)**: excluded by the map (#1), whatever the macro accepts.
- **not determined**: neither the test nor a reading settles it.

The ticket column names the ticket of epic A1 (#299) that owns a gap, for rows whose status is not "rule".

- #76 (A1.2): the tensor-descriptor names, and `.load` and `.store` when every target kernel that calls them also calls `tl.make_tensor_descriptor`.
- #77 (A1.3): every other row that a target kernel uses, because A1.3 covers `tl.range` and "any other name the census shows a target kernel needs". "if it is a gap" is added when the status is not "no rule". The assignment applies that sentence of the ticket and nothing else, so it also takes in statement kinds, builtins and methods, which are not names of Triton.
- No ticket: a row that no target kernel uses, which A1.3 leaves as a parse failure; `With`, when every target kernel that uses it calls `dl.simt_exec_region`; a value whose uses the census does not count. Benchmark counts do not assign a ticket.
- A row whose status comes from other rows points to them.

### Where the decisions can be wrong

- A rule is evidence that a spelling parses. The expander behind it can still reject an argument form, a dtype or a shape, and only the failure markers and the entries read by hand are looked at. "rule" is not "supported" in the sense of the checklist above.
- Only the call name and the keyword names are compared: not the number of positional arguments, not the values of keywords (the macro writes `true` where Python writes `True`), not operators, subscripts, conditional expressions or the targets of assignments. The census does not report those.
- "no rule" means that no declaration names the construct. A declaration that takes any name, the call of a bound helper `<ident>(...)`, can still parse such a call; it is accepted only for helpers bound outside the block. Whether Lean reports a construct without a rule as a parse error or as an elaboration error was not determined.
- The word test for keywords assumes that an expander recognises a keyword by naming it. An expander that ignores every keyword would be missed; the ones found by reading are the failure markers.
- The census resolves imports, so the spelling a kernel uses is not known. A call has a rule when a declaration spells its full name or its last two components (`libdevice.exp(`); the basis says which.
- For a statement kind the census gives the kind, not the operator of an augmented assignment or the annotation of an annotated one. Where only some spellings have a declaration, it is not known whether the kernels use the others; the ticket column then says "if it is a gap".
- Counts are of kernels whose source uses the construct by that name. A method call (`x.sum()`) and the function (`tl.sum(x)`) are different rows. A parameter passed by position is counted in the row of the call, not in its keyword row. Values that are not called (dtypes, modules, `tl.constexpr` as an annotation) are not counted by the census: their count is `-`.
- Whole files are scanned, so kernels that no target reaches are counted, and helpers imported from another file are not followed (see `documents/census/MinimalSyntax.md`).

### Totals

350 rows.

| status | rows | decided mechanically | decided by reading |
| --- | --- | --- | --- |
| never (PTX boundary) | 8 | 7 | 1 |
| no rule | 156 | 145 | 11 |
| not determined | 3 | 2 | 1 |
| partial | 54 | 45 | 9 |
| rule | 129 | 94 | 35 |

### `triton.language.__all__` (139 names)

| construct | status | basis | Triton (23) | Triton-distributed (119) | bench (183 files) | ticket |
| --- | --- | --- | --- | --- | --- | --- |
| `tl.PropagateNan` | no rule | no declaration spells `tl.PropagateNan` | - | - | - | not determined: uses are not counted |
| `tl.TRITON_MAX_TENSOR_NUMEL` | no rule | no declaration spells `tl.TRITON_MAX_TENSOR_NUMEL` | - | - | - | not determined: uses are not counted |
| `tl.abs(...)` | rule | `tritonExpr` `Syntax.lean:98` | 1 | 3 | 6 |  |
| `tl.add(...)` | no rule | no declaration spells `tl.add(` | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.add(sanitize_overflow=)` | no rule | the call has no rule | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.advance(...)` | rule | `tritonExpr` `Syntax.lean:247-249` | 0 | 0 | 9 |  |
| `tl.advance(offsets=)` | rule | a keyword position exists (`Syntax.lean:249`), so the expander decides; by reading `Expansion/Main.lean:1927` | 0 | 0 | 1 |  |
| `tl.arange(...)` | rule | `tritonExpr` `Syntax.lean:45-46` | 22 | 63 | 178 |  |
| `tl.argmax(...)` | partial | `tritonExpr` `Syntax.lean:226`; a keyword below has no rule | 0 | 0 | 2 | see its keyword rows |
| `tl.argmax(tie_break_left=)` | partial | a keyword position exists (`Syntax.lean:210, 212`), so the expander decides; by reading `Expansion/Compute.lean:743`: only the literal `True` | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.argmax(keep_dims=)` | no rule | a keyword position exists (`Syntax.lean:203-204, 210, 212`), so the expander decides; by reading `Expansion/Compute.lean:715`: rejected | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.argmax(axis=)` | rule | a keyword position exists (`Syntax.lean:200-201, 210, 212`), so the expander decides; by reading `Expansion/Compute.lean:705`: a numeric literal | 0 | 0 | 1 |  |
| `tl.argmin(...)` | partial | `tritonExpr` `Syntax.lean:227`; a keyword below has no rule | 0 | 0 | 0 | see its keyword rows |
| `tl.argmin(tie_break_left=)` | no rule | a keyword position exists (`Syntax.lean:210, 212`), so the expander decides; by reading `Expansion/Compute.lean:720`: rejected: only `tl.argmax` takes the keyword (`Expansion/Compute.lean:729-732`) | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.argmin(keep_dims=)` | no rule | a keyword position exists (`Syntax.lean:203-204, 210, 212`), so the expander decides; by reading `Expansion/Compute.lean:715`: rejected | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.associative_scan(...)` | rule | `tritonExpr` `Syntax.lean:225` | 0 | 0 | 0 |  |
| `tl.associative_scan(reverse=)` | rule | a keyword position exists (`Syntax.lean:207-208, 210, 212`), so the expander decides; by reading `Expansion/Compute.lean:654` | 0 | 0 | 0 |  |
| `tl.assume(...)` | no rule | no declaration spells `tl.assume(` | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.async_task(...)` | no rule | no declaration spells `tl.async_task(` | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.atomic_add(...)` | partial | `tritonStmt` `Syntax.lean:407`; a keyword below has no rule | 0 | 8 | 6 | see its keyword rows |
| `tl.atomic_add(mask=)` | rule | a keyword position exists (`Syntax.lean:186-187, 190`), so the expander decides; by reading `Expansion/Memory.lean:517` | 0 | 1 | 6 |  |
| `tl.atomic_add(sem=)` | no rule | a keyword position exists (`Syntax.lean:186-187, 190`) and no file mentions `sem` | 0 | 3 | 0 | #77 |
| `tl.atomic_add(scope=)` | no rule | a keyword position exists (`Syntax.lean:186-187, 190`), so the expander decides; by reading `Expansion/Memory.lean:527`: rejected as an unknown keyword | 0 | 3 | 0 | #77 |
| `tl.atomic_and(...)` | partial | `tritonStmt` `Syntax.lean:410`; lowers to a failure marker only (`Expansion/Main.lean:3974`) | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.atomic_and(mask=)` | partial | the call lowers to a failure marker only | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.atomic_and(sem=)` | partial | the call lowers to a failure marker only | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.atomic_and(scope=)` | partial | the call lowers to a failure marker only | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.atomic_cas(...)` | partial | `tritonExpr` `Syntax.lean:319`; `tritonStmt` `Syntax.lean:414`; a keyword below has no rule | 0 | 0 | 1 | see its keyword rows |
| `tl.atomic_cas(sem=)` | no rule | a keyword position exists (`Syntax.lean:186-187, 190`) and no file mentions `sem` | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.atomic_cas(scope=)` | no rule | a keyword position exists (`Syntax.lean:186-187, 190`), so the expander decides; by reading `Expansion/Main.lean:2770`: any keyword is rejected | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.atomic_max(...)` | partial | `tritonStmt` `Syntax.lean:408`; lowers to a failure marker only (`Expansion/Main.lean:3970`) | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.atomic_max(mask=)` | partial | the call lowers to a failure marker only | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.atomic_max(sem=)` | partial | the call lowers to a failure marker only | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.atomic_max(scope=)` | partial | the call lowers to a failure marker only | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.atomic_min(...)` | partial | `tritonStmt` `Syntax.lean:409`; lowers to a failure marker only (`Expansion/Main.lean:3972`) | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.atomic_min(mask=)` | partial | the call lowers to a failure marker only | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.atomic_min(sem=)` | partial | the call lowers to a failure marker only | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.atomic_min(scope=)` | partial | the call lowers to a failure marker only | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.atomic_or(...)` | partial | `tritonStmt` `Syntax.lean:411`; lowers to a failure marker only (`Expansion/Main.lean:3976`) | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.atomic_or(mask=)` | partial | the call lowers to a failure marker only | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.atomic_or(sem=)` | partial | the call lowers to a failure marker only | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.atomic_or(scope=)` | partial | the call lowers to a failure marker only | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.atomic_xchg(...)` | partial | `tritonExpr` `Syntax.lean:318`; `tritonStmt` `Syntax.lean:413`; a keyword below has no rule | 0 | 0 | 1 | see its keyword rows |
| `tl.atomic_xchg(mask=)` | no rule | a keyword position exists (`Syntax.lean:186-187, 190`), so the expander decides; by reading `Expansion/Main.lean:2770`: any keyword is rejected | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.atomic_xchg(sem=)` | no rule | a keyword position exists (`Syntax.lean:186-187, 190`) and no file mentions `sem` | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.atomic_xchg(scope=)` | no rule | a keyword position exists (`Syntax.lean:186-187, 190`), so the expander decides; by reading `Expansion/Main.lean:2770`: any keyword is rejected | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.atomic_xor(...)` | partial | `tritonStmt` `Syntax.lean:412`; lowers to a failure marker only (`Expansion/Main.lean:3978`) | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.atomic_xor(mask=)` | partial | the call lowers to a failure marker only | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.atomic_xor(sem=)` | partial | the call lowers to a failure marker only | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.atomic_xor(scope=)` | partial | the call lowers to a failure marker only | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.bfloat16` | rule | `tritonDType` `Syntax.lean:164`; `tritonExpr` `Syntax.lean:34` | - | - | - |  |
| `tl.bitonic_merge(...)` | no rule | no declaration spells `tl.bitonic_merge(` | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.bitonic_merge(dim=)` | no rule | the call has no rule | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.bitonic_merge(descending=)` | no rule | the call has no rule | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.block_type(...)` | no rule | no declaration spells `tl.block_type(` | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.broadcast(...)` | rule | `tritonExpr` `Syntax.lean:274`; `tritonStmt` `Syntax.lean:379-382` | 0 | 0 | 1 |  |
| `tl.broadcast_to(...)` | rule | `tritonExpr` `RegisterShapeBroadcast.lean:9`, `Syntax.lean:278` | 0 | 0 | 3 |  |
| `tl.cast(...)` | partial | `tritonExpr` `Syntax.lean:115`; a keyword below has no rule | 0 | 39 | 50 | see its keyword rows |
| `tl.cast(fp_downcast_rounding=)` | no rule | no declaration of `tl.cast(` has a position for this keyword | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.cast(bitcast=)` | no rule | no declaration of `tl.cast(` has a position for this keyword | 0 | 10 | 0 | #77 |
| `tl.cast(_semantic=)` | no rule | no declaration of `tl.cast(` has a position for this keyword | 0 | 3 | 0 | #77 |
| `tl.cast(dtype=)` | no rule | no declaration of `tl.cast(` has a position for this keyword | 0 | 10 | 0 | #77 |
| `tl.cat(...)` | no rule | no declaration spells `tl.cat(` | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.cat(can_reorder=)` | no rule | the call has no rule | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.cdiv(...)` | rule | `tritonExpr` `Syntax.lean:102` | 0 | 55 | 45 |  |
| `tl.ceil(...)` | no rule | no declaration spells `tl.ceil(` | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.clamp(...)` | no rule | no declaration spells `tl.clamp(` | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.clamp(propagate_nan=)` | no rule | the call has no rule | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.condition(...)` | no rule | no declaration spells `tl.condition(` | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.condition(disable_licm=)` | no rule | the call has no rule | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.const(...)` | no rule | no declaration spells `tl.const(` | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.constexpr(...)` | no rule | no declaration spells `tl.constexpr(`; the bare name has one: `tritonStmt` `SourceDType.lean:16` | 0 | 26 | 1 | #77 |
| `tl.constexpr_type(...)` | no rule | no declaration spells `tl.constexpr_type(` | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.cos(...)` | rule | `tritonExpr` `Syntax.lean:91` | 0 | 0 | 4 |  |
| `tl.cumprod(...)` | rule | `tritonExpr` `Syntax.lean:224` | 0 | 0 | 0 |  |
| `tl.cumprod(axis=)` | rule | a keyword position exists (`Syntax.lean:200-201, 210, 212`), so the expander decides; by reading `Expansion/Compute.lean:667`: a numeric literal | 0 | 0 | 0 |  |
| `tl.cumprod(reverse=)` | rule | a keyword position exists (`Syntax.lean:207-208, 210, 212`), so the expander decides; by reading `Expansion/Compute.lean:654` | 0 | 0 | 0 |  |
| `tl.cumsum(...)` | partial | `tritonExpr` `Syntax.lean:223`; a keyword below has no rule | 0 | 6 | 4 | see its keyword rows |
| `tl.cumsum(axis=)` | rule | a keyword position exists (`Syntax.lean:200-201, 210, 212`), so the expander decides; by reading `Expansion/Compute.lean:667`: a numeric literal | 0 | 3 | 3 |  |
| `tl.cumsum(reverse=)` | rule | a keyword position exists (`Syntax.lean:207-208, 210, 212`), so the expander decides; by reading `Expansion/Compute.lean:654` | 0 | 0 | 0 |  |
| `tl.cumsum(dtype=)` | no rule | a keyword position exists (`Syntax.lean:210, 212`), so the expander decides; by reading `Expansion/Compute.lean:682`: rejected | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.debug_barrier(...)` | partial | `tritonStmt` `Syntax.lean:419`; lowers to a failure marker only (`Expansion/Main.lean:3995`) | 0 | 0 | 10 | none: no target kernel uses it |
| `tl.device_assert(...)` | no rule | no declaration spells `tl.device_assert(` | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.device_assert(msg=)` | no rule | the call has no rule | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.device_assert(mask=)` | no rule | the call has no rule | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.device_print(...)` | no rule | no declaration spells `tl.device_print(` | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.device_print(hex=)` | no rule | the call has no rule | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.div_rn(...)` | rule | `tritonExpr` `Syntax.lean:60` | 0 | 0 | 1 |  |
| `tl.dot(...)` | partial | `tritonExpr` `Syntax.lean:181-184`; a keyword below has no rule | 0 | 22 | 60 | see its keyword rows |
| `tl.dot(acc=)` | no rule | a keyword position exists (`Syntax.lean:210, 212`), so the expander decides; by reading `Expansion/Compute.lean:1214`: rejected as a keyword; a third positional argument has a declaration | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.dot(input_precision=)` | rule | a keyword position exists (`Syntax.lean:210-212`), so the expander decides; by reading `Expansion/Compute.lean:1198` | 0 | 0 | 0 |  |
| `tl.dot(allow_tf32=)` | rule | a keyword position exists (`Syntax.lean:210, 212`), so the expander decides; by reading `Expansion/Compute.lean:1210` | 0 | 0 | 15 |  |
| `tl.dot(max_num_imprecise_acc=)` | rule | a keyword position exists (`Syntax.lean:210, 212`), so the expander decides; by reading `Expansion/Compute.lean:1212` | 0 | 0 | 0 |  |
| `tl.dot(out_dtype=)` | rule | a keyword position exists (`Syntax.lean:209-210, 212`), so the expander decides; by reading `Expansion/Compute.lean:1193` | 0 | 0 | 6 |  |
| `tl.dot_scaled(...)` | no rule | no declaration spells `tl.dot_scaled(` | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.dot_scaled(acc=)` | no rule | the call has no rule | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.dot_scaled(fast_math=)` | no rule | the call has no rule | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.dot_scaled(lhs_k_pack=)` | no rule | the call has no rule | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.dot_scaled(rhs_k_pack=)` | no rule | the call has no rule | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.dot_scaled(out_dtype=)` | no rule | the call has no rule | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.dtype(...)` | no rule | no declaration spells `tl.dtype(` | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.erf(...)` | rule | `tritonExpr` `Syntax.lean:96` | 0 | 0 | 0 |  |
| `tl.exp(...)` | rule | `tritonExpr` `Syntax.lean:61` | 0 | 0 | 32 |  |
| `tl.exp2(...)` | rule | `tritonExpr` `Syntax.lean:70` | 0 | 0 | 0 |  |
| `tl.expand_dims(...)` | rule | `tritonExpr` `Syntax.lean:261-262` | 0 | 0 | 0 |  |
| `tl.extra` (module) | rule | a declaration spells a name under it: `tritonExpr` `Syntax.lean:63, 65, 67, 69, 72, 83, 87, 97, 125, 128-129` | - | - | - |  |
| `tl.extra.cuda.libdevice.erf(...)` | rule | `tritonExpr` `Syntax.lean:97` | 0 | 0 | 1 |  |
| `tl.extra.cuda.libdevice.exp(...)` | rule | `tritonExpr` `Syntax.lean:63` | 6 | 0 | 0 |  |
| `tl.extra.cuda.libdevice.expm1(...)` | rule | `tritonExpr` `Syntax.lean:67` | 0 | 0 | 0 |  |
| `tl.extra.cuda.libdevice.finitef(...)` | rule | `tritonExpr` `Syntax.lean:128` | 0 | 0 | 2 |  |
| `tl.extra.cuda.libdevice.isfinited(...)` | rule | `tritonExpr` `Syntax.lean:129` | 0 | 0 | 0 |  |
| `tl.extra.cuda.libdevice.llrint(...)` | rule | `tritonExpr` `Syntax.lean:125` | 0 | 0 | 3 |  |
| `tl.extra.cuda.libdevice.log(...)` | rule | `tritonExpr` `Syntax.lean:65` | 0 | 0 | 0 |  |
| `tl.extra.cuda.libdevice.log1p(...)` | rule | `tritonExpr` `Syntax.lean:69` | 0 | 0 | 0 |  |
| `tl.extra.cuda.libdevice.pow(...)` | rule | `tritonExpr` `Syntax.lean:72` | 0 | 0 | 5 |  |
| `tl.extra.cuda.libdevice.rsqrt(...)` | rule | `tritonExpr` `Syntax.lean:83` | 0 | 0 | 0 |  |
| `tl.extra.cuda.libdevice.tanh(...)` | rule | `tritonExpr` `Syntax.lean:87` | 0 | 0 | 1 |  |
| `tl.extra.cuda.num_threads(...)` | no rule | no declaration spells `tl.extra.cuda.num_threads(` | 0 | 1 | 0 | #77 |
| `tl.extra.cuda.num_warps(...)` | no rule | no declaration spells `tl.extra.cuda.num_warps(` | 0 | 3 | 0 | #77 |
| `tl.extra.cuda.utils.num_warps(...)` | no rule | no declaration spells `tl.extra.cuda.utils.num_warps(` | 0 | 3 | 0 | #77 |
| `tl.extra.libdevice.tanh(...)` | rule | under the spelling `libdevice.tanh(`: `tritonExpr` `Syntax.lean:86` | 0 | 0 | 1 |  |
| `tl.fdiv(...)` | no rule | no declaration spells `tl.fdiv(` | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.fdiv(ieee_rounding=)` | no rule | the call has no rule | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.flip(...)` | rule | `tritonExpr` `Syntax.lean:269-270` | 0 | 0 | 0 |  |
| `tl.flip(dim=)` | rule | a keyword position exists (`Syntax.lean:210, 212`), so the expander decides; by reading `Expansion/Compute.lean:1448`: a numeric literal | 0 | 0 | 0 |  |
| `tl.float16` | rule | `tritonDType` `Syntax.lean:163`; `tritonExpr` `Syntax.lean:33` | - | - | - |  |
| `tl.float32` | rule | `tritonDType` `Syntax.lean:162`; `tritonExpr` `Syntax.lean:35` | - | - | - |  |
| `tl.float64` | rule | `tritonDType` `Syntax.lean:161`; `tritonExpr` `Syntax.lean:36` | - | - | - |  |
| `tl.float8e4b15` | no rule | no declaration spells `tl.float8e4b15` | - | - | - | not determined: uses are not counted |
| `tl.float8e4b8` | no rule | no declaration spells `tl.float8e4b8` | - | - | - | not determined: uses are not counted |
| `tl.float8e4nv` | rule | `tritonDType` `Syntax.lean:165`; `tritonExpr` `Syntax.lean:37` | - | - | - |  |
| `tl.float8e5` | rule | `tritonDType` `Syntax.lean:166`; `tritonExpr` `Syntax.lean:38` | - | - | - |  |
| `tl.float8e5b16` | no rule | no declaration spells `tl.float8e5b16` | - | - | - | not determined: uses are not counted |
| `tl.floor(...)` | no rule | no declaration spells `tl.floor(` | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.fma(...)` | rule | `tritonExpr` `Syntax.lean:109` | 0 | 0 | 2 |  |
| `tl.full(...)` | rule | `tritonExpr` `ControlSource.lean:64`, `FloatDTypeConstants.lean:11`, `PythonFullConstructors.lean:9`, `RawBodyControl.lean:4`, `Syntax.lean:280-289` | 5 | 10 | 14 |  |
| `tl.full(dtype=)` | rule | a keyword position exists (`Syntax.lean:281-289`), so the expander decides; by reading `Expansion/Main.lean:2197` | 0 | 4 | 4 |  |
| `tl.full(value=)` | rule | a keyword position exists (`Syntax.lean:281-289`), so the expander decides; by reading `Expansion/Main.lean:2212`: only together with `dtype=` | 0 | 0 | 3 |  |
| `tl.gather(...)` | no rule | no declaration spells `tl.gather(` | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.histogram(...)` | no rule | no declaration spells `tl.histogram(` | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.histogram(mask=)` | no rule | the call has no rule | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.inline_asm_elementwise(...)` | never (PTX boundary) | `tritonExpr` `InlineAsmWordAdd.lean:10`; by reading `InlineAsmWordAdd.lean:26`: the map (#1) keeps everything at or below PTX out of scope. The one declaration takes a single fixed form, a pure `add.u32` of two inputs, and rejects any other assembly | 0 | 14 | 2 |  |
| `tl.inline_asm_elementwise(_semantic=)` | never (PTX boundary) | as the call | 0 | 9 | 0 |  |
| `tl.inline_asm_elementwise(args=)` | never (PTX boundary) | as the call | 0 | 14 | 2 |  |
| `tl.inline_asm_elementwise(asm=)` | never (PTX boundary) | as the call | 0 | 14 | 0 |  |
| `tl.inline_asm_elementwise(constraints=)` | never (PTX boundary) | as the call | 0 | 14 | 2 |  |
| `tl.inline_asm_elementwise(dtype=)` | never (PTX boundary) | as the call | 0 | 14 | 2 |  |
| `tl.inline_asm_elementwise(is_pure=)` | never (PTX boundary) | as the call | 0 | 14 | 2 |  |
| `tl.inline_asm_elementwise(pack=)` | never (PTX boundary) | as the call | 0 | 14 | 2 |  |
| `tl.int1` | rule | `tritonDType` `Syntax.lean:167` | - | - | - |  |
| `tl.int16` | rule | `tritonDType` `Syntax.lean:169` | - | - | - |  |
| `tl.int32` | rule | `tritonDType` `Syntax.lean:170` | - | - | - |  |
| `tl.int64` | rule | `tritonDType` `Syntax.lean:171` | - | - | - |  |
| `tl.int8` | rule | `tritonDType` `Syntax.lean:168` | - | - | - |  |
| `tl.interleave(...)` | rule | `tritonExpr` `Syntax.lean:272` | 0 | 6 | 1 |  |
| `tl.join(...)` | rule | `tritonExpr` `Syntax.lean:271` | 0 | 0 | 2 |  |
| `tl.load(...)` | partial | `tritonExpr` `PositionalLoadOther.lean:9`, `Syntax.lean:230-237`; a keyword below has no rule | 23 | 72 | 182 | see its keyword rows |
| `tl.load(mask=)` | rule | a keyword position exists (`Syntax.lean:186-187, 190`), so the expander decides; by reading `Expansion/Memory.lean:168` | 3 | 50 | 153 |  |
| `tl.load(other=)` | rule | a keyword position exists (`Syntax.lean:186-187, 190`), so the expander decides; by reading `Expansion/Memory.lean:175` | 3 | 19 | 134 |  |
| `tl.load(boundary_check=)` | rule | a keyword position exists (`SignedRankOneBlockPointers.lean:13`, `Syntax.lean:186-187, 190-194`), so the expander decides; by reading `Expansion/Memory.lean:132` | 0 | 0 | 19 |  |
| `tl.load(padding_option=)` | rule | a keyword position exists (`Syntax.lean:186-187, 190, 195`), so the expander decides; by reading `Expansion/Memory.lean:148` | 0 | 0 | 15 |  |
| `tl.load(cache_modifier=)` | no rule | a keyword position exists (`Syntax.lean:186-187, 190`) and no file mentions `cache_modifier` | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.load(eviction_policy=)` | rule | a keyword position exists (`Syntax.lean:186-187, 190, 196-198`), so the expander decides; by reading `Expansion/Memory.lean:153`: only the literals `"evict_last"` and `"evict_first"`, which are dropped as hints | 0 | 0 | 5 |  |
| `tl.load(volatile=)` | no rule | a keyword position exists (`Syntax.lean:186-187, 190`) and no file mentions `volatile` | 0 | 4 | 0 | #77 |
| `tl.load_tensor_descriptor(...)` | no rule | no declaration spells `tl.load_tensor_descriptor(` | 0 | 0 | 0 | #76 |
| `tl.log(...)` | rule | `tritonExpr` `Syntax.lean:73` | 2 | 0 | 11 |  |
| `tl.log2(...)` | rule | `tritonExpr` `Syntax.lean:74` | 0 | 0 | 0 |  |
| `tl.make_block_ptr(...)` | rule | `tritonExpr` `PositionalBlockPointer.lean:4`, `SignedRankOneBlockPointers.lean:10`, `Syntax.lean:238, 241, 244` | 0 | 0 | 21 |  |
| `tl.make_block_ptr(base=)` | rule | a keyword position exists (`SignedRankOneBlockPointers.lean:10`, `Syntax.lean:238, 241, 244`), so the expander decides; by reading `Expansion/Main.lean:1818`: only with the keywords `base`, `shape`, `strides`, `offsets`, `block_shape` all given, in that order | 0 | 0 | 9 |  |
| `tl.make_block_ptr(block_shape=)` | rule | a keyword position exists (`SignedRankOneBlockPointers.lean:10`, `Syntax.lean:238, 241, 244`), so the expander decides; by reading `Expansion/Main.lean:1822`: only with the keywords `base`, `shape`, `strides`, `offsets`, `block_shape` all given, in that order | 0 | 0 | 9 |  |
| `tl.make_block_ptr(offsets=)` | rule | a keyword position exists (`SignedRankOneBlockPointers.lean:10`, `Syntax.lean:238, 241, 244`), so the expander decides; by reading `Expansion/Main.lean:1821`: only with the keywords `base`, `shape`, `strides`, `offsets`, `block_shape` all given, in that order | 0 | 0 | 9 |  |
| `tl.make_block_ptr(order=)` | rule | a keyword position exists (`SignedRankOneBlockPointers.lean:10`, `Syntax.lean:238, 241, 244`), so the expander decides; by reading `Expansion/Main.lean:1869`: only after the other five keywords; its value is dropped | 0 | 0 | 11 |  |
| `tl.make_block_ptr(shape=)` | rule | a keyword position exists (`SignedRankOneBlockPointers.lean:10`, `Syntax.lean:238, 241, 244`), so the expander decides; by reading `Expansion/Main.lean:1819`: only with the keywords `base`, `shape`, `strides`, `offsets`, `block_shape` all given, in that order | 0 | 0 | 9 |  |
| `tl.make_block_ptr(strides=)` | rule | a keyword position exists (`SignedRankOneBlockPointers.lean:10`, `Syntax.lean:238, 241, 244`), so the expander decides; by reading `Expansion/Main.lean:1820`: only with the keywords `base`, `shape`, `strides`, `offsets`, `block_shape` all given, in that order | 0 | 0 | 9 |  |
| `tl.make_tensor_descriptor(...)` | no rule | no declaration spells `tl.make_tensor_descriptor(` | 0 | 5 | 0 | #76 |
| `tl.make_tensor_descriptor(padding_option=)` | no rule | the call has no rule | 0 | 0 | 0 | #76 |
| `tl.make_tensor_descriptor(block_shape=)` | no rule | the call has no rule | 0 | 5 | 0 | #76 |
| `tl.make_tensor_descriptor(shape=)` | no rule | the call has no rule | 0 | 5 | 0 | #76 |
| `tl.make_tensor_descriptor(strides=)` | no rule | the call has no rule | 0 | 5 | 0 | #76 |
| `tl.map_elementwise(...)` | no rule | no declaration spells `tl.map_elementwise(` | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.map_elementwise(pack=)` | no rule | the call has no rule | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.math` (module) | rule | a declaration spells a name under it: `tritonExpr` `Syntax.lean:71, 75, 80, 90` | - | - | - |  |
| `tl.math.exp2(...)` | rule | `tritonExpr` `Syntax.lean:71` | 0 | 0 | 17 |  |
| `tl.math.log2(...)` | rule | `tritonExpr` `Syntax.lean:75` | 0 | 0 | 5 |  |
| `tl.math.rsqrt(...)` | rule | `tritonExpr` `Syntax.lean:80` | 0 | 0 | 6 |  |
| `tl.math.sin(...)` | rule | `tritonExpr` `Syntax.lean:90` | 0 | 0 | 1 |  |
| `tl.max(...)` | rule | `tritonExpr` `Syntax.lean:219, 221`; `tritonStmt` `Syntax.lean:396-397, 415-416` | 5 | 3 | 44 |  |
| `tl.max(axis=)` | rule | a keyword position exists (`Syntax.lean:200-201, 210, 212`), so the expander decides; by reading `Expansion/Compute.lean:965`: a numeric literal | 5 | 0 | 16 |  |
| `tl.max(return_indices=)` | partial | a keyword position exists (`Syntax.lean:205-206, 210, 212`), so the expander decides; by reading `Expansion/Compute.lean:984`: `return_indices=True` is rejected in an expression; the tuple statement `v, i = tl.max(...)` has a declaration of its own | 0 | 0 | 2 | none: no target kernel uses it |
| `tl.max(return_indices_tie_break_left=)` | partial | a keyword position exists (`Syntax.lean:210, 212`), so the expander decides; by reading `Expansion/Main.lean:142`: only in the tuple statement `v, i = tl.max(...)`, and only the literal `True` | 0 | 0 | 1 | none: no target kernel uses it |
| `tl.max(keep_dims=)` | rule | a keyword position exists (`Syntax.lean:203-204, 210, 212`), so the expander decides; by reading `Expansion/Compute.lean:978` | 0 | 3 | 0 |  |
| `tl.max_constancy(...)` | no rule | no declaration spells `tl.max_constancy(` | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.max_contiguous(...)` | rule | `tritonExpr` `Syntax.lean:152` | 0 | 0 | 3 |  |
| `tl.maximum(...)` | partial | `tritonExpr` `Syntax.lean:105`; a keyword below has no rule | 1 | 4 | 31 | see its keyword rows |
| `tl.maximum(propagate_nan=)` | no rule | no declaration of `tl.maximum(` has a position for this keyword | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.min(...)` | partial | `tritonExpr` `Syntax.lean:220`; a keyword below has no rule | 0 | 0 | 1 | see its keyword rows |
| `tl.min(axis=)` | rule | a keyword position exists (`Syntax.lean:200-201, 210, 212`), so the expander decides; by reading `Expansion/Compute.lean:817`: a numeric literal or `None` | 0 | 0 | 0 |  |
| `tl.min(return_indices=)` | partial | a keyword position exists (`Syntax.lean:205-206, 210, 212`), so the expander decides; by reading `Expansion/Compute.lean:833`: `return_indices=True` is rejected; `False` is accepted | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.min(return_indices_tie_break_left=)` | no rule | a keyword position exists (`Syntax.lean:210, 212`), so the expander decides; by reading `Expansion/Compute.lean:856`: rejected | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.min(keep_dims=)` | rule | a keyword position exists (`Syntax.lean:203-204, 210, 212`), so the expander decides; by reading `Expansion/Compute.lean:828` | 0 | 0 | 0 |  |
| `tl.minimum(...)` | partial | `tritonExpr` `Syntax.lean:106`; a keyword below has no rule | 0 | 6 | 16 | see its keyword rows |
| `tl.minimum(propagate_nan=)` | no rule | no declaration of `tl.minimum(` has a position for this keyword | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.multiple_of(...)` | rule | `tritonExpr` `Syntax.lean:151` | 0 | 8 | 16 |  |
| `tl.num_programs(...)` | rule | `tritonExpr` `Syntax.lean:43-44` | 0 | 46 | 8 |  |
| `tl.num_programs(axis=)` | rule | `axis=` is spelled in `Syntax.lean:44` | 0 | 24 | 0 |  |
| `tl.pair_uniform_to_normal(...)` | no rule | no declaration spells `tl.pair_uniform_to_normal(` | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.permute(...)` | rule | `tritonExpr` `Syntax.lean:265` | 0 | 2 | 0 |  |
| `tl.philox(...)` | partial | `tritonStmt` `Syntax.lean:373`; a keyword below has no rule | 0 | 0 | 1 | see its keyword rows |
| `tl.philox(n_rounds=)` | no rule | no declaration of `tl.philox(` has a position for this keyword | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.philox_impl(...)` | no rule | no declaration spells `tl.philox_impl(` | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.philox_impl(n_rounds=)` | no rule | the call has no rule | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.pi32_t` | no rule | no declaration spells `tl.pi32_t` | - | - | - | not determined: uses are not counted |
| `tl.pointer_type(...)` | no rule | no declaration spells `tl.pointer_type(` | 0 | 32 | 0 | #77 |
| `tl.pointer_type(address_space=)` | no rule | the call has no rule | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.pointer_type(const=)` | no rule | the call has no rule | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.program_id(...)` | rule | `tritonExpr` `Syntax.lean:41-42` | 23 | 82 | 178 |  |
| `tl.program_id(axis=)` | rule | `axis=` is spelled in `Syntax.lean:42` | 0 | 34 | 47 |  |
| `tl.rand(...)` | rule | `tritonExpr` `Syntax.lean:48-50` | 0 | 0 | 3 |  |
| `tl.rand(n_rounds=)` | rule | `n_rounds=` is spelled in `Syntax.lean:50` | 0 | 0 | 1 |  |
| `tl.rand4x(...)` | no rule | no declaration spells `tl.rand4x(` | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.rand4x(n_rounds=)` | no rule | the call has no rule | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.randint(...)` | no rule | no declaration spells `tl.randint(` | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.randint(n_rounds=)` | no rule | the call has no rule | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.randint4x(...)` | no rule | no declaration spells `tl.randint4x(` | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.randint4x(n_rounds=)` | no rule | the call has no rule | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.randn(...)` | no rule | no declaration spells `tl.randn(` | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.randn(n_rounds=)` | no rule | the call has no rule | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.randn4x(...)` | no rule | no declaration spells `tl.randn4x(` | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.randn4x(n_rounds=)` | no rule | the call has no rule | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.range(...)` | no rule | no declaration spells `tl.range(` | 0 | 9 | 0 | #77 |
| `tl.range(arg2=)` | no rule | the call has no rule | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.range(step=)` | no rule | the call has no rule | 0 | 3 | 0 | #77 |
| `tl.range(num_stages=)` | no rule | the call has no rule | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.range(loop_unroll_factor=)` | no rule | the call has no rule | 0 | 3 | 0 | #77 |
| `tl.range(disallow_acc_multi_buffer=)` | no rule | the call has no rule | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.range(flatten=)` | no rule | the call has no rule | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.range(warp_specialize=)` | no rule | the call has no rule | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.range(disable_licm=)` | no rule | the call has no rule | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.ravel(...)` | partial | `tritonExpr` `Syntax.lean:268`; a keyword below has no rule | 0 | 0 | 0 | see its keyword rows |
| `tl.ravel(can_reorder=)` | no rule | no declaration of `tl.ravel(` has a position for this keyword | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.reduce(...)` | no rule | no declaration spells `tl.reduce(` | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.reduce(keep_dims=)` | no rule | the call has no rule | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.reduce_or(...)` | no rule | no declaration spells `tl.reduce_or(` | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.reduce_or(keep_dims=)` | no rule | the call has no rule | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.reshape(...)` | partial | `tritonExpr` `PythonTensorTuples.lean:12`, `Syntax.lean:266`, `WelfordCallMetadata.lean:15`; a keyword below has no rule | 0 | 6 | 3 | see its keyword rows |
| `tl.reshape(can_reorder=)` | no rule | no declaration of `tl.reshape(` has a position for this keyword | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.rsqrt(...)` | rule | `tritonExpr` `Syntax.lean:81` | 0 | 0 | 1 |  |
| `tl.sigmoid(...)` | rule | `tritonExpr` `Syntax.lean:76` | 4 | 0 | 5 |  |
| `tl.sin(...)` | rule | `tritonExpr` `Syntax.lean:89` | 0 | 0 | 4 |  |
| `tl.slice(...)` | no rule | no declaration spells `tl.slice(` | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.softmax(...)` | no rule | no declaration spells `tl.softmax(` | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.softmax(dim=)` | no rule | the call has no rule | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.softmax(keep_dims=)` | no rule | the call has no rule | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.softmax(ieee_rounding=)` | no rule | the call has no rule | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.sort(...)` | partial | `tritonExpr` `Syntax.lean:228`; a keyword below has no rule | 0 | 0 | 0 | see its keyword rows |
| `tl.sort(dim=)` | no rule | a keyword position exists (`Syntax.lean:210, 212`), so the expander decides; by reading `Expansion/Compute.lean:720`: rejected: the macro spells the axis of `tl.sort` `axis`, Triton spells it `dim` | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.sort(descending=)` | no rule | a keyword position exists (`Syntax.lean:210, 212`) and no file mentions `descending` | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.split(...)` | rule | `tritonExpr` `Syntax.lean:273`; `tritonStmt` `PythonTensorTuples.lean:13` | 0 | 2 | 2 |  |
| `tl.sqrt(...)` | rule | `tritonExpr` `Syntax.lean:79` | 2 | 0 | 11 |  |
| `tl.sqrt_rn(...)` | no rule | no declaration spells `tl.sqrt_rn(` | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.static_assert(...)` | partial | `tritonStmt` `Syntax.lean:421`; a keyword below has no rule | 0 | 26 | 6 | see its keyword rows |
| `tl.static_assert(msg=)` | no rule | no declaration of `tl.static_assert(` has a position for this keyword | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.static_print(...)` | partial | `tritonStmt` `Syntax.lean:420`; a keyword below has no rule | 0 | 0 | 0 | see its keyword rows |
| `tl.static_print(sep=)` | no rule | no declaration of `tl.static_print(` has a position for this keyword | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.static_print(end=)` | no rule | no declaration of `tl.static_print(` has a position for this keyword | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.static_print(file=)` | no rule | no declaration of `tl.static_print(` has a position for this keyword | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.static_print(flush=)` | no rule | no declaration of `tl.static_print(` has a position for this keyword | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.static_range(...)` | partial | `tritonStmt` `StaticRanges.lean:10`, `Syntax.lean:430-431`; a keyword below has no rule | 0 | 3 | 3 | see its keyword rows |
| `tl.static_range(arg2=)` | no rule | no declaration of `tl.static_range(` has a position for this keyword | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.static_range(step=)` | no rule | no declaration of `tl.static_range(` has a position for this keyword | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.store(...)` | partial | `tritonStmt` `Syntax.lean:404-406`; a keyword below has no rule | 23 | 62 | 183 | see its keyword rows |
| `tl.store(mask=)` | rule | a keyword position exists (`Syntax.lean:186-187, 190`), so the expander decides; by reading `Expansion/Memory.lean:371` | 3 | 50 | 149 |  |
| `tl.store(boundary_check=)` | rule | a keyword position exists (`SignedRankOneBlockPointers.lean:13`, `Syntax.lean:186-187, 190-194`), so the expander decides; by reading `Expansion/Memory.lean:343` | 0 | 0 | 17 |  |
| `tl.store(cache_modifier=)` | no rule | a keyword position exists (`Syntax.lean:186-187, 190`) and no file mentions `cache_modifier` | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.store(eviction_policy=)` | rule | a keyword position exists (`Syntax.lean:186-187, 190, 196-198`), so the expander decides; by reading `Expansion/Memory.lean:338`: only the literals `"evict_first"` and `"evict_last"`, which are dropped as hints | 0 | 0 | 1 |  |
| `tl.store_tensor_descriptor(...)` | no rule | no declaration spells `tl.store_tensor_descriptor(` | 0 | 0 | 0 | #76 |
| `tl.sum(...)` | rule | `tritonExpr` `Syntax.lean:214, 217` | 9 | 8 | 76 |  |
| `tl.sum(axis=)` | rule | `axis=` is spelled in `Syntax.lean:217` | 9 | 2 | 40 |  |
| `tl.sum(keep_dims=)` | rule | `keep_dims=` is spelled in `Syntax.lean:217` | 0 | 0 | 0 |  |
| `tl.sum(dtype=)` | rule | a keyword position exists (`Syntax.lean:210, 212`), so the expander decides; by reading `Expansion/Compute.lean:836` | 0 | 0 | 0 |  |
| `tl.swizzle2d(...)` | no rule | no declaration spells `tl.swizzle2d(` | 0 | 5 | 0 | #77 |
| `tl.target_info` (module) | no rule | no declaration spells a name under it | - | - | - | not determined: uses are not counted |
| `tl.tensor(...)` | no rule | no declaration spells `tl.tensor(` | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.tensor_descriptor(...)` | no rule | no declaration spells `tl.tensor_descriptor(` | 0 | 0 | 0 | #76 |
| `tl.topk(...)` | no rule | no declaration spells `tl.topk(` | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.topk(dim=)` | no rule | the call has no rule | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.trans(...)` | rule | `tritonExpr` `Syntax.lean:264` | 0 | 2 | 10 |  |
| `tl.tuple(...)` | no rule | no declaration spells `tl.tuple(` | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.tuple(type=)` | no rule | the call has no rule | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.uint16` | rule | `tritonDType` `Syntax.lean:173` | - | - | - |  |
| `tl.uint32` | rule | `tritonDType` `Syntax.lean:174` | - | - | - |  |
| `tl.uint64` | rule | `tritonDType` `Syntax.lean:175` | - | - | - |  |
| `tl.uint8` | rule | `tritonDType` `Syntax.lean:172` | - | - | - |  |
| `tl.uint_to_uniform_float(...)` | no rule | no declaration spells `tl.uint_to_uniform_float(`; the bare name has one: `tritonStmt` `Syntax.lean:358` | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.umulhi(...)` | no rule | no declaration spells `tl.umulhi(` | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.view(...)` | rule | `tritonExpr` `Syntax.lean:267` | 0 | 0 | 0 |  |
| `tl.void` | no rule | no declaration spells `tl.void` | - | - | - | not determined: uses are not counted |
| `tl.where(...)` | rule | `tritonExpr` `Syntax.lean:112` | 1 | 17 | 85 |  |
| `tl.xor_sum(...)` | no rule | no declaration spells `tl.xor_sum(` | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.xor_sum(axis=)` | no rule | the call has no rule | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.xor_sum(keep_dims=)` | no rule | the call has no rule | 0 | 0 | 0 | none: no target kernel uses it |
| `tl.zeros(...)` | rule | `tritonExpr` `CallSignature.lean:65-67`, `Syntax.lean:290-293` | 0 | 24 | 88 |  |
| `tl.zeros(dtype=)` | rule | a keyword position exists (`CallSignature.lean:67`, `Syntax.lean:292-293`), so the expander decides; by reading `Expansion/Main.lean:2257` | 0 | 24 | 85 |  |
| `tl.zeros_like(...)` | rule | `tritonExpr` `Syntax.lean:294` | 0 | 0 | 1 |  |

### Python forms

The statement kinds of Python that a kernel body can hold, and the calls in the scanned kernels that are not names of `triton.language`: builtins, methods (a leading dot) and functions of the same file. Keyword rows are the keywords the kernels pass.

| construct | status | basis | Triton (23) | Triton-distributed (119) | bench (183 files) | ticket |
| --- | --- | --- | --- | --- | --- | --- |
| `Assign` | rule | `Syntax.lean:399` | 23 | 116 | 183 |  |
| `AugAssign` | partial | `+=` `-=` `*=` `/=`: `Syntax.lean:400-403`; no declaration for `//=` `%=` `**=` `<<=` `>>=` `&=` `\|=` `^=` `@=` | 0 | 44 | 106 | #77 if it is a gap |
| `AnnAssign` | partial | only with the annotation `tl.constexpr`: `SourceDType.lean:16` | 0 | 30 | 8 | #77 if it is a gap |
| `Expr` | partial | follows the call: a call stands as a statement only through a `tritonStmt` declaration, which its row then cites | 23 | 109 | 183 | see the calls |
| `For` | rule | over `range(...)`, `tl.static_range(...)`: `CallSignature.lean:68-70`, `StaticRanges.lean:10`, `Syntax.lean:425-431` | 3 | 88 | 105 |  |
| `While` | rule | `RawBodyControl.lean:5` | 0 | 29 | 5 |  |
| `If` | rule | `Syntax.lean:439, 441, 443, 446-447, 449, 451-452` | 0 | 82 | 87 |  |
| `With` | no rule | no `tritonStmt` declaration has this form; by reading `RawSourceText.lean:230`: pasted Python source is rejected for any compound statement other than `if`, `for` and `while` | 0 | 9 | 0 | none in A1: every target kernel that uses it calls `dl.simt_exec_region` (epic A2, #300) |
| `Return` | rule | `StatementJitFunctions.lean:26`, `Syntax.lean:422` | 0 | 43 | 49 |  |
| `Pass` | no rule | no `tritonStmt` declaration has this form; other categories have one: `controlSourceStmt` `ControlSource.lean:61` | 0 | 15 | 1 | #77 |
| `Break` | no rule | no `tritonStmt` declaration has this form | 0 | 0 | 0 | none: no target kernel uses it |
| `Continue` | no rule | no `tritonStmt` declaration has this form | 0 | 0 | 0 | none: no target kernel uses it |
| `Assert` | no rule | no `tritonStmt` declaration has this form | 0 | 0 | 0 | none: no target kernel uses it |
| call of a local function decorated `triton.constexpr_function` | not determined | no mechanical test for the call of a function of the same file | 0 | 1 | 0 | #77 if it is a gap |
| call of a local function decorated `triton.jit` | not determined | no mechanical test for the call of a function of the same file; by reading `StatementJitFunctions.lean:23`: a call of a named helper has a declaration, and the helper must be bound outside the block; whether the calls of the kernels fit was not checked | 0 | 25 | 24 | #77 if it is a gap |
| call of a local function decorated `triton.language.core.extern` | not determined | no mechanical test for the call of a function of the same file | 0 | 9 | 0 | #77 if it is a gap |
| `.bit_length(...)` | no rule | no declaration spells `.bit_length(` | 0 | 1 | 0 | #77 |
| `.is_floating(...)` | partial | no declaration spells `.is_floating(`; by reading `SourceDType.lean:57`: recognised only in the dotted name `<pointer>.dtype.element_ty.is_floating` | 0 | 0 | 1 | none: no target kernel uses it |
| `.is_fp64(...)` | partial | no declaration spells `.is_fp64(`; by reading `FiniteHelperSource.lean:57`: `x.dtype.is_fp64()` parses as one dotted identifier (`Syntax.lean:368`) and is accepted only inside a checked source helper | 0 | 0 | 1 | none: no target kernel uses it |
| `.is_int(...)` | partial | no declaration spells `.is_int(`; by reading `SourceDType.lean:54`: recognised only in the dotted name `<pointer>.dtype.element_ty.is_int` | 0 | 0 | 1 | none: no target kernel uses it |
| `.is_int1(...)` | partial | no declaration spells `.is_int1(`; by reading `SourceDType.lean:55`: recognised only in the dotted name `<pointer>.dtype.element_ty.is_int1` | 0 | 0 | 1 | none: no target kernel uses it |
| `.is_int_signed(...)` | partial | no declaration spells `.is_int_signed(`; by reading `SourceDType.lean:56`: recognised only in the dotted name `<pointer>.dtype.element_ty.is_int_signed` | 0 | 0 | 1 | none: no target kernel uses it |
| `.load(...)` | no rule | no declaration spells `.load(` | 0 | 5 | 0 | #76 |
| `.record(...)` | no rule | no declaration spells `.record(` | 0 | 15 | 0 | #77 |
| `.record(is_start=)` | no rule | the call has no rule | 0 | 15 | 0 | #77 |
| `.record(task_type=)` | no rule | the call has no rule | 0 | 15 | 0 | #77 |
| `.store(...)` | no rule | no declaration spells `.store(` | 0 | 4 | 0 | #76 |
| `.sum(...)` | no rule | no declaration spells `.sum(` | 0 | 1 | 0 | #77 |
| `.sum(axis=)` | no rule | the call has no rule | 0 | 1 | 0 | #77 |
| `.to(...)` | rule | `tritonExpr` `ChainedSourceCasts.lean:9-10`, `MutableScalarCalls.lean:11`, `PastedControlSource.lean:20-22`, `PythonBitcastBooleans.lean:10`, `Syntax.lean:120-122, 132-137, 147-150, 232-237` | 13 | 44 | 154 |  |
| `.to(bitcast=)` | rule | `bitcast=` is spelled in `MutableScalarCalls.lean:11`, `PastedControlSource.lean:20-22`, `PythonBitcastBooleans.lean:10`, `Syntax.lean:120-122` | 0 | 0 | 7 |  |
| `bin(...)` | no rule | no declaration spells `bin(` | 0 | 1 | 0 | #77 |
| `float(...)` | rule | `tritonExpr` `Syntax.lean:154-155, 159` | 1 | 1 | 38 |  |
| `len(...)` | no rule | no declaration spells `len(` | 0 | 4 | 0 | #77 |
| `min(...)` | rule | `tritonExpr` `Syntax.lean:104` | 0 | 21 | 28 |  |
| `range(...)` | partial | `tritonStmt` `CallSignature.lean:68-70`, `Syntax.lean:425-429`; a keyword below has no rule | 3 | 80 | 104 | see its keyword rows |
| `range(step=)` | no rule | no declaration of `range(` has a position for this keyword | 0 | 1 | 0 | #77 |
| `torch._inductor.runtime.triton_helpers.libdevice.rsqrt(...)` | rule | under the spelling `libdevice.rsqrt(`: `tritonExpr` `Syntax.lean:82` | 0 | 0 | 2 |  |
| `torch._inductor.runtime.triton_helpers.welford(...)` | rule | under the spelling `triton_helpers.welford(`: `tritonStmt` `Syntax.lean:388`, `WelfordCallMetadata.lean:13` | 0 | 0 | 1 |  |
| `torch._inductor.runtime.triton_helpers.welford_reduce(...)` | rule | under the spelling `triton_helpers.welford_reduce(`: `tritonStmt` `Syntax.lean:386` | 0 | 0 | 1 |  |
| `triton.cdiv(...)` | rule | `tritonExpr` `PackedIntegerPromotion.lean:12` | 0 | 0 | 2 |  |
| `tl.libdevice.finitef(...)` | rule | under the spelling `libdevice.finitef(`: `tritonExpr` `Syntax.lean:130` | 0 | 0 | 1 |  |
| `tl.libdevice.isfinited(...)` | rule | under the spelling `libdevice.isfinited(`: `tritonExpr` `Syntax.lean:131` | 0 | 0 | 1 |  |
| `tl.libdevice.pow(...)` | no rule | no declaration spells `tl.libdevice.pow(` | 0 | 0 | 1 | none: no target kernel uses it |
| `triton.next_power_of_2(...)` | no rule | no declaration spells `triton.next_power_of_2(` | 0 | 2 | 0 | #77 |
| `triton_tanh(...)` | rule | `tritonExpr` `Syntax.lean:85` | 0 | 0 | 1 |  |

75 further call names of the scanned kernels are not rows: names under `triton_dist` and names from relative imports. They are the requirement for the Triton-distributed front end (epic A2, #300) and are listed in `documents/census/MinimalSyntax.md`.

### Rules for names outside `__all__`

30 names after `tl.` occur in a declaration and are not in `triton.language.__all__` at v3.5.1. They are spellings of the macro itself, internal forms, or names of other Triton versions: `tl.__finite_helper_resolved`, `tl.__philox_profiled`, `tl.__rand_profiled`, `tl.__signed_and64_profiled`, `tl.__signed_shift64_profiled`, `tl.__uniform32_profiled`, `tl.__unsigned_add_profiled`, `tl.__unsigned_mul_profiled`, `tl.async_copy`, `tl.async_wait`, `tl.atan`, `tl.bitcast`, `tl.call_signature`, `tl.cosh`, `tl.for`, `tl.if`, `tl.implicit_store_cast`, `tl.logical_and`, `tl.logical_not`, `tl.logical_or`, `tl.param`, `tl.ports`, `tl.region`, `tl.sinh`, `tl.source_alias`, `tl.source_helper`, `tl.source_integer`, `tl.tan`, `tl.tanh`, `tl.toReal`.

<!-- generated by scripts/census/triton_coverage.py: end -->
