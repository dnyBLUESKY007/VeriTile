# Bridge theorem: existing lemmas and differences

Ticket I4.1. This document supports the bridge theorem of
[the interpreter interface](InterpInterface.md):

```lean
theorem bridge_exec {ck : ComputeKernel} {ak : AlgKernel}
    (s : FP.Structural.State (WithBot ℝ))
    (h : ck.toAlgorithm? = .ok ak) (pre : BridgePre ck s) :
    (FP.Structural.exec realAlgebra ck s).map erase = exec ak (erase s)
```

The statement was first signed over the carrier `ℝ`. The proof semantics
computes floating values in `WithBot ℝ` and uses `⊥` for `-inf`; over `ℝ` a
kernel that keeps `-inf` in a register could not be covered. The user decided
on 2026-11-12 to state the bridge over `WithBot ℝ`. This document describes
that version.

It has three parts: how `realAlgebra` and `erase` are defined
([`VeriTile/Interp/Bridge/RealAlgebra.lean`](../VeriTile/Interp/Bridge/RealAlgebra.lean)),
which upstream lemmas already cover part of the proof, and where the two
execution definitions differ. Each difference names the side condition it
needs; tickets I4.2 and I4.3 turn these into the conjuncts of `BridgePre`.

"Structural side" means `FP.Structural.exec realAlgebra` on the source kernel
(`VeriTile/Triton/Float/Structural.lean`). "Proof side" means `exec` on the
projected kernel (`VeriTile/Triton/Semantics/EvalOp.lean`, `Step.lean`).
Upstream sources were read at `18bd31dcb`.

Every claim carries one of four marks:

| Mark | Meaning |
| --- | --- |
| checked | Lean accepted a proof of it in a scratch file. The file is not part of the repository. |
| read | Both sides were read in the source and the claim restates them. |
| inferred | Follows from what was read, by an argument that was not run through Lean. |
| unchecked | Not examined. |

## 1. The two definitions

At every dtype the structural value type over `WithBot ℝ` and the carrier of
the proof semantics are the same type (checked: `toCarrier` and `ofCarrier`
are inverse and each is the identity at every dtype). So a register needs no
translation, and a field of `realAlgebra` can be the function the real
evaluator applies.

### `realAlgebra`

Every field ignores its precision and dtype arguments.

| Fields | Definition | Proof-side counterpart | Mark |
| --- | --- | --- | --- |
| `literal` | `some c` | `Op.const c` gives `some c`; `Op.constFloat d c` gives `d.ofReal c` | checked |
| `negInf` | `⊥` | `Op.negInf` gives `⊥` | checked |
| `binary` | `WithBot.realAdd`, `realSub`, `realMul`, `realDiv`, `max`, `WithBot.realPow` | The same functions | checked for `add`, `sub`, `mul`, `div` at every numeric dtype, `max2`, `maximumFloat`, `pow` |
| `unary` | `WithBot.realExp`, `realLog` and the other `WithBot.real*` functions; `neg` and `abs` as `WithBot.map`; `minPosInf` as identity | `Tile.uop` of the same function; `WithBot.map` for `negFloat` and `absFloat` | checked for all 22 unary constructors, `negFloat` and `absFloat` |
| `fma`, `fmaTyped` | `realAdd (realMul a b) c` | The projection rewrites both nodes to `Op.add (Op.mul a b) c` | checked |
| `atomicAdd` | `WithBot.realAdd` | `NumericDType.add` on the value read back | checked on one kernel (region address, no mask, scalar); see D1 |
| `libdevicePowi` | `WithBot.realPowi` | `Tile.libdevicePowi` | checked |
| `compareLt`, `compareLe`, `compareEq` | `decide` of the order and equality of `WithBot ℝ` | `ComparableDType.lt`, `le`, `eq`, `gt`, `ge`, `ne` | checked for the six comparisons at every comparable dtype |
| `cast`, `fp32Load` | Identity | `FloatDType.cast` is the identity; an `fp32` compute load projects to a `.real` load | checked |
| `fromNat`, `fromInt` | The coercion through `ℝ` | `Tile.natToReal`, `Tile.intToReal` | checked |
| `fp32Bits` | `Float32Bits.decodeReal` | A projected `fp32` constant is `Op.const` of `Float32Bits.decodeRat`; random values are the rational value of the same bits | checked for a constant with a finite encoding and for `uniform32`; inferred for `rand`, `randTensor` and bitcasts |
| `float8E5Bits` | `decodeRat` of the byte, `0` when it has none | `Float8E5Bits.decodeTile`; see D5 | read |
| `minimum`, `divRN` | `min`, `WithBot.realDiv` | `min`, `NumericDType.real.div` | checked |
| `reduceSum`, `reduceMin`, `reduceMax`, `reduceSumTyped`, `reduceMaxTyped`, `scan`, `dot`, `dotWithOptions`, `integerDot` | The upstream `Tile.reduceSum`, `Tile.reduceMin`, `Tile.reduceMax`, `Tile.reduceSumTyped`, `Tile.reduceMaxTyped`, `Tile.scan`, `Tile.dot`, `Tile.dotInt` on the same tile | The same function | checked |
| `dotTyped`, `dotAcc` | `Tile.dot`, then `WithBot.realAdd` of the accumulator | `Tile.dot` between identity casts, then `WithBot.realAdd` | checked |
| `indexedMax` | `Tile.reduceMaxTyped … false` for the value, `Tile.argMaxTyped` for the index | `Tile.reduceMaxTyped … false`, `Tile.argMaxDrop`, `Tile.argMaxTyped` | checked; the index needs a non-empty axis, see D6 |
| `floatToInt`, `floatToNat` | Left at the default `none` | The real evaluator returns `none` on both | checked: both sides fail |
| `isFinite` | Left at the default `none` | None: the projection rejects `ComputeExpr.isFinite` | read |

`reduceMax` is total in `Structural.Algebra` but partial upstream; it returns
the all-`⊥` tile on an empty axis. Structural execution never calls it there
(checked: the two sides of `Op.reduceMax` agree on every axis).

The fields above are the whole structure: 31 are set and 3 keep their default.

What "checked" means for an expression constructor `C`: for every precision
context `p` and every state `s`, if each sub-expression `e` satisfies

```lean
(Structural.evalOp realAlgebra p e s).map toTile = evalOp e (erase s)
```

then so does `C` applied to them. This is one step of the induction that
I4.2 needs, and it holds for every value, `⊥` included. It was proved for 97
of the 113 constructors of `Op`. The other 16 are `load` (D1, D8, D13),
`castRealToInt8` (D3), `argMin` and `sort` (D4), `decodeFloat8E5` (D5),
`reshape` (D7), and ten that were only read: `rand`, `randTensor`,
`philox4x32`, `ptrBase`, the four `makeBlockPtr` forms and the two
`advanceBlockPtr` forms. `argMax` and `argMaxTyped` were proved on a non-empty
axis (D6), `argMaxTyped` without a precision override.

### `erase`

| Component | Definition | Mark |
| --- | --- | --- |
| Register | Copied: `Option.map toTile` at the same dtype and shape | checked: `erase` commutes with a register update |
| Memory cell | `eraseCell`: `Cell.mk d v` becomes `MemCell.of d (storedValue d v)`, under the same dtype tag. `storedValue` replaces a floating `⊥` by `0` and changes nothing else | checked: `erase` commutes with a typed write, for every value, and with a masked scatter |
| Program ids, grid sizes | Copied | read |
| Undefined-value table | `r o ↦ (s.undef .real r o).unbotD 0` | checked on a masked `fp32` load, see D8 |

`erase` changes a value in two places, and in both the proof semantics holds a
real where the structural state holds a `WithBot ℝ`:

- **Memory.** The proof side writes `0` for a floating `⊥`
  (`BlockState.writeMemAs` stores `dtype.ofReal (dtype.storeValue v)`) and
  reads `0` for one (`BlockState.readMemAs` does the same) (read). The
  structural side copies the value into the cell and reads it back unchanged
  (read). `eraseCell` applies the proof side's rule, so a store of `⊥` gives
  the same final state on both sides (checked). An erasure that kept `⊥` in
  the cell would make every store of `⊥` a difference (inferred). What
  remains is D1.
- **The undefined-value table.** `BlockState.undef` has the type
  `RegionName → Nat → ℝ`; the structural state has one table per dtype with
  values in `WithBot ℝ`. See D8.

The interface once described `erase` as forgetting dtype tags. Cells and
registers keep their tag, because `toAlgorithm?` keeps `.fp32` and the other
floating dtypes in the projected kernel (read: `ComputeOp.toAlgorithm?`
returns the wrapped `Op` unchanged). A projected `Op.load .fp32` reads `0`
from a cell tagged `.real`, so an erasure that retagged cells would break
every typed load.

### What was reused

Every `WithBot.real*` function and every `Tile.*` operation named in the table
is the upstream one, applied to the same value; nothing is lifted from `ℝ` and
lowered again. `MemCell.of` builds the cell, `Float32Bits.decodeReal` and
`Float8E5Bits.decodeRat` carry the bit decoders, and `Structural.toFloat` and
`Structural.ofFloat` pass between the typed channels in `dotTyped` and
`dotAcc`. Five small functions are new: `toCarrier`, `ofCarrier`, `toTile`,
`ofTile` (the identity between the two value types) and `storedValue`.
`Float/Erasure.lean` and `Float/StateErasure.lean` were not reused; section 2
says why.

## 2. Inventory of upstream lemmas

| Lemma | File | What it proves | What remains for the bridge |
| --- | --- | --- | --- |
| `ComputeKernel.computeCorrectR_of_toAlgKernel` | `Float/Refine.lean` | `(h : ck.toAlgorithm? = Except.ok ck.toAlgKernel) (hc : Kernel.Correct R ck.toAlgKernel post) : ComputeCorrectR R ck post`. It unfolds a definition: correctness of the projected kernel under `execR R` is correctness of the compute kernel under `R` | Everything about structural execution. `execR` runs on `BlockState` with a rounding model and never mentions `FP.Structural`. The lemma is a pattern for the corollary of I4.3, which transfers `Kernel.Correct_without_Rounding` |
| `execR_triv`, `stepStmtR_triv`, `stepStmtsR_triv`, `evalOpR_triv` | `Float/StepR.lean`, `Float/EvalOpR.lean` | `execR RoundingModel.triv k s = exec k s`, and the same for one statement, a statement list and one expression | They relate two executions on `BlockState`. They are the nearest existing proof of the same shape, an induction over every constructor of `Op` and `Stmt` including the offset lists of dynamic block pointers; `evalOpR_triv_offsets` is private |
| `eraseDType`, `Op.eraseDType`, `MemAccess.eraseDType`, `MaskOpt.eraseDType`, `Stmt.eraseDType`, `Kernel.eraseDType`, `ComputeKernel.eraseDType`, and the witness maps `NumericDType.eraseDType`, `IntegralDType.eraseDType`, `ComparableDType.eraseDType` | `Float/Erasure.lean` | Definitions. They rewrite a kernel so that every floating dtype becomes `.real`: `constFloat` to `const`, `castFloat` to its operand, typed reductions and dots to the real ones | Not on the bridge's path: `bridge_exec` runs `ak` as projected, with its dtype tags. They would matter only for a bridge stated against `ak.eraseDType` |
| `eraseDType_float`, `MemAccess.eraseDType_region`, `_ptr`, `_blockPtr`, `MaskOpt.eraseDType_none`, `_mask`, `_maskOther`, `Op.eraseDType_ref`, `_castFloat`, `_load`, `_reduceMax`, `_reduceSum`, `Stmt.eraseDType_store`, `_atomicAdd`, `_atomicRMW`, `ComputeKernel.eraseDType_of_projection_error` | `Float/Erasure.lean` | Unfolding equations of the definitions above, and that a kernel whose projection fails is erased to itself. None mentions `evalOp` or `exec` | Same as above. No lemma in the repository states that erasure preserves execution (read: a search for theorems named `eraseDType` that mention `exec`, `evalOp` or a step function finds none) |
| `eraseDTypeCarrier`, `Tile.eraseDType`, `BlockState.eraseDTypeRegs`, `BlockState.eraseDType` | `Float/StateErasure.lean` | Definitions on `BlockState`: retag every floating cell and register as `.real` | Wrong domain and wrong tags for the bridge: the map needed goes from `Structural.State (WithBot ℝ)` and must keep tags. Two details (read): `BlockState.eraseDType` does not copy `numPids`, which falls back to the default `fun _ => 1`; its register map looks up `.real`, `.fp32`, `.fp16`, `.bf16` and drops `.f8e4` and `.f8e5` registers |
| `BlockState.eraseDType_pids`, `_pid`, `_undef` | `Float/StateErasure.lean` | The erased state has the same program ids and undefined-value table | Counterparts for `erase` are in `RealAlgebra.lean` (`erase_pids`, `erase_numPids`, `erase_undef`) |
| `Structural.stepAlg` | `Float/Structural.lean` | A definition, not a lemma: structural execution of an algorithm statement. It runs `assign`, `store` and `atomicAdd` with no precision context and returns `none` on every other `Stmt` | Agreement with `stepStmt` on the three statements it runs; a side condition for the others (D11) |
| `Structural.run_append`, `exec_seq_cons`, `exec_seq_nil` | `Float/Structural.lean` | `run M (a ++ b) s = (run M a s).bind (run M b)`, and the same for `ComputeKernel.seq` | Used as is for statement lists. The proof-side counterpart is `stepStmts.append_some_iff` (`Semantics/Step.lean`) |
| `Structural.State.setReg_same`, `setReg_ne_name`, `write_same`, `write_other`, `masked_scatter_eq_filter`, `scatter_frame` | `Float/Structural.lean` | Register and cell updates of a structural state read back as written | `erase` commutes with `setReg`, with a typed `write` and with a masked scatter (checked in the scratch file; the proofs are short and belong to I4.2) |
| `BlockState.ext`, `setReg_*`, `writeMemTyped_*` | `Semantics/State.lean` | Extensionality and the frame lemmas of the proof-side state | Used as is |
| `MemCell.readAs_of_same` | `Semantics/State.lean` | A cell read at its own dtype returns its value | Gives the proof-side half of D1: with `s.mem r o = Cell.mk d v`, `(erase s).readMemValue d r o = storedValue d v` (checked) |
| `DecoderCompatibility.equals_software_decoder` | `Float/DecoderCompatibility.lean` | `Float32Bits.decodeRat x = (FP.Value.mk x.bits : FP.Value .fp32).toRat?` | With `Random.uniform_toRat_uniformRat` it gives the `uniform32` step (checked). `rand` and `randTensor` need `Random.rand_toRat_randRat` in the same way (inferred) |
| `WithBot.sum_someTerm_eq_some`, `WithBot.unbotD_sum_some`, `WithBot.sup'_someTerm_eq_some`, `WithBot.unbotD_sup'_some`, `WithBot.real*_some` | `Semantics/TileOps.lean` | A sum or a maximum of `some` values is `some`, and each unary operation maps `some` to `some` | Not needed for the bridge any more: the algebra applies each tile operation to the same `WithBot ℝ` tile. They matter where a specification reads a stored result as a real |
| `Tile.reduceMaxDrop_eq_none`, `Tile.reduceMinDrop_eq_none`, `Tile.reduceMaxTyped_eq_none` | `Semantics/TileOps.lean` | A maximum or minimum over an empty axis is `none` | The typed one is used in the `reduceMaxTyped` and `indexedMaxValue` steps (checked); the other two state the fact that the `reduceMax` and `reduceMin` steps need |
| `Float8E5Bits.decodeTile_of_finite` | `Semantics/Float8.lean` | The byte decoder succeeds with the rational values when every byte is a valid `int8` with a finite encoding | Gives agreement under the side condition of D5 |
| `BlockState.paddingValue_of_undef_zero` | `Semantics/State.lean` | With an all-zero undefined-value table every padding value is the dtype default | Gives agreement under the strongest form of the side condition of D8 |
| `Structural.Safety.stepSafe`, `safeAlg` | `Float/StructuralSafety.lean` | Definitions that return `False` on every statement structural execution rejects | A ready description of the accepted statement fragment, mixed with memory bounds the bridge does not need |

No existing lemma relates `FP.Structural.exec` to `exec`, and no
`Structural.Algebra` over `ℝ` or `WithBot ℝ` existed (read: a search finds
concrete algebras only over bit-level values, words and syntactic terms).

## 3. Differences

`h` below is the hypothesis that the projection succeeds. A row is a
difference only when `h` can hold.

Compared with the version over `ℝ`: the old D1 (`Op.negInf` has no real
value) shrinks to memory cells, the old D2 (operations on `⊥`) is gone, D8
changes its wording, D13 is new, and D3 to D7 and D9 to D12 do not depend on
the carrier.

| Id | Construct | Structural side | Proof side | Side condition | Mark |
| --- | --- | --- | --- | --- | --- |
| D1 | A floating memory cell that holds `⊥`, read by a load or an atomic add. Such a cell comes from the initial state or from a store of `⊥` | The cell keeps `⊥`; the read returns `⊥` | A store writes `0` and a read returns `0` | No load and no atomic add reads a floating cell that holds `⊥`. It is enough that the initial memory holds no floating `⊥` and that the kernel stores none | checked: a load of such a cell differs, a load of a finite cell agrees; "store `-inf`, then load it" differs for every state; an atomic add onto a finite cell agrees for every added value, and onto a `⊥` cell it differs whenever the added value is neither `0` nor `⊥`. A store of `⊥` alone agrees |
| D2 | Operations on `⊥` in registers | — | — | None. No difference remains | checked: the step of every arithmetic, unary, comparison, selection and reduction constructor holds for every value |
| D3 | `Op.castRealToInt8` | `none` | Succeeds, truncating toward zero | The kernel does not use it | checked that the structural side fails; read for the proof side |
| D4 | `Op.argMin`, `Op.sort` | `none` | Succeed | The kernel does not use them | checked that the structural side fails; read for the proof side |
| D5 | `Op.decodeFloat8E5` on a byte whose encoding is an infinity or a NaN | Succeeds; `float8E5Bits` returns `0` for that byte | `none` for the whole tile | Every decoded byte has a finite encoding | read |
| D6 | `Op.argMax`, `Op.argMaxTyped` over an empty axis | `none` | Succeed with index `0` | The reduced axis is non-empty | checked for `argMax`: the sides differ on an empty axis; both constructors agree on a non-empty one |
| D7 | `Op.reshape` between shapes of different sizes | `none` | Succeeds; a lane without a source reads the dtype default | The two shapes have the same number of elements | read |
| D8 | Masked-off lane of a load without `other`, and out-of-bounds lane of a block-pointer load with unspecified padding | `s.undef d` at the address, for the dtype `d` of the load; a value of `WithBot ℝ` for a floating `d` | From the single real table `u = (s.undef .real · ·).unbotD 0`: a floating lane is `some u`; a masked-off discrete lane is the dtype default; an unspecified-padding lane is `⌊u⌋` for `Int` and `Nat`, `u ≠ 0` for `Bool`, the default for pointers | `s.undef d r o = some u` for each floating `d` that is loaded: no `⊥` in the tables, and the floating tables equal the `.real` one. The discrete tables agree with the values at left. A table that is `some 0` on the floating dtypes and the default on the discrete ones satisfies every case | checked for `.fp32`: the sides agree exactly when `s.undef .fp32 r o = some u`; inferred for the other floating dtypes; read for the discrete cases |
| D9 | `ComputeOp.const` of dtype `fp64`, `ComputeOp.load` of dtype `fp64` | `none` | Succeed when the constant decodes; the load is a `.real` load | The kernel has no `fp64` compute constant or load | read |
| D10 | `ComputeOp.bitcast` other than `int32` or `uint32` to `fp32` | `none` | Succeeds on a scalar constant payload, for example `uint32` to `int32` | The kernel has no such bitcast | read |
| D11 | `ComputeStmt.alg st` with `st` a loop, a conditional or an atomic read-modify-write | `stepAlg` returns `none` | `stepStmt` runs `st` | `ComputeStmt.alg` wraps only `assign`, `store` and `atomicAdd` | checked for `ifThen` |
| D12 | `ComputeStmt.atomicRMW` | `none` | `stepAtomicRMW`; exchange and compare-and-swap can succeed | The kernel has no atomic read-modify-write | read |
| D13 | A floating read of a cell tagged with another dtype. An `fp32` compute load reads at `.real`, so it meets this on a cell tagged `.fp32` | `Structural.defaultValue`, which is the default of the carrier: `(default : WithBot ℝ) = ⊥` | `0` | Every floating load and atomic add reads cells of its own dtype | checked: the sides differ for every value in the cell; a discrete read (`Nat`, `Int`, `Bool`) of a mistagged cell agrees |

D3, D4, D6, D7 and D9 to D12 have the same form: structural execution fails
and the proof side may succeed. One semantic conjunct, "structural execution
of `ck` from `s` succeeds", would cover all of them at once (inferred; the
choice between it and syntactic conjuncts belongs to I4.2). It does not cover
D1, D5, D8 and D13, where structural execution succeeds.

D1, D8 and D13 are conditions on the state as well as on the kernel. For an
initial state whose memory holds finite values under the dtype the kernel
loads them at, and whose undefined-value tables are zero everywhere, D8 holds,
and D1 and D13 reduce to conditions on the kernel: it does not load a `-inf`
it stored, and it loads what it stored at the dtype it stored it (inferred).

D13 did not exist over `ℝ`, whose default is `0`. It comes from the
`Inhabited (WithBot ℝ)` instance that `Structural.exec` takes as an implicit
argument. Passing the instance `⟨0⟩` explicitly in the statement would remove
D13, and would change the signed statement again (inferred; not done).

### Missing backend support

Structural execution fails when the algebra lacks an optional operation and
never falls back to real evaluation (read: the header of `Structural.lean`
and the `←` on each optional field). `realAlgebra` supplies every optional
field except `floatToInt` and `floatToNat`, on which the proof side fails as
well (checked), and `isFinite`, whose projection fails. So this rule causes
no difference for `realAlgebra`; the failures that remain are the hard-coded
`none` branches of D3, D4 and D9 to D12 (read).

### Constructs whose projection fails

For these `h` is false and the theorem says nothing (read in
`Core/Ast.lean`):

- `ComputeExpr.opaque`, `ComputeExpr.llrint`, `ComputeExpr.isFinite`;
- `ComputeStmt.effectMarker` other than `"tl.debug_barrier"`;
- an `fp32` constant whose bits are an infinity or a NaN; an `fp64` constant
  that is not a normal number, which includes zero;
- a bitcast that is not scalar, or whose operand is not a constant payload,
  or that mixes `fp64` with a 32-bit dtype.

`"tl.debug_barrier"` projects to `ifThen (constBool false) []`; both sides
return the state unchanged (read).

### Agreement that still needs a lemma

Both sides were read and look equal; no proof was run (inferred):

- `Op.load` in general. The cell-level facts are checked (D1, D13, and the
  two readings of one cell); the three address forms, the three mask forms
  and the padding rule were only read. With `other`, a masked-off lane takes
  the value of `other` on both sides, `⊥` included;
- `rand`, `randTensor` (through `Random.rand_toRat_randRat`) and `philox4x32`
  (both sides call `Random.Direct.evalField`);
- `Op.reshape` on equal sizes: `reshapeIndex` is the successful lookup of
  `Tile.reshape`;
- pointers and block pointers: `ptrBase`, the `makeBlockPtr` forms, the
  `advanceBlockPtr` forms, and the dynamic lists (`evalOffsets` and
  `evalDeltas` against `List.mapM`);
- compute constants and loads of dtype `int32` and `uint32`; the bitcast of
  an `int32` or `uint32` constant to `fp32`, which needs
  `BitVec.ofInt 32 c.toInt = c.bits`;
- stores through a pointer or a block pointer, atomic adds beyond the kernel
  that was checked, and the three loop forms and two conditionals: the folds
  have the same shape as the masked store that was checked.

### Unchecked

- Which bench kernels satisfy the side conditions. A text search finds
  `negInf` or `float("-inf")` in `bench/examples/OnlineSoftmax`,
  `bench/examples/StableLogSumExp` and `VeriTile/Examples/FlashAttention1`;
  whether any of them stores a `⊥` and loads it again was not examined. The
  DSL emits loops as `ComputeStmt.forLoop`, not under `ComputeStmt.alg`;
  `ComputeKernel.fromKernelBody` puts every statement under `alg` (read).
- The helper functions that both sides call by the same name
  (`BlockPtr.inBounds`, `BlockPtr.address`, `BlockPtr.advance`,
  `TileShape.*`, `Random.*`): only the call sites were compared.
- `Float/StepR.lean` and `Float/EvalOpR.lean` beyond the statements quoted in
  section 2.

## 4. Consequences for the statement

- On five small kernels the statement holds for every state, with no side
  condition (checked): assign a constant; assign `-inf`, which differed for
  every state over `ℝ`; store `-inf`; a masked store; `sum(exp(v))`.
- A register may hold `⊥`. Online softmax and flash attention seed a running
  maximum with `-inf`; `max ⊥ x = x`, `exp ⊥ = 0` and `0 * ⊥ = ⊥` are now the
  same function on both sides (checked).
- Memory cannot hold `-inf` on the proof side: a store writes `0` for it and a
  load reads a finite value. This is a property of the proof semantics, not
  of the bridge, and it is the reason for D1. A kernel that loads a tensor
  containing `-inf`, such as an additive attention mask, is outside both the
  proof semantics and the bridge (inferred).
- `BridgePre` therefore needs conjuncts about the initial state (D1, D8,
  D13), and the corollary of I4.3 speaks about initial states with finite
  memory (inferred).
- Nothing found here shows that the statement cannot hold. Every remaining
  difference has a side condition that a kernel without the constructs of D3
  to D12, run from a finite and correctly tagged memory, satisfies (inferred).
