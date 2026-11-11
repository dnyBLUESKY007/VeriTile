# Bridge theorem: existing lemmas and differences

Ticket I4.1. This document supports the bridge theorem of
[the interpreter interface](InterpInterface.md):

```lean
theorem bridge_exec {ck : ComputeKernel} {ak : AlgKernel} (s : FP.Structural.State ℝ)
    (h : ck.toAlgorithm? = .ok ak) (pre : BridgePre ck s) :
    (FP.Structural.exec realAlgebra ck s).map erase = exec ak (erase s)
```

It has three parts: how `realAlgebra` and `erase` are defined
([`VeriTile/Interp/Bridge/RealAlgebra.lean`](../VeriTile/Interp/Bridge/RealAlgebra.lean)),
which upstream lemmas already cover part of the proof, and where the two
execution definitions differ. Each difference names the side condition it
needs; tickets I4.2 and I4.3 turn these into the conjuncts of `BridgePre`.

"Structural side" means `FP.Structural.exec realAlgebra` on the source kernel
(`VeriTile/Triton/Float/Structural.lean`). "Proof side" means `exec` on the
projected kernel (`VeriTile/Triton/Semantics/EvalOp.lean`, `Step.lean`).
Upstream sources were read at `cc49c2ad1`.

Every claim carries one of four marks:

| Mark | Meaning |
| --- | --- |
| checked | Lean accepted a proof of it in a scratch file. The file is not part of the repository. |
| read | Both sides were read in the source and the claim restates them. |
| inferred | Follows from what was read, by an argument that was not run through Lean. |
| unchecked | Not examined. |

## 1. The two definitions

The proof side computes every floating channel in `WithBot ℝ`, where `⊥`
stands for `-inf`. A structural state over `ℝ` holds no `⊥`. So `erase` embeds
each real as `some`, and no state in its range holds `⊥` (read).

### `realAlgebra`

Every field ignores its precision and dtype arguments.

| Fields | Definition | Proof-side counterpart | Mark |
| --- | --- | --- | --- |
| `literal` | The literal itself | `Op.const c` gives `some c`; `Op.constFloat d c` gives `d.ofReal c` | checked for `const` |
| `negInf` | `0` | `Op.negInf` gives `⊥`. No real agrees; see D1 | checked |
| `binary` | `+`, `-`, `*`, `/`, `max`, `Real.rpow` | `WithBot.realAdd`, `realSub`, `realMul`, `realDiv`, `max`, `realPow` on two `some` | read |
| `unary` | `Real.exp`, `Real.log`, and so on, exactly the `some` branch of each `WithBot.real*`; `neg` and `abs` as `-x` and `\|x\|`; `minPosInf` as identity | `Tile.uop WithBot.real*`, `Option.map` for `negFloat` and `absFloat`, `Tile.uop id` | checked for `exp`, read for the rest |
| `fma`, `fmaTyped` | `a * b + c` | The projection rewrites both nodes to `Op.add (Op.mul a b) c` with the same broadcast indices | read |
| `atomicAdd` | `+` | `NumericDType.add` on the value read back | read |
| `libdevicePowi` | `x ^ n` with an integer exponent | `WithBot.realPowi` | read |
| `compareLt`, `compareLe`, `compareEq` | `decide` of the real order and equality | `ComparableDType.lt`, `le`, `eq`, `gt`, `ge`, `ne` on `WithBot ℝ` | read |
| `cast`, `fp32Load` | Identity | `FloatDType.cast` is the identity on `WithBot ℝ`; a compute load projects to a plain load | read |
| `fromNat`, `fromInt` | The coercion to `ℝ` | `Tile.natToReal`, `Tile.intToReal` | read |
| `fp32Bits` | `(Float32Bits.decodeReal b).unbotD 0` | A projected `fp32` constant is `Op.const` of `Float32Bits.decodeRat`; random values are the rational value of the same bits | checked for `uniform32`, inferred for the rest |
| `float8E5Bits` | `decodeRat` of the byte, `0` when it has none | `Float8E5Bits.decodeTile`; see D5 | read |
| `minimum`, `divRN` | `min`, `/` | `min` on `WithBot ℝ`, `NumericDType.real.div` | read |
| `reduceSum`, `reduceSumTyped`, `reduceMin`, `reduceMax`, `reduceMaxTyped`, `scan`, `dot`, `dotWithOptions`, `dotTyped`, `dotAcc`, `integerDot` | The upstream `Tile.reduceSum`, `Tile.reduceMin`, `Tile.reduceMax`, `Tile.scan`, `Tile.dot`, `Tile.dotInt` applied to the tile lifted by upstream `Tile.ofReal`, and read back with `reals` (`⊥` as `0`) | The same `Tile.*` function on the erased tile. The typed variants are these functions between identity casts | checked for `reduceSum`, inferred for the rest |
| `indexedMax` | `Tile.reduceMaxDrop` for the value, `Tile.argMaxDrop` for the index | `Tile.reduceMaxTyped … false` and `Tile.argMaxDrop` / `Tile.argMaxTyped` | inferred |
| `isFinite` | Always `true` | None: the projection rejects `ComputeExpr.isFinite` | read |
| `floatToInt`, `floatToNat` | Left at the default `none` | The real evaluator returns `none` on `Op.floatToInt` and `Op.floatToNat`, so both sides fail | read |

`reduceMax` is total in `Structural.Algebra` but partial upstream; it returns
the zero tile on an empty axis. Structural execution never calls it there
(read).

The fields above are the whole structure: 32 are set and 2 keep their default.

### `erase`

| Component | Definition | Mark |
| --- | --- | --- |
| Memory cell | `liftCell`: `Cell.mk d v` becomes `MemCell.of d (liftValue d v)`, under the same dtype tag | checked on a masked store |
| Register | `Option.map` of `liftValue` at the same dtype and shape | checked on an assignment |
| `liftValue` | A real becomes `some`; `Nat`, `Int`, `Bool`, pointer and block-pointer values are unchanged | read |
| Program ids, grid sizes | Copied | read |
| Undefined-value table | `s.undef .real` | read |

The interface describes `erase` as forgetting dtype tags. Cells and registers
keep their tag here, because `toAlgorithm?` keeps `.fp32` and the other
floating dtypes in the projected kernel (read: the docstring of
`ComputeKernel.evalR` in `Float/Refine.lean` says so, and
`ComputeOp.toAlgorithm?` returns the wrapped `Op` unchanged). A projected
`Op.load .fp32` reads `0` from a cell tagged `.real`, so an erasure that
retagged cells would break every typed load. The one place a dtype index is
dropped is the undefined-value table: the structural state has one table per
dtype and `BlockState` has a single real table. See D8.

A read at the wrong dtype agrees on both sides: `Cell.read` returns
`defaultValue`, which is `0` for `ℝ` (`default = 0`, checked), and
`readMemValue` returns `some 0` or the same discrete default (checked for an
`.fp32` cell read as `.real`, read for the others).

### What was reused

`Tile.ofReal` (`Semantics/TileOps.lean`) lifts a real tile; `MemCell.of`
builds the cell; the tile operations listed above carry the reductions, scans
and matrix products; `Float32Bits.decodeReal` and `Float8E5Bits.decodeRat`
carry the bit decoders. `Float/Erasure.lean` and `Float/StateErasure.lean`
were not reused; section 2 says why.

## 2. Inventory of upstream lemmas

| Lemma | File | What it proves | What remains for the bridge |
| --- | --- | --- | --- |
| `ComputeKernel.computeCorrectR_of_toAlgKernel` | `Float/Refine.lean` | `(h : ck.toAlgorithm? = Except.ok ck.toAlgKernel) (hc : Kernel.Correct R ck.toAlgKernel post) : ComputeCorrectR R ck post`. It unfolds a definition: correctness of the projected kernel under `execR R` is correctness of the compute kernel under `R` | Everything about structural execution. `execR` runs on `BlockState` with a rounding model and never mentions `FP.Structural`. The lemma is a pattern for the corollary of I4.3, which transfers `Kernel.Correct_without_Rounding` |
| `execR_triv`, `stepStmtR_triv`, `stepStmtsR_triv`, `evalOpR_triv` | `Float/StepR.lean`, `Float/EvalOpR.lean` | `execR RoundingModel.triv k s = exec k s`, and the same for one statement, a statement list and one expression | They relate two executions on `BlockState`. They are the nearest existing proof of the same shape, an induction over every constructor of `Op` and `Stmt` including the offset lists of dynamic block pointers; `evalOpR_triv_offsets` is private |
| `eraseDType`, `Op.eraseDType`, `MemAccess.eraseDType`, `MaskOpt.eraseDType`, `Stmt.eraseDType`, `Kernel.eraseDType`, `ComputeKernel.eraseDType`, and the witness maps `NumericDType.eraseDType`, `IntegralDType.eraseDType`, `ComparableDType.eraseDType` | `Float/Erasure.lean` | Definitions. They rewrite a kernel so that every floating dtype becomes `.real`: `constFloat` to `const`, `castFloat` to its operand, typed reductions and dots to the real ones | Not on the bridge's path: `bridge_exec` runs `ak` as projected, with its dtype tags. They would matter only for a bridge stated against `ak.eraseDType` |
| `eraseDType_float`, `MemAccess.eraseDType_region`, `_ptr`, `_blockPtr`, `MaskOpt.eraseDType_none`, `_mask`, `_maskOther`, `Op.eraseDType_ref`, `_castFloat`, `_load`, `_reduceMax`, `_reduceSum`, `Stmt.eraseDType_store`, `_atomicAdd`, `_atomicRMW`, `ComputeKernel.eraseDType_of_projection_error` | `Float/Erasure.lean` | Unfolding equations of the definitions above, and that a kernel whose projection fails is erased to itself. None mentions `evalOp` or `exec` | Same as above. No lemma in the repository states that erasure preserves execution (searched for theorems named `eraseDType`: read) |
| `eraseDTypeCarrier`, `Tile.eraseDType`, `BlockState.eraseDTypeRegs`, `BlockState.eraseDType` | `Float/StateErasure.lean` | Definitions on `BlockState`: retag every floating cell and register as `.real` | Wrong domain and wrong tags for the bridge: the map needed goes from `Structural.State ℝ` and must keep tags. Two details (read): `BlockState.eraseDType` does not copy `numPids`, which falls back to the default `fun _ => 1`; its register map looks up `.real`, `.fp32`, `.fp16`, `.bf16` and drops `.f8e4` and `.f8e5` registers |
| `BlockState.eraseDType_pids`, `_pid`, `_undef` | `Float/StateErasure.lean` | The erased state has the same program ids and undefined-value table | Counterparts for `erase` are in `RealAlgebra.lean` (`erase_pids`, `erase_numPids`, `erase_undef`) |
| `Structural.stepAlg` | `Float/Structural.lean` | A definition, not a lemma: structural execution of an algorithm statement. It runs `assign`, `store` and `atomicAdd` with no precision context and returns `none` on every other `Stmt` | Agreement with `stepStmt` on the three statements it runs; a side condition for the others (D11) |
| `Structural.run_append`, `exec_seq_cons`, `exec_seq_nil` | `Float/Structural.lean` | `run M (a ++ b) s = (run M a s).bind (run M b)`, and the same for `ComputeKernel.seq` | Used as is for statement lists. The proof-side counterpart is `stepStmts.append_some_iff` (`Semantics/Step.lean`) |
| `Structural.State.setReg_same`, `setReg_ne_name`, `write_same`, `write_other`, `masked_scatter_eq_filter`, `scatter_frame` | `Float/Structural.lean` | Register and cell updates of a structural state read back as written | `erase` commutes with `setReg` and with a typed `write` (checked in the scratch file; the proofs are short and belong to I4.2) |
| `BlockState.ext`, `setReg_*`, `writeMemTyped_*` | `Semantics/State.lean` | Extensionality and the frame lemmas of the proof-side state | Used as is |
| `DecoderCompatibility.equals_software_decoder` | `Float/DecoderCompatibility.lean` | `Float32Bits.decodeRat x = (FP.Value.mk x.bits : FP.Value .fp32).toRat?` | With `Random.uniform_toRat_uniformRat` it gives the `uniform32` case (checked). `rand` and `randTensor` need `Random.rand_toRat_randRat` in the same way (inferred) |
| `WithBot.sum_someTerm_eq_some`, `WithBot.unbotD_sum_some`, `WithBot.sup'_someTerm_eq_some`, `WithBot.unbotD_sup'_some`, `WithBot.real*_some` | `Semantics/TileOps.lean` | A sum or a maximum of `some` values is `some`, and each unary operation maps `some` to `some` | These are the facts that make `reals` undo `Tile.ofReal` after a tile operation. Minimum, scan and matrix product need the same fact and have no such lemma (read) |
| `Float8E5Bits.decodeTile_of_finite` | `Semantics/Float8.lean` | The byte decoder succeeds with the rational values when every byte is a valid `int8` with a finite encoding | Gives agreement under the side condition of D5 |
| `BlockState.paddingValue_of_undef_zero` | `Semantics/State.lean` | With an all-zero undefined-value table every padding value is the dtype default | Gives agreement under the strongest form of the side condition of D8 |
| `Structural.Safety.stepSafe`, `safeAlg` | `Float/StructuralSafety.lean` | Definitions that return `False` on every statement structural execution rejects | A ready description of the accepted statement fragment, mixed with memory bounds the bridge does not need |

No existing lemma relates `FP.Structural.exec` to `exec`, and no
`Structural.Algebra ℝ` existed (read: a search finds concrete algebras only
over bit-level values, words and syntactic terms).

## 3. Differences

`h` below is the hypothesis that the projection succeeds. A row is a
difference only when `h` can hold.

| Id | Construct | Structural side | Proof side | Side condition | Mark |
| --- | --- | --- | --- | --- | --- |
| D1 | `Op.negInf` | `realAlgebra.negInf = 0` | `⊥` | The kernel has no `Op.negInf`, or no `⊥` reaches a register that survives or an operation other than a store | checked: an assignment of `negInf` differs for every state; a store of `negInf` agrees |
| D2 | `⊥` in an operation | Cannot occur | `0 * ⊥ = ⊥`, `exp ⊥ = 0`, `max ⊥ x = x`, `log ⊥ = ⊥`, a store writes `0` for `⊥` | Covered by D1: `Op.negInf` is the only constructor that produces `⊥` from finite operands | read for each rule; inferred that no other source exists |
| D3 | `Op.castRealToInt8` | `none` | Succeeds, truncating toward zero | The kernel does not use it | read |
| D4 | `Op.argMin`, `Op.sort` | `none` | Succeed | The kernel does not use them | read |
| D5 | `Op.decodeFloat8E5` on a byte whose encoding is an infinity or a NaN | Succeeds; `float8E5Bits` returns `0` for that byte | `none` for the whole tile | Every decoded byte has a finite encoding | read |
| D6 | `Op.argMax`, `Op.argMaxTyped` over an empty axis | `none` | Succeed with index `0` | The reduced axis is non-empty | read |
| D7 | `Op.reshape` between shapes of different sizes | `none` | Succeeds; a lane without a source reads the dtype default | The two shapes have the same number of elements | read |
| D8 | Masked-off lane of a load without `other`, and out-of-bounds lane of a block-pointer load with unspecified padding | `s.undef d` at the address, for the dtype `d` of the load | From the single real table `u`: a floating lane is `some u`; a masked-off discrete lane is the dtype default; an unspecified-padding lane is `⌊u⌋` for `Int` and `Nat`, `u ≠ 0` for `Bool`, the default for pointers | `s.undef d = s.undef .real` for each floating `d`, and the discrete tables agree with the values at right. An all-default table satisfies every case | checked for `.fp32`: the sides differ exactly when the two tables differ at the address; read for the discrete cases |
| D9 | `ComputeOp.const` of dtype `fp64`, `ComputeOp.load` of dtype `fp64` | `none` | Succeed when the constant decodes; the load is a `.real` load | The kernel has no `fp64` compute constant or load | read |
| D10 | `ComputeOp.bitcast` other than `int32` or `uint32` to `fp32` | `none` | Succeeds on a scalar constant payload, for example `uint32` to `int32` | The kernel has no such bitcast | read |
| D11 | `ComputeStmt.alg st` with `st` a loop, a conditional or an atomic read-modify-write | `stepAlg` returns `none` | `stepStmt` runs `st` | `ComputeStmt.alg` wraps only `assign`, `store` and `atomicAdd` | checked for `ifThen` |
| D12 | `ComputeStmt.atomicRMW` | `none` | `stepAtomicRMW`; exchange and compare-and-swap can succeed | The kernel has no atomic read-modify-write | read |

D3, D4, D6, D7 and D9 to D12 have the same form: structural execution fails
and the proof side may succeed. One semantic conjunct, "structural execution
of `ck` from `s` succeeds", would cover all of them at once (inferred; the
choice between it and syntactic conjuncts belongs to I4.2). It does not cover
D1, D5 and D8, where structural execution succeeds.

### Missing backend support

Structural execution fails when the algebra lacks an optional operation and
never falls back to real evaluation (read: the header of `Structural.lean`
and the `←` on each optional field). `realAlgebra` supplies every optional
field except `floatToInt` and `floatToNat`, on which the proof side fails as
well. So this rule causes no difference for `realAlgebra`; the failures that
remain are the hard-coded `none` branches of D3, D4 and D9 to D12 (read).

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

- arithmetic, comparisons, casts, selections, pointer arithmetic, block
  pointers, `philox4x32` (both sides call `Random.Direct.evalField`),
  discrete reductions (`Tile.reduceMaxNat` repeats the body of
  `TileShape.discreteMaximum`);
- `Op.reshape` on equal sizes: `reshapeIndex` is the successful lookup of
  `Tile.reshape`;
- dynamic block-pointer lists: `evalOffsets` and `evalDeltas` against
  `List.mapM`;
- bitcast of an `int32` or `uint32` constant to `fp32`: needs
  `BitVec.ofInt 32 c.toInt = c.bits`;
- scans: the folded lane set always contains the output lane, so `max` and
  `min` never return `⊥`;
- `atomicAdd` and the three loop forms: the folds have the same shape as the
  store fold that was checked.

### Unchecked

- Which bench kernels satisfy the side conditions. A text search finds
  `Op.negInf` emitted by the DSL for `-inf` and `float("-inf")`
  (`DSL/Expansion/Main.lean`), and used in `bench/examples/OnlineSoftmax` and
  `VeriTile/Examples/FlashAttention1`. The DSL emits loops as
  `ComputeStmt.forLoop`, not under `ComputeStmt.alg`; kernels built with
  `ComputeKernel.fromKernelBody` put every statement under `alg`.
- The helper functions that both sides call by the same name
  (`BlockPtr.inBounds`, `BlockPtr.address`, `BlockPtr.advance`,
  `TileShape.*`, `Random.*`): only the call sites were compared.
- `Float/StepR.lean` and `Float/EvalOpR.lean` beyond the statements quoted in
  section 2.

## 4. Consequences for the frozen statement

- The statement can hold with the definitions given here for kernels without
  `Op.negInf` and with a coherent undefined-value table (checked on a constant
  assignment, a masked store and a sum of exponentials, for every state).
- It cannot cover a kernel that keeps `-inf` in a register, for any choice of
  `realAlgebra.negInf`: the erased state has `some` where the proof side has
  `⊥` (checked). Online softmax and flash attention seed a running maximum
  with `-inf`, and `max ⊥ x = x` has no real counterpart. Covering them needs
  a carrier with a bottom element, which changes the frozen types of
  `realAlgebra` and `erase` (inferred).
- The corollary of I4.3 speaks only about initial states in the range of
  `erase`: states of the proof semantics that hold no `⊥` (inferred).
