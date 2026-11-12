/-
VeriTile.Interp.EvalOp

Computable twins of the expression evaluators of `FP.Structural` (ticket
I2.3): the mutual block `evalOp` / `evalOffsets` / `evalDeltas`, then
`evalComputeOp` and `evalExpr`. Each twin has the cases of its original in the
same order, so that the adequacy proof (I3.2) can go case by case.

The text of a twin differs from the original in these places only:

| Original | Twin |
| --- | --- |
| `Algebra α` | `CAlgebra α`: a literal is a rational |
| `Op`, `ComputeOp`, `ComputeExpr` | `QOp`, `QComputeOp`, `QExpr` |
| `Structural.State α`, `s.regs`, `s.mem` | `State α`, `s.getReg`, `s.read` |
| `Structural.numeric`, `numericLt`, `numericLe`, `numericEq` | the twins in `Interp.Numeric` |
| `Structural.collectValues?` | `collectValues?` in `Interp.Enum` |
| `Structural.boundaryZero` | `boundaryZero` below |

Every other helper is the upstream one, which is computable already. Failure
is kept: a twin returns `none` where its original does, and an optional
algebra operation that is missing fails.

Every constructor of `QOp`, `QComputeOp` and `QExpr` has its case. As in the
originals, these cases are `none` on every input: `QOp.castRealToInt8`,
`argMin` and `sort`; `QComputeOp.const` and `load` at `fp64` and `bitcast`
other than `int32` or `uint32` to `fp32`; `QExpr.opaque` and `llrint`.
-/
import VeriTile.Interp.Enum
import VeriTile.Interp.Mirror
import VeriTile.Interp.Numeric
import VeriTile.Interp.State

namespace VeriTile.Interp

open VeriTile.Triton
open VeriTile.Triton.FP
open VeriTile.Triton.FP.Structural
  (Value Values ofFloat toFloat bop integralFloorDiv integralMod transposeValues)

/-- Twin of `Structural.boundaryZero`: explicit zero padding is the algebra's
literal of the rational zero on a floating dtype. -/
def boundaryZero {α : Type} [Inhabited α] (M : CAlgebra α)
    (p : Option ComputeDType) : (d : TileDType) → Value α d
  | .real => M.literal p .real 0
  | .fp32 => M.literal p .fp32 0
  | .fp16 => M.literal p .fp16 0
  | .bf16 => M.literal p .bf16 0
  | .f8e4 => M.literal p .f8e4 0
  | .f8e5 => M.literal p .f8e5 0
  | d => Structural.defaultValue d

set_option maxHeartbeats 1600000 in
mutual
/-- Twin of `Structural.evalOp`. -/
def evalOp {α : Type} [Inhabited α] (M : CAlgebra α) (p : Option ComputeDType) :
    QOp dtype shape → State α → Option (Values α dtype shape)
  | .const c, _ => some (fun _ => M.literal p .real c)
  | .constFloat d c, _ => some (fun _ => ofFloat d (M.literal p d c))
  | .constNat n, _ => some (fun _ => n)
  | .constInt n, _ => some (fun _ => n)
  | .constBool b, _ => some (fun _ => b)
  | .negInf, _ => some (fun _ => M.negInf)
  | .programId a, s => some (fun _ => s.pids a)
  | .numPrograms a, s => some (fun _ => s.numPids a)
  | .ref d sh n, s => s.getReg d sh n
  | .arange _, _ => some (fun i => i.1.val)
  | .philox4x32 cfg plan field seed c0 c1 c2 c3, s => do
      let vs ← evalOp M p seed s
      let v0 ← evalOp M p c0 s
      let v1 ← evalOp M p c1 s
      let v2 ← evalOp M p c2 s
      let v3 ← evalOp M p c3 s
      let result ← Random.Direct.evalField cfg plan field (vs PUnit.unit)
        ⟨v0, v1, v2, v3⟩
      return result
  | .uniform32 word, s => do
      let v ← evalOp M p word s
      if Random.Direct.uniformInputsValid v then
        some (fun i => M.fp32Bits ⟨(Random.uniform (BitVec.ofNat 32 (v i))).bits⟩)
      else none
  | .rand cfg seed offsets, s => do
      let vs ← evalOp M p seed s
      let vo ← evalOp M p offsets s
      let seedValue := vs PUnit.unit
      if cfg.inputsValid seedValue vo then
        some (fun i => M.fp32Bits
          ⟨(Random.rand seedValue (vo i) cfg.offsetType.offsetWidth cfg.nRounds).bits⟩)
      else none
  | .randTensor cfg seed offsets, s => do
      let vs ← evalOp M p seed s
      let vo ← evalOp M p offsets s
      if cfg.tensorInputsValid vs vo then
        some (fun i => M.fp32Bits
          ⟨(Random.rand (vs i) (vo i) cfg.offsetType.offsetWidth cfg.nRounds).bits⟩)
      else none
  | .broadcast e _, s | .full _ e, s => do
      let v ← evalOp M p e s
      return fun _ => v PUnit.unit
  | .withPrecision _ precision e, s =>
      evalOp M (some (match precision with | .fp32 => .fp32 | .fp64 => .fp64)) e s
  | .castFloat src dst e, s => do
      let v ← evalOp M p e s
      return fun i => ofFloat dst (M.cast p src dst (toFloat src (v i)))
  | .decodeFloat8E5 dst e, s => do
      let decode ← M.float8E5Bits
      let v ← evalOp M p e s
      if (TileShape.allIndices shape).all (fun i => Float8E5Bits.isInt8 (v i)) then
        some (fun i => ofFloat dst (decode dst (Float8E5Bits.ofInt (v i))))
      else none
  | .negFloat dtype e, s => do
      let v ← evalOp M p e s
      return fun i => ofFloat dtype (M.unary p (.neg dtype) (toFloat dtype (v i)))
  | .absFloat dtype e, s => do
      let v ← evalOp M p e s
      return fun i => ofFloat dtype (M.unary p (.abs dtype) (toFloat dtype (v i)))
  | .floatToInt precision src width e, s => do
      let convert ← M.floatToInt precision src width
      let sourceContext := match precision with
        | none => p
        | some .fp32 => some ComputeDType.fp32
        | some .fp64 => some ComputeDType.fp64
      let v ← evalOp M sourceContext e s
      collectValues? (fun i => convert (toFloat src (v i)))
  | .floatToNat precision src width e, s => do
      let convert ← M.floatToNat precision src width
      let sourceContext := match precision with
        | none => p
        | some .fp32 => some ComputeDType.fp32
        | some .fp64 => some ComputeDType.fp64
      let v ← evalOp M sourceContext e s
      collectValues? (fun i => convert (toFloat src (v i)))
  | .castNatToInt e, s => do
      let v ← evalOp M p e s
      return fun i => Int.ofNat (v i)
  | .castIntToNat e, s => do
      let v ← evalOp M p e s
      return fun i => (v i).toNat
  | .add d bc a b, s => return bop (numeric M p .add d) bc (← evalOp M p a s) (← evalOp M p b s)
  | .sub d bc a b, s => return bop (numeric M p .sub d) bc (← evalOp M p a s) (← evalOp M p b s)
  | .mul d bc a b, s => return bop (numeric M p .mul d) bc (← evalOp M p a s) (← evalOp M p b s)
  | .div d bc a b, s => return bop (numeric M p .div d) bc (← evalOp M p a s) (← evalOp M p b s)
  | .exp a, s => return (M.unary p .exp) ∘ (← evalOp M p a s)
  | .libdeviceExp a, s => return (M.unary p .libdeviceExp) ∘ (← evalOp M p a s)
  | .libdeviceLog a, s => return (M.unary p .libdeviceLog) ∘ (← evalOp M p a s)
  | .libdeviceExpm1 a, s => return (M.unary p .libdeviceExpm1) ∘ (← evalOp M p a s)
  | .libdeviceTanh a, s => return (M.unary p .libdeviceTanh) ∘ (← evalOp M p a s)
  | .libdeviceLog1p a, s => return (M.unary p .libdeviceLog1p) ∘ (← evalOp M p a s)
  | .exp2 a, s => return (M.unary p .exp2) ∘ (← evalOp M p a s)
  | .log a, s => return (M.unary p .log) ∘ (← evalOp M p a s)
  | .log2 a, s => return (M.unary p .log2) ∘ (← evalOp M p a s)
  | .sigmoid a, s => return (M.unary p .sigmoid) ∘ (← evalOp M p a s)
  | .sqrt a, s => return (M.unary p .sqrt) ∘ (← evalOp M p a s)
  | .rsqrt a, s => return (M.unary p .rsqrt) ∘ (← evalOp M p a s)
  | .libdeviceRsqrt a, s => return (M.unary p .libdeviceRsqrt) ∘ (← evalOp M p a s)
  | .tanh a, s => return (M.unary p .tanh) ∘ (← evalOp M p a s)
  | .sin a, s => return (M.unary p .sin) ∘ (← evalOp M p a s)
  | .cos a, s => return (M.unary p .cos) ∘ (← evalOp M p a s)
  | .tan a, s => return (M.unary p .tan) ∘ (← evalOp M p a s)
  | .atan a, s => return (M.unary p .atan) ∘ (← evalOp M p a s)
  | .cosh a, s => return (M.unary p .cosh) ∘ (← evalOp M p a s)
  | .sinh a, s => return (M.unary p .sinh) ∘ (← evalOp M p a s)
  | .erf a, s => return (M.unary p .erf) ∘ (← evalOp M p a s)
  | .minPosInf a, s => return (M.unary p .minPosInf) ∘ (← evalOp M p a s)
  | .max2 bc a b, s => return bop (M.binary p .real .max) bc (← evalOp M p a s) (← evalOp M p b s)
  | .maximumFloat dtype bc a b, s =>
      return bop (fun x y => ofFloat dtype (M.binary p dtype .max
        (toFloat dtype x) (toFloat dtype y))) bc (← evalOp M p a s) (← evalOp M p b s)
  | .minimumFloat dtype bc a b, s => do
      let minimum ← M.minimum p dtype
      let va ← evalOp M p a s
      let vb ← evalOp M p b s
      return bop (fun x y => ofFloat dtype
        (minimum (toFloat dtype x) (toFloat dtype y))) bc va vb
  | .pow bc a b, s => return bop (M.binary p .real .pow) bc (← evalOp M p a s) (← evalOp M p b s)
  | .libdevicePowi bc a b, s => do
      let va ← evalOp M p a s
      let vb ← evalOp M p b s
      let powi ← M.libdevicePowi p
      if Int32Bits.valuesValid vb then some (bop powi bc va vb) else none
  | .natToReal e, s => do
      let v ← evalOp M p e s
      return fun i => M.fromNat p (v i)
  | .intToReal e, s => do
      let v ← evalOp M p e s
      return fun i => M.fromInt p (v i)
  | .boolAnd bc a b, s => return bop (· && ·) bc (← evalOp M p a s) (← evalOp M p b s)
  | .boolOr bc a b, s => return bop (· || ·) bc (← evalOp M p a s) (← evalOp M p b s)
  | .boolNot a, s => do
      let v ← evalOp M p a s
      return fun i => !(v i)
  | .lt h bc a b, s => do
      let f ← numericLt M p h
      return bop f bc (← evalOp M p a s) (← evalOp M p b s)
  | .gt h bc a b, s => do
      let f ← numericLt M p h
      return bop (fun a b => f b a) bc (← evalOp M p a s) (← evalOp M p b s)
  | .eq h bc a b, s => do
      let f ← numericEq M p h
      return bop f bc (← evalOp M p a s) (← evalOp M p b s)
  | .ne h bc a b, s => do
      let f ← numericEq M p h
      return bop (fun a b => !(f a b)) bc (← evalOp M p a s) (← evalOp M p b s)
  | .le h bc a b, s => do
      let f ← numericLe M p h
      return bop f bc (← evalOp M p a s) (← evalOp M p b s)
  | .ge h bc a b, s => do
      let f ← numericLe M p h
      return bop (fun a b => f b a) bc (← evalOp M p a s) (← evalOp M p b s)
  | .ptrBase r, _ => some (fun _ => (Region.cast r, 0))
  | .ptrAdd bc a b, s => return bop (fun (a : RegionName × Nat) (b : Nat) => (a.1, a.2 + b)) bc (← evalOp M p a s) (← evalOp M p b s)
  | .where c a b, s => do
      let vc ← evalOp M p c s
      let va ← evalOp M p a s
      let vb ← evalOp M p b s
      return fun i => if vc i then va i else vb i
  | .ite c a b, s => do
      let vc ← evalOp M p c s
      if vc PUnit.unit then evalOp M p a s else evalOp M p b s
  | .load d (.blockPtr ptr checkedAxes) mask padding, s => do
      let ptrs ← evalOp M p ptr s
      let read := fun i =>
        let bp := ptrs i
        let ix := TileShape.indexToList _ i
        if bp.inBounds ix checkedAxes then
          (s.read bp.region (bp.address ix)).read d
        else match padding with
          | .zero => boundaryZero M p d
          | .unspecified => s.undef d bp.region (bp.address ix)
      match mask with
      | .none => return read
      | .mask cond => do
          let active ← evalOp M p cond s
          return fun i =>
            let bp := ptrs i
            let ix := TileShape.indexToList _ i
            if active i then read i
            else if bp.inBounds ix checkedAxes then s.undef d bp.region (bp.address ix)
            else match padding with
              | .zero => boundaryZero M p d
              | .unspecified => s.undef d bp.region (bp.address ix)
      | .maskOther cond other => do
          let active ← evalOp M p cond s
          let fallback ← evalOp M p other s
          return fun i => if active i then read i else fallback i
  | .load d mem mask padding, s => do
      let addressResult : Option (TileIndex _ → RegionName × Nat) := match mem with
        | .region r off => (evalOp M p off s).map (fun offsets i => (Region.cast r, offsets i))
        | .ptr e => evalOp M p e s
        | .blockPtr .. => none
      let addresses ← addressResult
      let activeResult : Option (TileIndex _ → Bool) := match mask with
        | .none => some (fun _ => Bool.true)
        | .mask e | .maskOther e _ => evalOp M p e s
      let active ← activeResult
      let otherResult : Option (Values α d _) := match mask with
        | .maskOther _ e => evalOp M p e s
        | _ => some (fun i => s.undef d (addresses i).1 (addresses i).2)
      let other ← otherResult
      return fun i => if active i then (s.read (addresses i).1 (addresses i).2).read d else other i
  | .reduceMin axis keepDims e, s => do
      let v ← evalOp M p e s
      if 0 < TileShape.axisDim _ axis then
        M.reduceMin p axis keepDims v
      else none
  | .reduceMax axis keepDims e, s => do
      let v ← evalOp M p e s
      if 0 < TileShape.axisDim _ axis then
        return M.reduceMax p axis keepDims v
      else none
  | .reduceSum axis keepDims e, s => do
      let v ← evalOp M p e s
      return M.reduceSum p axis keepDims v
  | .reduceSumBool axis keepDims e, s => do
      let v ← evalOp M p e s
      return TileShape.discreteBoolSum axis keepDims v
  | .reduceSumTyped dtype axis keepDims e, s => do
      M.reduceSumTyped p dtype axis keepDims (← evalOp M p e s)
  | .reduceMaxTyped dtype axis keepDims e, s => do
      let v ← evalOp M p e s
      if 0 < TileShape.axisDim _ axis then
        M.reduceMaxTyped p dtype axis keepDims v
      else none
  | .castRealToInt8 .., _ => none
  | .divRN bc a b, s => do
      let va ← evalOp M p a s
      let vb ← evalOp M p b s
      let f ← M.divRN
      pure (bop f bc va vb)
  | .floorDiv h bc a b, s =>
      return bop (integralFloorDiv h) bc (← evalOp M p a s) (← evalOp M p b s)
  | .mod h bc a b, s =>
      return bop (integralMod h) bc (← evalOp M p a s) (← evalOp M p b s)
  | .bitAnd bc a b, s => do
      let va ← evalOp M p a s
      let vb ← evalOp M p b s
      return bop (· &&& ·) bc va vb
  | .bitOr bc a b, s => do
      let va ← evalOp M p a s
      let vb ← evalOp M p b s
      return bop (· ||| ·) bc va vb
  | .bitXor bc a b, s => do
      let va ← evalOp M p a s
      let vb ← evalOp M p b s
      return bop (· ^^^ ·) bc va vb
  | .shiftLeft bc a b, s => do
      let va ← evalOp M p a s
      let vb ← evalOp M p b s
      return bop (· <<< ·) bc va vb
  | .shiftRight bc a b, s => do
      let va ← evalOp M p a s
      let vb ← evalOp M p b s
      return bop (· >>> ·) bc va vb
  | .reduceMaxNat axis keepDims e, s => do
      let v ← evalOp M p e s
      TileShape.discreteMaximum axis keepDims v
  | .reduceMaxInt axis keepDims e, s => do
      let v ← evalOp M p e s
      TileShape.discreteMaximum axis keepDims v
  | .scan op axis direction e, s => do
      M.scan p op axis direction (← evalOp M p e s)
  | .argMax axis e, s => do
      let v ← evalOp M p e s
      if TileShape.axisDim _ axis = 0 then none else (M.indexedMax p .real axis v).map Prod.snd
  | .argMaxInt axis e, s => do
      let v ← evalOp M p e s
      return TileShape.discreteIntArgMax axis v
  | .argMaxTyped dtype axis e precision, s => do
      let actual := match precision with
        | some .fp32 => some ComputeDType.fp32
        | some .fp64 => some ComputeDType.fp64
        | none => p
      let v ← evalOp M actual e s
      if TileShape.axisDim _ axis = 0 then none else (M.indexedMax actual dtype axis v).map Prod.snd
  | .indexedMaxValue dtype axis e, s => do
      let v ← evalOp M p e s
      if TileShape.axisDim _ axis = 0 then none else (M.indexedMax p dtype axis v).map Prod.fst
  | .argMin .., _ => none
  | .sort .., _ => none
  | .dot (options := options) a b, s => do
      let va ← evalOp M p a s
      let vb ← evalOp M p b s
      if options = {} then M.dot p va vb else M.dotWithOptions p options va vb
  | .dotTyped left right out a b options, s => do
      M.dotTyped p left right out options (← evalOp M p a s) (← evalOp M p b s)
  | .dotAcc left right out a b acc options, s => do
      M.dotAcc p left right out options
        (← evalOp M p a s) (← evalOp M p b s) (← evalOp M p acc s)
  | .dotInt a b options, s => do
      M.integerDot options (← evalOp M p a s) (← evalOp M p b s)
  | .transpose (batch := batch) a, s => do
      return transposeValues batch (← evalOp M p a s)
  | .reshape (shape := input) output e, s => do
      let v ← evalOp M p e s
      if h : TileShape.product input = TileShape.product output then
        some (fun i => v (TileShape.reshapeIndex h i))
      else none
  | .remap _ map a, s => do
      let v ← evalOp M p a s
      return fun i => v (map i)
  | .join (shape := shape) a b, s => do
      let va ← evalOp M p a s
      let vb ← evalOp M p b s
      return fun i =>
        let parts := TileShape.splitLastIndex shape i
        if parts.2.val = 0 then va parts.1 else vb parts.1
  | .split (shape := shape) side a, s => do
      let v ← evalOp M p a s
      return fun i => v (TileShape.appendAxisIndex shape i side)
  | .expandDim (shape := shape) axis a, s => do
      let v ← evalOp M p a s
      return fun i => v (TileShape.dropInsertedIndex shape axis 1 i)
  | .ptrSub bc a b, s =>
      return bop (fun (a : RegionName × Nat) (b : Nat) => (a.1, a.2 - b)) bc
        (← evalOp M p a s) (← evalOp M p b s)
  | .makeBlockPtr region base parent block strides offsets, _ =>
      some (fun _ => ⟨Region.cast region, base, parent, block, strides, offsets⟩)
  | .makeBlockPtrDyn region base parent block strides offsets, s => do
      let base ← evalOp M p base s
      return fun _ => ⟨region, base PUnit.unit, parent, block, strides, offsets⟩
  | .makeBlockPtrDynOffsets region base parent block strides offsets, s => do
      let base ← evalOp M p base s
      let offsets ← evalOffsets M p offsets s
      return fun _ => ⟨region, base PUnit.unit, parent, block, strides, offsets⟩
  | .makeBlockPtrDynMetadata region base parent block strides offsets, s => do
      let base ← evalOp M p base s
      let strides ← evalOffsets M p strides s
      let offsets ← evalOffsets M p offsets s
      return fun _ => ⟨region,base PUnit.unit,parent,block,strides,offsets⟩
  | .advanceBlockPtr ptr deltas, s => do
      let ptrs ← evalOp M p ptr s
      return fun i => (ptrs i).advance deltas
  | .advanceBlockPtrDyn ptr deltas, s => do
      let ptrs ← evalOp M p ptr s
      let ds ← evalDeltas M p deltas s
      return fun i => (ptrs i).advance ds
termination_by structural e _ => e

/-- Twin of `Structural.evalOffsets`. -/
def evalOffsets {α : Type} [Inhabited α] (M : CAlgebra α)
    (p : Option ComputeDType) : List (QOp .nat []) → State α → Option (List Nat)
  | [], _ => some []
  | e :: rest, s => do
      let v ← evalOp M p e s
      let vs ← evalOffsets M p rest s
      return v PUnit.unit :: vs
termination_by structural es _ => es

/-- Twin of `Structural.evalDeltas`. -/
def evalDeltas {α : Type} [Inhabited α] (M : CAlgebra α)
    (p : Option ComputeDType) : List (QOp .int []) → State α → Option (List Int)
  | [], _ => some []
  | e :: rest, s => do
      let v ← evalOp M p e s
      let vs ← evalDeltas M p rest s
      return v PUnit.unit :: vs
termination_by structural es _ => es
end

-- As upstream `Structural.evalOp_unfold`: a module that imports this one would
-- generate the unfold theorem under the default heartbeat limit, which is too low.
set_option maxHeartbeats 1600000 in
abbrev evalOp_unfold := @evalOp.eq_def

/-- Twin of `Structural.evalComputeOp`. -/
def evalComputeOp {α : Type} [Inhabited α] (M : CAlgebra α) :
    QComputeOp dtype shape → State α → Option (Values α dtype.eraseDType shape)
  | .alg d e, s => evalOp M (some d) e s
  | .const (dtype := .fp64) _, _ => none
  | .const (dtype := .fp32) c, _ => some (fun _ => M.fp32Bits c)
  | .const (dtype := .int32) c, _ => some (fun _ => c.toInt)
  | .const (dtype := .uint32) c, _ => some (fun _ => c.toNat)
  | .full _ e, s => do
      let v ← evalComputeOp M e s
      return fun _ => v PUnit.unit
  | .load .fp64 .., _ => none
  | .load .fp32 mem mask padding, s => do
      let v ← evalOp M (some .fp32) (.load .real mem mask padding) s
      return fun i => M.fp32Load (v i)
  | .load .int32 mem mask padding, s => evalOp M (some .int32) (.load .int mem mask padding) s
  | .load .uint32 mem mask padding, s => evalOp M (some .uint32) (.load .nat mem mask padding) s
  | .bitcast .int32 .fp32 _ e, s => do
      let v ← evalComputeOp M e s
      return fun i => M.fp32Bits ⟨BitVec.ofInt 32 (v i)⟩
  | .bitcast .uint32 .fp32 _ e, s => do
      let v ← evalComputeOp M e s
      return fun i => M.fp32Bits ⟨BitVec.ofNat 32 (v i)⟩
  | .bitcast .., _ => none

/-- Twin of `Structural.evalExpr`. -/
def evalExpr {α : Type} [Inhabited α] (M : CAlgebra α) :
    QExpr dtype shape → State α → Option (Values α dtype shape)
  | .alg e, s => evalOp M none e s
  | .compute e, s => evalComputeOp M e s
  | .isFinite precision e, s => do
      let values ← evalExpr M e s
      let classify ← M.isFinite precision
      return fun i => classify (values i)
  | .fma precision bcAB bcC a b c, s => do
      let fused ← M.fma precision
      let av ← evalExpr M a s
      let bv ← evalExpr M b s
      let cv ← evalExpr M c s
      return fun i => fused
        (av (bcAB.leftIndex (bcC.leftIndex i)))
        (bv (bcAB.rightIndex (bcC.leftIndex i)))
        (cv (bcC.rightIndex i))
  | .fmaTyped dtype bcAB bcC a b c, s => do
      let fused ← M.fmaTyped dtype
      let av ← evalExpr M a s
      let bv ← evalExpr M b s
      let cv ← evalExpr M c s
      return fun i => ofFloat dtype (fused
        (toFloat dtype (av (bcAB.leftIndex (bcC.leftIndex i))))
        (toFloat dtype (bv (bcAB.rightIndex (bcC.leftIndex i))))
        (toFloat dtype (cv (bcC.rightIndex i))))
  | .opaque _, _ => none
  | .llrint _, _ => none
termination_by structural e _ => e

/-! Unit checks on the carrier `ℚ`. Each `#guard` states the expected value, so
a wrong result fails the build. -/

/-- Exact rational arithmetic. A literal under a compute precision is off by
1000, so a check sees which precision reached it. `reduceSum` adds along the
axis; `dot` multiplies two matrices without a batch. Of the optional
operations, only `dot` and `fma` are present. -/
private def Q : CAlgebra ℚ where
  literal := fun p _ q => if p.isSome then q + 1000 else q
  negInf := -1
  binary := fun _ _ op a b => match op with
    | .add => a + b | .sub => a - b | .mul => a * b | .div => a / b | .max => max a b | .pow => 0
  fma := fun _ => some fun a b c => a * b + c
  unary := fun _ _ a => a
  cast := fun _ _ _ a => a
  fromNat := fun _ n => n
  fromInt := fun _ i => i
  fp32Bits := fun b => b.bits.toNat
  fp32Load := (· + 1/2)
  reduceMax := fun _ {_} _ _ _ _ => 0
  reduceSum := fun _ {shape} axis keepDims v => match keepDims with
    | false => fun out =>
        ((List.finRange _).map fun k => v (TileShape.insertAxisIndex shape axis out k)).sum
    | true => fun out =>
        ((List.finRange _).map fun k => v (TileShape.replaceAxisIndex shape axis out k)).sum
  dot := fun _ {batch} {_} {k} {_} a b => match batch, a, b with
    | [], a, b => some fun (i, j, _) =>
        ((List.finRange k).map fun l => a (i, l, ()) * b (l, j, ())).sum
    | _ :: _, _, _ => none

/-- Cell `o` of every region holds the real `10 + o`, except `x[1]`, which was
written; `m` is the matrix `[[0, 1], [2, 3]]`. -/
private def st : State ℚ :=
  let s : State ℚ :=
    { init := fun _ o => .mk .real (10 + o : ℚ), pids := fun a => 7 + a, numPids := fun _ => 9,
      undef := fun d _ _ => Structural.defaultValue d }
  (s.write "x" 1 (.mk .real (1/4))).setReg "m" .real [2, 2] fun (i, j, _) => (i.val * 2 + j.val : ℚ)

/-- The lanes of a result in index order. A check names `d` where the dtype of
the expression is not a constructor, so that equality on the lanes is found. -/
private def lanes {d : TileDType} {sh : TileShape} (v : Option (Values ℚ d sh)) :
    Option (List (Value ℚ d)) :=
  v.map (TileShape.allIndices sh).map

private abbrev m : QOp .real ([] ++ [2, 2]) := .ref .real [2, 2] "m"
private abbrev x3 : QMemAccess .real [3] := .region "x" (.arange 3)
private abbrev small : QOp .bool [3] := .lt .nat .scalarR (.arange 3) (.constNat 2)
private abbrev real4 : QOp .real [4] := .natToReal (.arange 4)

-- Elementwise: a floating addition goes through the algebra, a discrete one is exact.
#guard lanes (evalOp Q none (.add .real .nil (.const (1/2)) (.const (1/3))) st) = some [(5/6 : ℚ)]
#guard lanes (evalOp Q none (.add .nat (.consSame .nil) (.arange 3) (.arange 3)) st)
  = some [(0 : Nat), 2, 4]
-- Broadcast: a scalar fills a shape; a binary operation broadcasts its scalar side.
#guard lanes (evalOp Q none (.broadcast (.const 2) [3]) st) = some [(2 : ℚ), 2, 2]
#guard lanes (evalOp Q none (.mul .nat .scalarL (.constNat 3) (.arange 3)) st)
  = some [(0 : Nat), 3, 6]
-- Reduction: the algebra's sum, the exact discrete reductions, and a missing `reduceMin`.
#guard lanes (evalOp Q none (.reduceSum (shape := [4]) ⟨0, by decide⟩ false real4) st)
  = some [(6 : ℚ)]
#guard lanes (evalOp Q none (.reduceMaxNat (shape := [4]) ⟨0, by decide⟩ false (.arange 4)) st)
  = some [(3 : Nat)]
#guard lanes (evalOp Q none (.reduceSumBool (shape := [3]) ⟨0, by decide⟩ false small) st)
  = some [(2 : Nat)]
#guard (evalOp Q none (.reduceMin (shape := [4]) ⟨0, by decide⟩ false real4) st).isNone
-- Dot: `m * m`; explicit options do not fall back to the default `dot`.
#guard lanes (evalOp Q none (.dot (batch := []) m m) st) = some [(2 : ℚ), 3, 6, 11]
#guard (evalOp Q none (.dot (batch := []) m m { allowTF32 := some true }) st).isNone
-- Where: lane by lane.
#guard lanes (evalOp Q none (.where small (.arange 3) (.broadcast (.constNat 9) [3])) st)
  = some [(0 : Nat), 1, 9]
-- Load: by region and offset, masked with a fallback, and through a block
-- pointer whose out-of-bounds lane is zero padding.
#guard lanes (evalOp Q none (.load .real x3 .none) st) = some [(10 : ℚ), 1/4, 12]
#guard lanes (evalOp Q none (.load .real x3 (.maskOther small (.broadcast (.const 5) [3]))) st)
  = some [(10 : ℚ), 1/4, 5]
#guard lanes (evalOp Q none
    (.load .real (.blockPtr (.makeBlockPtr (d := .real) "x" 0 [2] [3] [1] [0]) [0]) .none .zero) st)
  = some [(10 : ℚ), 1/4, 0]
-- Failure: an unbound register, a missing optional operation, an unsupported constructor.
#guard (evalOp Q none (.ref .real [] "unbound") st).isNone
#guard (evalOp Q none (.add .real .nil (.const 1) (.ref .real [] "unbound")) st).isNone
#guard (evalOp Q none (.minimumFloat .real .nil (.const 1) (.const 2)) st).isNone
#guard (evalOp Q none (.floatToNat none .real 32 (.const 1)) st).isNone
#guard (evalOp Q none (.argMin (shape := [4]) ⟨0, by decide⟩ real4) st).isNone
-- `withPrecision` changes the precision its operand is evaluated under.
#guard lanes (d := .real) (evalOp Q none (.withPrecision .real .fp32 (.const 1)) st)
  = some [(1001 : ℚ)]
-- `evalOffsets` and `evalDeltas`: every element, or failure.
#guard evalOffsets Q none [.constNat 1, .programId 0, .numPrograms 0] st = some [1, 7, 9]
#guard evalOffsets Q none [.constNat 1, .ref .nat [] "unbound"] st = none
#guard evalDeltas Q none [.constInt (-1), .constInt 2] st = some [-1, 2]
-- A block pointer at offset 1, advanced by 1, reads `x[2]` and `x[3]`.
#guard lanes (evalOp Q none (.load .real (.blockPtr (.advanceBlockPtrDyn
      (.makeBlockPtrDynOffsets "x" (.constNat 0) [4] [2] [1] [.constNat 1]) [.constInt 1]) [])
    .none) st) = some [(12 : ℚ), 13]
-- `evalComputeOp`: the precision of the operation reaches the literal; a bit
-- pattern is exact; an fp32 load applies `fp32Load`; fp64 fails.
#guard lanes (d := .real) (evalComputeOp Q (.alg .fp32 (.const (1/2))) st) = some [(2001/2 : ℚ)]
#guard lanes (d := .nat) (evalComputeOp Q (.full [2] (.const (dtype := .uint32) ⟨5⟩)) st)
  = some [(5 : Nat), 5]
#guard lanes (d := .real)
    (evalComputeOp Q (.bitcast .uint32 .fp32 rfl (.const (dtype := .uint32) ⟨5⟩)) st)
  = some [(5 : ℚ)]
#guard lanes (d := .real) (evalComputeOp Q (.load .fp32 x3 .none) st) = some [(21/2 : ℚ), 3/4, 25/2]
#guard (evalComputeOp Q (.load .fp64 (shape := [3]) (.region "x" (.arange 3)) .none) st).isNone
-- `evalExpr`: both embeddings, a present and a missing optional operation, an opaque call.
#guard lanes (evalExpr Q (.alg (.const (1/2))) st) = some [(1/2 : ℚ)]
#guard lanes (d := .real) (evalExpr Q (.compute (.alg .fp32 (.const (1/2)))) st)
  = some [(2001/2 : ℚ)]
#guard lanes (evalExpr Q
    (.fma .fp32 .nil .nil (.alg (.const 2)) (.alg (.const 3)) (.alg (.const 4))) st)
  = some [(10 : ℚ)]
#guard (evalExpr Q (.fmaTyped .fp32 .nil .nil (.alg (.constFloat .fp32 2))
    (.alg (.constFloat .fp32 3)) (.alg (.constFloat .fp32 4))) st).isNone
#guard (evalExpr Q (.isFinite .fp32 (.alg (.const 1))) st).isNone
#guard (evalExpr (dtype := .real) (shape := []) Q (.opaque "f") st).isNone

end VeriTile.Interp
