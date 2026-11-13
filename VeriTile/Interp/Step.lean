/-
VeriTile.Interp.Step

Computable twins of Structural's stores and statement execution (I2.4).
Cases, evaluation order, failure and termination measures stay upstream's;
the write folds use State.scatter over the same lane enumeration. The mutual
step/run/loop/range/rangeInt block stays together to preserve its recursion.
-/
import VeriTile.Interp.EvalOp

namespace VeriTile.Interp

open VeriTile.Triton
open VeriTile.Triton.FP
open VeriTile.Triton.FP.Structural (Value Values)

/-- The address and mask are evaluated before writes, as in the operational
semantics. Each active lane copies its already typed value into memory. -/
def store {α : Type} [Inhabited α] (M : CAlgebra α)
    (mem : QMemAccess dtype shape) (mask : QMaskOpt dtype shape)
    (values : Values α dtype shape) (s : State α) : Option (State α) :=
  match mem with
  | .blockPtr ptr checkedAxes => do
      let ptrs ← evalOp M none ptr s
      let activeResult : Option (TileIndex shape → Bool) := match mask with
        | .none => some (fun _ => Bool.true)
        | .mask e | .maskOther e _ => evalOp M none e s
      let active ← activeResult
      return s.scatter (TileShape.allIndices shape)
        (fun i => active i && (ptrs i).inBounds (TileShape.indexToList shape i) checkedAxes)
        (fun i => ((ptrs i).region, (ptrs i).address (TileShape.indexToList shape i)))
        (fun i _ => .mk dtype (values i))
  | _ => do
  let addressResult : Option (TileIndex shape → RegionName × Nat) := match mem with
    | .region r off => (evalOp M none off s).map (fun offsets i => (Region.cast r, offsets i))
    | .ptr e => evalOp M none e s
    | .blockPtr .. => none
  let addresses ← addressResult
  let activeResult : Option (TileIndex shape → Bool) := match mask with
    | .none => some (fun _ => Bool.true)
    | .mask e | .maskOther e _ => evalOp M none e s
  let active ← activeResult
  return s.scatter (TileShape.allIndices shape) active addresses
    (fun i _ => .mk dtype (values i))

def atomicAddition {α : Type} (M : CAlgebra α) :
    {dtype : TileDType} → NumericDType dtype → Option (Value α dtype → Value α dtype → Value α dtype)
  | _, .real => M.atomicAdd .real
  | _, .fp32 => M.atomicAdd .fp32
  | _, .fp16 => M.atomicAdd .fp16
  | _, .bf16 => M.atomicAdd .bf16
  | _, .f8e4 => M.atomicAdd .f8e4
  | _, .f8e5 => M.atomicAdd .f8e5
  | _, .nat => some (· + ·)
  | _, .int => some (· + ·)

/-- One program's masked read-modify-write fold. Each lane reads the state
left by the preceding lane, including repeated addresses. Floating atomic
addition is its own opaque backend, selected by storage dtype. There is no
commutation, reassociation or connection to ordinary floating addition. -/
def atomicAdd {α : Type} [Inhabited α] (M : CAlgebra α)
    (nd : NumericDType dtype) (mem : QMemAccess dtype shape) (mask : QMaskOpt dtype shape)
    (values : Values α dtype shape) (s : State α) : Option (State α) := do
  let combine ← atomicAddition M nd
  match mem with
  | .blockPtr ptr checkedAxes => do
      let ptrs ← evalOp M none ptr s
      let activeResult : Option (TileIndex shape → Bool) := match mask with
        | .none => some (fun _ => Bool.true)
        | .mask e | .maskOther e _ => evalOp M none e s
      let active ← activeResult
      return s.scatter (TileShape.allIndices shape)
        (fun i => active i && (ptrs i).inBounds (TileShape.indexToList shape i) checkedAxes)
        (fun i => ((ptrs i).region, (ptrs i).address (TileShape.indexToList shape i)))
        (fun i c => .mk dtype (combine (c.read dtype) (values i)))
  | _ => do
      let addressResult : Option (TileIndex shape → RegionName × Nat) := match mem with
        | .region r off => (evalOp M none off s).map (fun offsets i => (Region.cast r, offsets i))
        | .ptr e => evalOp M none e s
        | .blockPtr .. => none
      let addresses ← addressResult
      let activeResult : Option (TileIndex shape → Bool) := match mask with
        | .none => some (fun _ => Bool.true)
        | .mask e | .maskOther e _ => evalOp M none e s
      let active ← activeResult
      return s.scatter (TileShape.allIndices shape) active addresses
        (fun i c => .mk dtype (combine (c.read dtype) (values i)))

def stepAlg {α : Type} [Inhabited α] (M : CAlgebra α) :
    QAlgStmt → State α → Option (State α)
  | .assign d sh n e, s => do
      let v ← evalOp M none e s
      return s.setReg n d sh v
  | .store _ _ mem e mask, s => do
      store M mem mask (← evalOp M none e s) s
  | .atomicAdd nd _ mem e mask, s => do
      atomicAdd M nd mem mask (← evalOp M none e s) s
  | _, _ => none

mutual

def step {α : Type} [Inhabited α] (M : CAlgebra α) :
    QStmt → State α → Option (State α)
  | .alg st, s => stepAlg M st s
  | .assign d sh n e, s => do
      let v ← evalExpr M e s
      return s.setReg n d sh v
  | .store _ _ mem e mask, s => do
      store M mem mask (← evalExpr M e s) s
  | .atomicAdd nd _ mem e mask, s => do
      atomicAdd M nd mem mask (← evalExpr M e s) s
  | .forLoop idx n body, s => loop M idx 0 n body s
  | .forRange idx start stop stride body, s => range M idx start stop stride body s
  | .forRangeDyn idx start stop stride body, s => do
      let a ← evalOp M none start s
      let b ← evalOp M none stop s
      let d ← evalOp M none stride s
      range M idx (a PUnit.unit) (b PUnit.unit) (d PUnit.unit) body s
  | .forRangeInt idx start stop stride body, s => do
      let a ← evalOp M none start s
      let b ← evalOp M none stop s
      let d ← evalOp M none stride s
      rangeInt M idx (a PUnit.unit) (b PUnit.unit) (d PUnit.unit) body s
  | .ifThen cond body, s => do
      let c ← evalOp M none cond s
      if c PUnit.unit then run M body s else some s
  | .ifThenElse cond yes no, s => do
      let c ← evalOp M none cond s
      if c PUnit.unit then run M yes s else run M no s
  -- One program executes synchronously in statement order. This precise
  -- intra-program barrier therefore preserves the whole abstract state,
  -- consistently with ComputeStmt.toAlgorithm?. It does not discharge
  -- native convergence or inter-program synchronization obligations.
  -- All async and unknown markers still use the failure fallback below.
  | .effectMarker "tl.debug_barrier", s => some s
  | _, _ => none
termination_by st _ => (sizeOf st, 0)
decreasing_by
  all_goals simp_wf
  all_goals (try omega)
  all_goals (have : 0 < sizeOf idx := by cases idx; simp)
  all_goals omega

def run {α : Type} [Inhabited α] (M : CAlgebra α) :
    List QStmt → State α → Option (State α)
  | [], s => some s
  | st :: rest, s => do run M rest (← step M st s)
termination_by code _ => (sizeOf code, 0)
decreasing_by all_goals (simp_wf; omega)

/-- Counted loops bind the exact natural index before every iteration. The
floating operations inside the body still use the same opaque algebra. -/
def loop {α : Type} [Inhabited α] (M : CAlgebra α)
    (idx : RegName) (start stop : Nat) (body : List QStmt) :
    State α → Option (State α)
  | s => if start < stop then do
      let t ← run M body (s.setReg idx .nat [] (fun _ => start))
      loop M idx (start + 1) stop body t
    else some s
termination_by _ => (sizeOf body + 1, stop - start)
decreasing_by all_goals omega

/-- Range bounds are evaluated once on entry. As in the existing operational
semantics, a zero stride or an empty interval executes no iterations. -/
def range {α : Type} [Inhabited α] (M : CAlgebra α)
    (idx : RegName) (cur stop stride : Nat) (body : List QStmt) :
    State α → Option (State α)
  | s => if stride = 0 then some s
    else if cur < stop then do
      let t ← run M body (s.setReg idx .nat [] (fun _ => cur))
      range M idx (cur + stride) stop stride body t
    else some s
termination_by _ => (sizeOf body + 1, stop - cur)
decreasing_by all_goals omega


/-- Signed range indices are never clipped to zero. Bounds and stride remain
fixed throughout execution; only the body changes the program state. -/
def rangeInt {α : Type} [Inhabited α] (M : CAlgebra α)
    (idx : RegName) (cur stop : Int) (stride : Nat) (body : List QStmt) :
    State α → Option (State α)
  | s => if stride = 0 then some s
    else if cur < stop then do
      let t ← run M body (s.setReg idx .int [] (fun _ => cur))
      rangeInt M idx (cur + stride) stop stride body t
    else some s
termination_by _ => (sizeOf body + 1, (stop - cur).toNat)
decreasing_by all_goals omega

end

/-! Unit checks use exact discrete values; floating values are irrelevant. -/

private def testAlgebra : CAlgebra Unit where
  literal := fun _ _ _ => ()
  negInf := ()
  binary := fun _ _ _ _ _ => ()
  unary := fun _ _ _ => ()
  cast := fun _ _ _ _ => ()
  fromNat := fun _ _ => ()
  fromInt := fun _ _ => ()
  fp32Bits := fun _ => ()
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => ()
  reduceSum := fun _ {_} _ _ _ _ => ()

private def testState : State Unit where
  init := fun _ _ => .mk .nat 10
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => Structural.defaultValue d

private def cells (s : Option (State Unit)) : Option (List Nat) :=
  s.map fun t => (List.range 4).map fun i => (t.read "x" i).read .nat

private def reg (s : Option (State Unit)) (name : RegName := "i") : Option Nat :=
  s.bind fun t => (t.getReg .nat [] name).map (· ())

private abbrev x3 : QMemAccess .nat [3] := .region "x" (.arange 3)
private abbrev mask3 : QOp .bool [3] := .lt .nat .scalarR (.arange 3) (.constNat 2)

-- Store every address form; inactive lanes and other regions keep their values.
#guard cells (store (dtype := .nat) testAlgebra x3 .none (fun i => i.1.val) testState) = some [0, 1, 2, 10]
#guard cells (store (dtype := .nat) testAlgebra x3 (.mask mask3) (fun i => i.1.val) testState) = some [0, 1, 10, 10]
#guard cells (store (dtype := .nat) testAlgebra x3 (.maskOther mask3 (.ref .nat [3] "unbound"))
    (fun i => i.1.val) testState) = some [0, 1, 10, 10]
#guard cells (store (dtype := .nat) testAlgebra (.ptr (.ptrAdd .scalarL (.ptrBase (d := .nat) "x") (.arange 3)))
    .none (fun i => i.1.val) testState) = some [0, 1, 2, 10]
#guard (store (dtype := .nat) testAlgebra x3 .none (fun i => i.1.val) testState).map
    (fun t => (t.read "other" 0).read .nat) = some 10
-- Bounds checks suppress the out-of-bounds lane, as in Structural.store.
#guard cells (store (dtype := .nat) testAlgebra
    (.blockPtr (.makeBlockPtr (d := .nat) "x" 0 [2] [3] [1] [0]) [0])
    .none (fun i => i.1.val) testState) = some [0, 1, 10, 10]
-- Address or mask evaluation failure fails the whole store.
#guard (store (dtype := .nat) testAlgebra (.region "x" (.ref .nat [] "unbound")) .none
    (fun _ => 1) testState).isNone
#guard (store (dtype := .nat) testAlgebra x3 (.mask (.ref .bool [3] "unbound"))
    (fun _ => 1) testState).isNone

-- Atomic addition reads preceding writes, including aliased lanes.
#guard (atomicAddition testAlgebra .nat).map (fun f => f 2 3) = some 5
#guard (atomicAddition testAlgebra .int).map (fun f => f (-2) 3) = some 1
#guard (atomicAddition testAlgebra .fp32).isNone
#guard cells (atomicAdd testAlgebra .nat x3 .none (fun _ => 1) testState) = some [11, 11, 11, 10]
#guard cells (atomicAdd testAlgebra .nat
    (.region "x" (.broadcast (.constNat 0) [3])) .none (fun _ => 1) testState) = some [13, 10, 10, 10]
#guard cells (atomicAdd testAlgebra .nat
    (.ptr (.ptrAdd .scalarL (.ptrBase (d := .nat) "x") (.arange 3))) (.mask mask3)
    (fun _ => 1) testState) = some [11, 11, 10, 10]
#guard cells (atomicAdd testAlgebra .nat
    (.blockPtr (.makeBlockPtr (d := .nat) "x" 0 [2] [3] [1] [0]) [0]) .none
    (fun _ => 1) testState) = some [11, 11, 10, 10]
#guard (atomicAdd testAlgebra .real (.region "x" (.constNat 0)) .none
    (fun _ => ()) testState).isNone

private abbrev setIndex : QStmt := .assign .nat [] "saved" (.alg (.ref .nat [] "i"))
private abbrev fail : QStmt := .effectMarker "unsupported"

-- Algorithm-only statements deliberately reject their control-flow forms.
#guard reg (stepAlg testAlgebra (.assign .nat [] "i" (.constNat 7)) testState) = some 7
#guard cells (stepAlg testAlgebra (.store .nat [3] x3 (.arange 3) .none) testState)
    = some [0, 1, 2, 10]
#guard cells (stepAlg testAlgebra (.atomicAdd .nat [3] x3 (.arange 3) .none) testState)
    = some [10, 11, 12, 10]
#guard (stepAlg testAlgebra (.forLoop "i" 1 []) testState).isNone
-- The public step and run entrypoints preserve order and propagate failure.
#guard reg (step testAlgebra (.alg (.assign .nat [] "i" (.constNat 7))) testState) = some 7
#guard cells (step testAlgebra (.store .nat [3] x3 (.alg (.arange 3)) .none) testState)
    = some [0, 1, 2, 10]
#guard cells (step testAlgebra (.atomicAdd .nat [3] x3 (.alg (.arange 3)) .none) testState)
    = some [10, 11, 12, 10]
#guard (run testAlgebra [] testState).isSome
#guard reg (run testAlgebra [.assign .nat [] "i" (.alg (.constNat 7)), setIndex] testState)
    "saved" = some 7
#guard (run testAlgebra [fail, setIndex] testState).isNone
#guard (step testAlgebra (.effectMarker "tl.debug_barrier") testState).isSome
#guard (step testAlgebra fail testState).isNone

-- Each loop binds its exact index; empty and zero-stride ranges do nothing.
#guard reg (loop testAlgebra "i" 2 5 [setIndex] testState) "saved" = some 4
#guard reg (range testAlgebra "i" 2 8 3 [setIndex] testState) "saved" = some 5
#guard (range testAlgebra "i" 2 8 0 [fail] testState).isSome
#guard (loop testAlgebra "i" 5 2 [fail] testState).isSome
#guard (range testAlgebra "i" 8 2 1 [fail] testState).isSome
#guard ((rangeInt testAlgebra "i" (-3) 0 1 [] testState).bind fun t =>
    (t.getReg .int [] "i").map (· ())) = some (-1)
#guard (rangeInt testAlgebra "i" (-3) 0 0 [fail] testState).isSome
#guard (rangeInt testAlgebra "i" 0 (-3) 1 [fail] testState).isSome
#guard (loop testAlgebra "i" 0 1 [fail] testState).isNone
#guard (range testAlgebra "i" 0 1 1 [fail] testState).isNone
#guard (rangeInt testAlgebra "i" 0 1 1 [fail] testState).isNone
#guard reg (step testAlgebra (.forLoop "i" 3 [setIndex]) testState) "saved" = some 2
#guard reg (step testAlgebra (.forRange "i" 1 5 2 [setIndex]) testState) "saved" = some 3
#guard reg (step testAlgebra (.forRangeDyn "i" (.constNat 1) (.constNat 5) (.constNat 2)
    [setIndex]) testState) "saved" = some 3
#guard ((step testAlgebra (.forRangeInt "i" (.constInt (-3)) (.constInt 0) (.constNat 1) [])
    testState).bind fun t => (t.getReg .int [] "i").map (· ())) = some (-1)
#guard (step testAlgebra (.forRangeDyn "i" (.ref .nat [] "unbound") (.constNat 5)
    (.constNat 1) []) testState).isNone
#guard (step testAlgebra (.ifThen (.constBool false) [fail]) testState).isSome
#guard (step testAlgebra (.ifThen (.constBool true) [fail]) testState).isNone
#guard (step testAlgebra (.ifThenElse (.constBool true) [] [fail]) testState).isSome
#guard (step testAlgebra (.ifThenElse (.constBool false) [fail] []) testState).isSome
#guard (step testAlgebra (.ifThen (.ref .bool [] "unbound") []) testState).isNone

end VeriTile.Interp
