/-
VeriTile.Interp.Algebra

The computable algebra of the interpreter (ticket I2.2). `CAlgebra α` has the
fields of `FP.Structural.Algebra α`, except that `literal` takes a rational,
so an algebra over a computable carrier can be executed. `CAlgebra.lift` is
the structural algebra with the same operations; `lift_literal` states what it
does on a rational literal. The command at the end of the file fails the build
when the field names of the two structures differ.
-/
import Mathlib.Data.Rat.Cast.CharZero
import VeriTile.Triton.Float.Structural

namespace VeriTile.Interp

open VeriTile.Triton
open VeriTile.Triton.FP
open VeriTile.Triton.FP.Structural (Values Binary Unary)

/-- `Structural.Algebra α` with a rational `literal`. Every other field has the
name, type, default and meaning of the upstream field. -/
structure CAlgebra (α : Type) where
  literal : Option ComputeDType → FloatDType → ℚ → α
  negInf : α
  binary : Option ComputeDType → FloatDType → Binary → α → α → α
  fma : IndexComputePrecision → Option (α → α → α → α) := fun _ => none
  atomicAdd : FloatDType → Option (α → α → α) := fun _ => none
  unary : Option ComputeDType → Unary → α → α
  libdevicePowi : Option ComputeDType → Option (α → Int → α) := fun _ => none
  isFinite : FinitePrecision → Option (α → Bool) := fun _ => none
  compareLt : Option ComputeDType → FloatDType → Option (α → α → Bool) := fun _ _ => none
  compareLe : Option ComputeDType → FloatDType → Option (α → α → Bool) := fun _ _ => none
  compareEq : Option ComputeDType → FloatDType → Option (α → α → Bool) := fun _ _ => none
  floatToInt : Option IndexComputePrecision → FloatDType → Nat → Option (α → Option Int) :=
    fun _ _ _ => none
  floatToNat : Option IndexComputePrecision → FloatDType → Nat → Option (α → Option Nat) :=
    fun _ _ _ => none
  cast : Option ComputeDType → FloatDType → FloatDType → α → α
  fromNat : Option ComputeDType → Nat → α
  fromInt : Option ComputeDType → Int → α
  fp32Bits : Float32Bits → α
  float8E5Bits : Option (FloatDType → Float8E5Bits → α) := none
  fp32Load : α → α
  reduceMin : Option ComputeDType → {shape : TileShape} →
    (axis : Fin shape.length) → (keepDims : Bool) → Values α .real shape →
    Option (Values α .real (TileShape.reduceShape shape axis keepDims)) :=
    fun _ {_} _ _ _ => none
  reduceMax : Option ComputeDType → {shape : TileShape} →
    (axis : Fin shape.length) → (keepDims : Bool) → Values α .real shape →
    Values α .real (TileShape.reduceShape shape axis keepDims)
  reduceSum : Option ComputeDType → {shape : TileShape} →
    (axis : Fin shape.length) → (keepDims : Bool) → Values α .real shape →
    Values α .real (TileShape.reduceShape shape axis keepDims)
  reduceSumTyped : Option ComputeDType → (dtype : FloatDType) → {shape : TileShape} →
    (axis : Fin shape.length) → (keepDims : Bool) → Values α dtype.toTileDType shape →
    Option (Values α dtype.toTileDType (TileShape.reduceShape shape axis keepDims)) :=
    fun _ _ {_} _ _ _ => none
  reduceMaxTyped : Option ComputeDType → (dtype : FloatDType) → {shape : TileShape} →
    (axis : Fin shape.length) → (keepDims : Bool) → Values α dtype.toTileDType shape →
    Option (Values α dtype.toTileDType (TileShape.reduceShape shape axis keepDims)) :=
    fun _ _ {_} _ _ _ => none
  indexedMax : Option ComputeDType → (dtype : FloatDType) → {shape : TileShape} →
    (axis : Fin shape.length) → Values α dtype.toTileDType shape →
    Option (Values α dtype.toTileDType (TileShape.eraseAxis shape axis) ×
      Values α .nat (TileShape.eraseAxis shape axis)) :=
    fun _ _ {_} _ _ => none
  scan : Option ComputeDType → {shape : TileShape} → ScanOp →
    (axis : Fin shape.length) → ScanDirection → Values α .real shape →
    Option (Values α .real shape) := fun _ {_} _ _ _ _ => none
  dot : Option ComputeDType → {batch : TileShape} → {m k n : Nat} →
    Values α .real (batch ++ [m, k]) → Values α .real (batch ++ [k, n]) →
    Option (Values α .real (batch ++ [m, n])) := fun _ {_} {_} {_} {_} _ _ => none
  dotWithOptions : Option ComputeDType → DotOptions →
    {batch : TileShape} → {m k n : Nat} →
    Values α .real (batch ++ [m, k]) → Values α .real (batch ++ [k, n]) →
    Option (Values α .real (batch ++ [m, n])) := fun _ _ {_} {_} {_} {_} _ _ => none
  dotTyped : Option ComputeDType → (left right out : FloatDType) → DotOptions →
    {batch : TileShape} → {m k n : Nat} →
    Values α left.toTileDType (batch ++ [m, k]) →
    Values α right.toTileDType (batch ++ [k, n]) →
    Option (Values α out.toTileDType (batch ++ [m, n])) :=
    fun _ _ _ _ _ {_} {_} {_} {_} _ _ => none
  dotAcc : Option ComputeDType → (left right out : FloatDType) → DotOptions →
    {batch : TileShape} → {m k n : Nat} →
    Values α left.toTileDType (batch ++ [m, k]) →
    Values α right.toTileDType (batch ++ [k, n]) →
    Values α out.toTileDType (batch ++ [m, n]) →
    Option (Values α out.toTileDType (batch ++ [m, n])) :=
    fun _ _ _ _ _ {_} {_} {_} {_} _ _ _ => none
  integerDot : DotOptions → {batch : TileShape} → {m k n : Nat} →
    Values α .int (batch ++ [m,k]) → Values α .int (batch ++ [k,n]) →
    Option (Values α .int (batch ++ [m,n])) := fun _ {_} {_} {_} {_} _ _ => none
  fmaTyped : FloatDType → Option (α → α → α → α) := fun _ => none
  minimum : Option ComputeDType → FloatDType → Option (α → α → α) :=
    fun _ _ => none
  divRN : Option (α → α → α) := none

/-- The structural algebra with the operations of `M`. Its `literal` on the
real `(q : ℝ)` is the `M` literal of `q` (`lift_literal`); on a real that is not
rational it is the `M` literal of an unspecified rational. -/
noncomputable def CAlgebra.lift {α : Type} (M : CAlgebra α) : Structural.Algebra α :=
  -- Every field but `literal` is copied from `M` by name.
  { M with literal := fun p d x => M.literal p d (Classical.epsilon fun q : ℚ => (q : ℝ) = x) }

theorem CAlgebra.lift_literal {α : Type} (M : CAlgebra α) (p : Option ComputeDType)
    (d : FloatDType) (q : ℚ) : M.lift.literal p d (q : ℝ) = M.literal p d q :=
  congrArg (M.literal p d)
    (Rat.cast_injective (Classical.epsilon_spec (p := fun r : ℚ => (r : ℝ) = q) ⟨q, rfl⟩))

/-! ## Field check -/

-- `lift` copies fields by name, so a field that exists on one side only would be
-- dropped or left at its default without an error; this fails the build instead.
open Lean Elab Command in
run_cmd
  let env ← getEnv
  let (u, c) := (getStructureFields env ``Structural.Algebra, getStructureFields env ``CAlgebra)
  unless u == c do
    throwError "CAlgebra does not have the fields of Structural.Algebra in the same order:\n  \
      missing {u.filter (!c.contains ·)}\n  extra {c.filter (!u.contains ·)}"

end VeriTile.Interp
