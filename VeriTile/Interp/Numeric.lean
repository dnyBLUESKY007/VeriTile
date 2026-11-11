/-
VeriTile.Interp.Numeric

Numeric helpers of the interpreter (ticket I2.2): the helpers of
`FP.Structural` that `evalOp` applies lane by lane.

`numeric`, `numericLt`, `numericLe` and `numericEq` take an algebra, so they
are restated here over `CAlgebra`; each equals the upstream helper at
`M.lift`. `Structural.bop`, `integralFloorDiv`, `integralMod`, `natLt` and
`natLe` take no algebra and are computable already: the interpreter uses them
as they are, and this file only checks that they evaluate.
-/
import VeriTile.Interp.Algebra

namespace VeriTile.Interp

open VeriTile.Triton
open VeriTile.Triton.FP
open VeriTile.Triton.FP.Structural (Value Binary)

/-- Twin of `Structural.numeric`: exact arithmetic on `nat` and `int`, the
algebra's `binary` on a floating dtype. -/
def numeric {α : Type} (M : CAlgebra α) (p : Option ComputeDType) (op : Binary) :
    {dtype : TileDType} → NumericDType dtype → Value α dtype → Value α dtype → Value α dtype
  | _, .real => M.binary p .real op
  | _, .fp32 => M.binary p .fp32 op
  | _, .fp16 => M.binary p .fp16 op
  | _, .bf16 => M.binary p .bf16 op
  | _, .f8e4 => M.binary p .f8e4 op
  | _, .f8e5 => M.binary p .f8e5 op
  | _, .nat => match op with
      | .add => (· + ·) | .sub => (· - ·) | .mul => (· * ·)
      | .div => (· / ·) | .max => max | .pow => (· ^ ·)
  | _, .int => match op with
      | .add => (· + ·) | .sub => (· - ·) | .mul => (· * ·)
      | .div => (· / ·) | .max => max | .pow => fun a b => a ^ b.toNat

/-- Twin of `Structural.numericLt`. -/
def numericLt {α : Type} (M : CAlgebra α) (p : Option ComputeDType) :
    {dtype : TileDType} → ComparableDType dtype → Option (Value α dtype → Value α dtype → Bool)
  | _, .real => M.compareLt p .real
  | _, .fp32 => M.compareLt p .fp32
  | _, .fp16 => M.compareLt p .fp16
  | _, .bf16 => M.compareLt p .bf16
  | _, .f8e4 => M.compareLt p .f8e4
  | _, .f8e5 => M.compareLt p .f8e5
  | _, .int => some (fun a b => decide (a < b))
  | _, h => Structural.natLt h

/-- Twin of `Structural.numericLe`. -/
def numericLe {α : Type} (M : CAlgebra α) (p : Option ComputeDType) :
    {dtype : TileDType} → ComparableDType dtype → Option (Value α dtype → Value α dtype → Bool)
  | _, .real => M.compareLe p .real
  | _, .fp32 => M.compareLe p .fp32
  | _, .fp16 => M.compareLe p .fp16
  | _, .bf16 => M.compareLe p .bf16
  | _, .f8e4 => M.compareLe p .f8e4
  | _, .f8e5 => M.compareLe p .f8e5
  | _, .int => some (fun a b => decide (a ≤ b))
  | _, h => Structural.natLe h

/-- Twin of `Structural.numericEq`. -/
def numericEq {α : Type} (M : CAlgebra α) (p : Option ComputeDType) :
    {dtype : TileDType} → ComparableDType dtype → Option (Value α dtype → Value α dtype → Bool)
  | _, .real => M.compareEq p .real
  | _, .fp32 => M.compareEq p .fp32
  | _, .fp16 => M.compareEq p .fp16
  | _, .bf16 => M.compareEq p .bf16
  | _, .f8e4 => M.compareEq p .f8e4
  | _, .f8e5 => M.compareEq p .f8e5
  | _, .nat => some (fun a b => decide (a = b))
  | _, .int => some (fun a b => decide (a = b))

/-! Each twin is the upstream helper at the lifted algebra. -/

variable {α : Type} (M : CAlgebra α) (p : Option ComputeDType) {dtype : TileDType}

theorem numeric_lift (op : Binary) (h : NumericDType dtype) :
    Structural.numeric M.lift p op h = numeric M p op h := by cases h <;> rfl

theorem numericLt_lift (h : ComparableDType dtype) :
    Structural.numericLt M.lift p h = numericLt M p h := by cases h <;> rfl

theorem numericLe_lift (h : ComparableDType dtype) :
    Structural.numericLe M.lift p h = numericLe M p h := by cases h <;> rfl

theorem numericEq_lift (h : ComparableDType dtype) :
    Structural.numericEq M.lift p h = numericEq M p h := by cases h <;> rfl

/-! Unit checks on the carrier `Int`. -/

/-- Floating addition is off by 100, so a check tells it from exact addition;
only `compareLt` has a backend. -/
private def A : CAlgebra Int where
  literal := fun _ _ q => q.num
  negInf := -1
  binary := fun _ _ op a b => if op = .add then a + b + 100 else a - b
  compareLt := fun _ _ => some (· < ·)
  unary := fun _ _ a => a
  cast := fun _ _ _ a => a
  fromNat := fun _ n => n
  fromInt := fun _ i => i
  fp32Bits := fun _ => 0
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => 0
  reduceSum := fun _ {_} _ _ _ _ => 0

-- `numeric`: the algebra on a floating dtype, exact on `nat` and `int`.
#guard numeric A none .add .fp32 (1 : Int) 2 = (103 : Int)
#guard numeric A none .mul .real (5 : Int) 2 = (3 : Int)
#guard numeric A none .sub .nat (2 : Nat) 5 = (0 : Nat)
#guard numeric A none .div .nat (7 : Nat) 0 = (0 : Nat)
#guard numeric A none .sub .int (2 : Int) 5 = (-3 : Int)
#guard numeric A none .pow .int (2 : Int) 3 = (8 : Int)
#guard numeric A none .max .int (-2 : Int) 3 = (3 : Int)
-- `numericLt`, `numericLe`, `numericEq`: a missing floating backend is `none`.
#guard ((numericLt A none .fp32).map fun f => f (1 : Int) 2) = some true
#guard ((numericLt A none .int).map fun f => f (2 : Int) 2) = some false
#guard ((numericLt A none .nat).map fun f => f (1 : Nat) 2) = some true
#guard (numericLe A none .fp32).isNone
#guard ((numericLe A none .int).map fun f => f (2 : Int) 2) = some true
#guard ((numericLe A none .nat).map fun f => f (3 : Nat) 2) = some false
#guard (numericEq A none .real).isNone
#guard ((numericEq A none .int).map fun f => f (-2 : Int) (-2)) = some true
#guard ((numericEq A none .nat).map fun f => f (1 : Nat) 2) = some false
-- The reused upstream helpers evaluate. `natLt` and `natLe` are defined on `nat` only.
#guard ((Structural.natLt (α := Int) .nat).map fun f => f (1 : Nat) 2) = some true
#guard (Structural.natLt (α := Int) .int).isNone
#guard ((Structural.natLe (α := Int) .nat).map fun f => f (2 : Nat) 2) = some true
#guard (Structural.natLe (α := Int) .fp32).isNone
-- Floor division and remainder, with the zero-divisor convention of `Nat` and `Int`.
#guard Structural.integralFloorDiv (α := Int) .int (-7 : Int) 2 = (-4 : Int)
#guard Structural.integralFloorDiv (α := Int) .nat (7 : Nat) 0 = (0 : Nat)
#guard Structural.integralMod (α := Int) .int (-7 : Int) 2 = (1 : Int)
#guard Structural.integralMod (α := Int) .nat (7 : Nat) 0 = (7 : Nat)
-- `bop` broadcasts a scalar against a vector.
#guard (TileShape.allIndices [3]).map
    (Structural.bop (· + ·) .scalarL (fun _ => 10) fun (i, _) => i.val) = [10, 11, 12]

end VeriTile.Interp
