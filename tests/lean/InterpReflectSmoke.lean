/-
tests.lean.InterpReflectSmoke

`reflect_kernel` (`VeriTile.Interp.Reflect`) on two bench kernels and on one
kernel that has every supported literal form, and the kernels it must reject.
-/
import Mathlib.Data.Real.Sqrt
import VeriTile.Interp.Reflect
import bench.examples.LogExp.Kernels
import bench.examples.VectorAdd.Kernels

namespace VeriTile.Tests.InterpReflectSmoke

open VeriTile Triton VeriTile.Interp
open VeriTile.Bench.Examples

/-! ## Bench kernels -/

-- `originalKernel` is `addKernel` at three fixed regions, and `addKernel` renames its
-- parameters with `let`: two definitions are unfolded and four `let`s replaced.
reflect_kernel VectorAdd.Kernels.originalKernel

example (blockSize : Nat) :
    (VectorAdd.Kernels.originalKernel.mirror blockSize).toReal
      = VectorAdd.Kernels.originalKernel blockSize :=
  VectorAdd.Kernels.originalKernel.toReal_mirror blockSize

-- The mirror is computable.
#guard match VectorAdd.Kernels.originalKernel.mirror 4 with
  | .mk inputs outputs body => (inputs.length, outputs.length, body.length) == (2, 1, 6)

-- `LogExp.optimizedKernel` has what `VectorAdd` lacks: the literals `0.5`, `80.0` and `0.0`,
-- comparisons, `boolAnd`, `where`, `broadcast`, `absFloat`, `castFloat` and two intrinsics.
reflect_kernel LogExp.optimizedKernel

example (xReg yReg : RegionName) (blockSize : Nat) :
    (LogExp.optimizedKernel.mirror xReg yReg blockSize).toReal
      = LogExp.optimizedKernel xReg yReg blockSize :=
  LogExp.optimizedKernel.toReal_mirror xReg yReg blockSize

/--
info: 'VeriTile.Bench.Examples.LogExp.optimizedKernel.toReal_mirror' depends on axioms: [propext,
 Classical.choice,
 Quot.sound]
-/
#guard_msgs in
#print axioms LogExp.optimizedKernel.toReal_mirror

/-! ## Literal forms, a loop and an algorithm-level statement -/

noncomputable def literalForms (n : Nat) (q : ℚ) : ComputeKernel :=
  .mk [] [] [.forLoop "i" n [
    .assign .real [] "a" (.alg (.add .real .nil (.const (1 / 3 - 2 * 0.25)) (.const (-2.5e-3)))),
    .alg (.assign FloatDType.fp32.toTileDType [] "b" (.constFloat .fp32 q))]]

reflect_kernel literalForms

example (n : Nat) (q : ℚ) : literalForms.mirror n q = .mk [] [] [.forLoop "i" n [
    .assign .real [] "a" (.alg (.add .real .nil (.const (1 / 3 - 2 * 0.25)) (.const (-2.5e-3)))),
    .alg (.assign FloatDType.fp32.toTileDType [] "b" (.constFloat .fp32 q))]] := rfl

example (n : Nat) (q : ℚ) : (literalForms.mirror n q).toReal = literalForms n q :=
  literalForms.toReal_mirror n q

/-! ## Negative examples: kernels that must be rejected -/

noncomputable def irrationalLiteral : ComputeKernel :=
  .mk [] [] [.assign .real [] "a" (.alg (.add .real .nil (.const (1 / 3)) (.const (Real.sqrt 2))))]

/-- error: reflect_kernel: non-rational literal √2 -/
#guard_msgs in
reflect_kernel irrationalLiteral

def realParameter (ε : ℝ) : ComputeKernel :=
  .mk [] [] [.assign .real [] "a" (.alg (.const ε))]

/-- error: reflect_kernel: non-rational literal ε -/
#guard_msgs in
reflect_kernel realParameter

def swapPorts : ComputeKernel → ComputeKernel
  | .mk inputs outputs body => .mk outputs inputs body

def computedKernel : ComputeKernel := swapPorts (.mk [] [] [])

/-- error: reflect_kernel: unsupported constructor VeriTile.Triton.ComputeKernel.rec -/
#guard_msgs in
reflect_kernel computedKernel

end VeriTile.Tests.InterpReflectSmoke
