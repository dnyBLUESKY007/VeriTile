/-
VeriTile.Staging.Interp.Interface

Interface of the interpreter line, frozen by ticket I0.1. Signed on 2026-11-11.

Every declaration is a stub: an `opaque` constant or a definition with a
`sorry` body. The ticket named in each docstring replaces the stub by a
definition with the same name and type in `VeriTile/Interp/` and deletes the
stub here in the same change. `documents/InterpInterface.md` states the
intended meaning of each name.
-/
import VeriTile.Triton
import VeriTile.Triton.Float.Structural

namespace VeriTile.Interp

open VeriTile.Triton
open VeriTile.Triton.FP

/-- Stub (I1.1). Mirror of `Op`: the same constructors, except that
`const` and `constFloat` carry a rational instead of a real. -/
opaque QOp : TileDType → TileShape → Type

/-- Stub (I1.1). Mirror of `ComputeExpr` over `QOp`. -/
opaque QExpr : TileDType → TileShape → Type

/-- Stub (I1.1). Mirror of `ComputeStmt` over `QOp` and `QExpr`. -/
opaque QStmt : Type

/-- Stub (I1.1). Mirror of `ComputeKernel`: inputs, outputs and a
body of `QStmt`. -/
opaque QKernel : Type

/-- Stub (I1.1). Replaces every rational literal `q` by the real
`(q : ℝ)` and changes nothing else. -/
opaque QKernel.toReal : QKernel → ComputeKernel

/-- Stub (I2.2). The operations of `Structural.Algebra α`, except that
`literal` takes a rational; every field is computable. -/
opaque CAlgebra : Type → Type

/-- Stub (I2.2). The structural algebra with the same operations. On a
real literal that is the image of a rational `q` it returns the `CAlgebra`
literal of `q`. -/
def CAlgebra.lift {α : Type} (M : CAlgebra α) : Structural.Algebra α := sorry

/-- Stub (I2.1). Executable state: registers, finite-map memory of
typed cells, program ids, grid sizes and the undefined-value table. -/
opaque State : Type → Type

/-- Stub (I2.1). The function-valued state that an executable state
represents. -/
def State.lift {α : Type} (s : State α) : Structural.State α := sorry

/-- Stub (I2.4). Executes one program instance of a kernel; `none` is
execution failure. -/
opaque interp {α : Type} [Inhabited α] (M : CAlgebra α) (kq : QKernel) (s : State α) :
    Option (State α)

/-- Stub (I4.1). Forgets dtype tags: the state of the proof semantics
that a real-carrier structural state stands for. -/
opaque erase : Structural.State ℝ → BlockState

end VeriTile.Interp
