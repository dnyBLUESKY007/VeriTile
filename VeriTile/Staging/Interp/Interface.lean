/-
VeriTile.Staging.Interp.Interface

Interface of the interpreter line, frozen by ticket I0.1. Signed on 2026-11-11.

Every declaration here is a stub: an `opaque` constant or a definition with a
`sorry` body. The ticket named in each docstring replaces the stub by a
definition with the same name and type in `VeriTile/Interp/`, and the stub is
deleted here. Already replaced: the mirror AST with `QKernel.toReal`
(`VeriTile.Interp.Mirror`, I1.1) and `State` with `State.lift`
(`VeriTile.Interp.State`, I2.1). `documents/InterpInterface.md` states the
intended meaning of each name.
-/
import VeriTile.Triton
import VeriTile.Triton.Float.Structural
import VeriTile.Interp.Mirror
import VeriTile.Interp.State

namespace VeriTile.Interp

open VeriTile.Triton
open VeriTile.Triton.FP

/-- Stub (I2.2). The operations of `Structural.Algebra α`, except that
`literal` takes a rational; every field is computable. -/
opaque CAlgebra : Type → Type

/-- Stub (I2.2). The structural algebra with the same operations. On a
real literal that is the image of a rational `q` it returns the `CAlgebra`
literal of `q`. -/
def CAlgebra.lift {α : Type} (M : CAlgebra α) : Structural.Algebra α := sorry

/-- Stub (I2.4). Executes one program instance of a kernel; `none` is
execution failure. -/
opaque interp {α : Type} [Inhabited α] (M : CAlgebra α) (kq : QKernel) (s : State α) :
    Option (State α)

/-- Stub (I4.1). Forgets dtype tags: the state of the proof semantics
that a real-carrier structural state stands for. -/
opaque erase : Structural.State ℝ → BlockState

end VeriTile.Interp
