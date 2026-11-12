/-
VeriTile.Staging.Interp.Interface

Interface of the interpreter line, frozen by ticket I0.1. Signed on 2026-11-11.

Every declaration here is a stub: an `opaque` constant or a definition with a
`sorry` body. The ticket named in each docstring replaces the stub by a
definition with the same name and type in `VeriTile/Interp/`, and the stub is
deleted here. Already replaced: the mirror AST with `QKernel.toReal`
(`VeriTile.Interp.Mirror`, I1.1) and `State` with `State.lift`
(`VeriTile.Interp.State`, I2.1), `CAlgebra` with `CAlgebra.lift`
(`VeriTile.Interp.Algebra`, I2.2), and `erase` with `realAlgebra`
(`VeriTile.Interp.Bridge.RealAlgebra`, I4.1). `documents/InterpInterface.md`
states the intended meaning of each name.
-/
import VeriTile.Triton
import VeriTile.Triton.Float.Structural
import VeriTile.Interp.Algebra
import VeriTile.Interp.Mirror
import VeriTile.Interp.State

namespace VeriTile.Interp

open VeriTile.Triton
open VeriTile.Triton.FP

/-- Stub (I2.4). Executes one program instance of a kernel; `none` is
execution failure. -/
opaque interp {α : Type} [Inhabited α] (M : CAlgebra α) (kq : QKernel) (s : State α) :
    Option (State α)

end VeriTile.Interp
