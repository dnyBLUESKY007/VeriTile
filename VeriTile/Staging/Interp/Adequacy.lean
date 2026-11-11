/-
VeriTile.Staging.Interp.Adequacy

Adequacy of the interpreter, frozen by ticket I0.1 and proved by I3.3. UNSIGNED.
-/
import VeriTile.Staging.Interp.Interface

namespace VeriTile.Interp

open VeriTile.Triton
open VeriTile.Triton.FP

/-- UNSIGNED. For every carrier and every algebra, the interpreter and the
structural execution fail together or succeed with the same final state. -/
theorem interp_eq_structural_exec {α : Type} [Inhabited α]
    (M : CAlgebra α) (kq : QKernel) (s : State α) :
    (interp M kq s).map State.lift = Structural.exec M.lift kq.toReal s.lift := sorry

end VeriTile.Interp
