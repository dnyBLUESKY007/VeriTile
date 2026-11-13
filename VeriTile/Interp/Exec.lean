/-
VeriTile.Interp.Exec

Computable kernel execution (I2.4). Like Structural.exec, this executes one
program instance; the caller supplies program ids and initial memory.
-/
import VeriTile.Interp.Step

namespace VeriTile.Interp

open VeriTile.Triton

/-- Twin of `Structural.exec`. -/
def exec {α : Type} [Inhabited α] (M : CAlgebra α) :
    QKernel → State α → Option (State α)
  | .mk _ _ code, s => run M code s

/-- Executes one program instance of a kernel; `none` is execution failure. -/
def interp {α : Type} [Inhabited α] (M : CAlgebra α) (kq : QKernel) (s : State α) :
    Option (State α) := exec M kq s

end VeriTile.Interp
