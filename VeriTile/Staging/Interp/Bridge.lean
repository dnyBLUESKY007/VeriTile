/-
VeriTile.Staging.Interp.Bridge

Bridge from structural execution to the proof semantics, frozen by ticket I0.1
and proved by I4.3. Signed on 2026-11-11 over the carrier `ℝ`; signed again on
2026-11-12 over `WithBot ℝ`, the carrier of the proof semantics, whose `⊥` is
`-inf` (I4.1). `realAlgebra` and `erase` are defined in
`VeriTile.Interp.Bridge.RealAlgebra`.
-/
import VeriTile.Staging.Interp.Interface
import VeriTile.Interp.Bridge.RealAlgebra

namespace VeriTile.Interp

open VeriTile.Triton
open VeriTile.Triton.FP

/-- Stub (I4.2). The named side conditions under which the two
execution definitions agree; each conjunct is listed in
`documents/BridgeTheoremDifferences.md`. -/
opaque BridgePre : ComputeKernel → Structural.State (WithBot ℝ) → Prop

/-- For a source kernel whose dtype-erasing projection succeeds,
structural execution over `WithBot ℝ` agrees with the proof semantics after
erasure. -/
theorem bridge_exec {ck : ComputeKernel} {ak : AlgKernel}
    (s : Structural.State (WithBot ℝ))
    (h : ck.toAlgorithm? = .ok ak) (pre : BridgePre ck s) :
    (Structural.exec realAlgebra ck s).map erase = VeriTile.Triton.exec ak (erase s) := sorry

end VeriTile.Interp
