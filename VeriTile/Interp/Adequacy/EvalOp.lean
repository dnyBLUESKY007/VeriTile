/-
VeriTile.Interp.Adequacy.EvalOp

The executable evaluators agree with Structural on the lifted algebra and
state, including their failure cases. No operation or carrier is assumed.

Constructor coverage: QOp 113/113, QComputeOp 5/5, QExpr 7/7.
-/
import VeriTile.Interp.Adequacy.State
import VeriTile.Interp.EvalOp

namespace VeriTile.Interp

open VeriTile.Triton
open VeriTile.Triton.FP

variable {α : Type} [Inhabited α]

theorem boundaryZero_eq (M : CAlgebra α) (p : Option ComputeDType) (d : TileDType) :
    boundaryZero M p d = Structural.boundaryZero M.lift p d := by
  cases d <;> simp only [boundaryZero, Structural.boundaryZero, CAlgebra.lift_literal_zero]

set_option maxHeartbeats 3200000 in
mutual
theorem evalOp_eq (M : CAlgebra α) (p : Option ComputeDType)
    (e : QOp d sh) (s : State α) :
    evalOp M p e s = Structural.evalOp M.lift p e.toReal s.lift := by
  match e with
  | .const q =>
    change evalOp M p (.const q) s = Structural.evalOp M.lift p (.const (q : ℝ)) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [CAlgebra.lift_literal]
  | .constFloat dtype q =>
    change evalOp M p (.constFloat dtype q) s = Structural.evalOp M.lift p (.constFloat dtype (q : ℝ)) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [CAlgebra.lift_literal]
  | .constNat n =>
    change evalOp M p (.constNat n) s = Structural.evalOp M.lift p (.constNat n) s.lift
    rfl
  | .constInt i =>
    change evalOp M p (.constInt i) s = Structural.evalOp M.lift p (.constInt i) s.lift
    rfl
  | .constBool v =>
    change evalOp M p (.constBool v) s = Structural.evalOp M.lift p (.constBool v) s.lift
    rfl
  | .negInf =>
    change evalOp M p (.negInf) s = Structural.evalOp M.lift p (.negInf) s.lift
    rfl
  | .programId axis =>
    change evalOp M p (.programId axis) s = Structural.evalOp M.lift p (.programId axis) s.lift
    rfl
  | .numPrograms axis =>
    change evalOp M p (.numPrograms axis) s = Structural.evalOp M.lift p (.numPrograms axis) s.lift
    rfl
  | .ref dtype shape name =>
    change evalOp M p (.ref dtype shape name) s = Structural.evalOp M.lift p (.ref dtype shape name) s.lift
    rfl
  | .arange n =>
    change evalOp M p (.arange n) s = Structural.evalOp M.lift p (.arange n) s.lift
    rfl
  | .rand cfg e₁ e₂ =>
    change evalOp M p (.rand cfg e₁ e₂) s = Structural.evalOp M.lift p (.rand cfg e₁.toReal e₂.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e₁ s, ← evalOp_eq M p e₂ s] <;> rfl
  | .randTensor cfg e₁ e₂ =>
    change evalOp M p (.randTensor cfg e₁ e₂) s =
      Structural.evalOp M.lift p (.randTensor cfg e₁.toReal e₂.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e₁ s, ← evalOp_eq M p e₂ s] <;> rfl
  | .philox4x32 cfg plan field e₁ e₂ e₃ e₄ e₅ =>
    change evalOp M p (.philox4x32 cfg plan field e₁ e₂ e₃ e₄ e₅) s =
      Structural.evalOp M.lift p (.philox4x32 cfg plan field e₁.toReal e₂.toReal e₃.toReal e₄.toReal e₅.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e₁ s, ← evalOp_eq M p e₂ s, ← evalOp_eq M p e₃ s, ← evalOp_eq M p e₄ s,
      ← evalOp_eq M p e₅ s]
  | .uniform32 e =>
    change evalOp M p (.uniform32 e) s = Structural.evalOp M.lift p (.uniform32 e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s] <;> rfl
  | .broadcast e shape =>
    change evalOp M p (.broadcast e shape) s = Structural.evalOp M.lift p (.broadcast e.toReal shape) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s]
  | .full shape e =>
    change evalOp M p (.full shape e) s = Structural.evalOp M.lift p (.full shape e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s]
  | .withPrecision dtype prec e =>
    change evalOp M p (.withPrecision dtype prec e) s =
      Structural.evalOp M.lift p (.withPrecision dtype prec e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold]
    dsimp only
    rw [← evalOp_eq M _ e s] <;>
    rfl
  | .castFloat src dst e =>
    change evalOp M p (.castFloat src dst e) s =
      Structural.evalOp M.lift p (.castFloat src dst e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s] <;> rfl
  | .negFloat dtype e =>
    change evalOp M p (.negFloat dtype e) s = Structural.evalOp M.lift p (.negFloat dtype e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s] <;> rfl
  | .absFloat dtype e =>
    change evalOp M p (.absFloat dtype e) s = Structural.evalOp M.lift p (.absFloat dtype e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s] <;> rfl
  | .floatToInt prec src n e =>
    change evalOp M p (.floatToInt prec src n e) s =
      Structural.evalOp M.lift p (.floatToInt prec src n e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold]
    dsimp only
    rw [← evalOp_eq M _ e s] <;>
    first | rfl | (simp only [← collectValues?_eq]; rfl)
  | .floatToNat prec src n e =>
    change evalOp M p (.floatToNat prec src n e) s =
      Structural.evalOp M.lift p (.floatToNat prec src n e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold]
    dsimp only
    rw [← evalOp_eq M _ e s] <;>
    first | rfl | (simp only [← collectValues?_eq]; rfl)
  | .castNatToInt e =>
    change evalOp M p (.castNatToInt e) s = Structural.evalOp M.lift p (.castNatToInt e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s]
  | .castIntToNat e =>
    change evalOp M p (.castIntToNat e) s = Structural.evalOp M.lift p (.castIntToNat e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s]
  | .castRealToInt8 e =>
    rfl
  | .decodeFloat8E5 dst e =>
    change evalOp M p (.decodeFloat8E5 dst e) s =
      Structural.evalOp M.lift p (.decodeFloat8E5 dst e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s] <;> rfl
  | .add h bc e₁ e₂ =>
    change evalOp M p (.add h bc e₁ e₂) s = Structural.evalOp M.lift p (.add h bc e₁.toReal e₂.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e₁ s, ← evalOp_eq M p e₂ s, ← numeric_lift]
  | .sub h bc e₁ e₂ =>
    change evalOp M p (.sub h bc e₁ e₂) s = Structural.evalOp M.lift p (.sub h bc e₁.toReal e₂.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e₁ s, ← evalOp_eq M p e₂ s, ← numeric_lift]
  | .mul h bc e₁ e₂ =>
    change evalOp M p (.mul h bc e₁ e₂) s = Structural.evalOp M.lift p (.mul h bc e₁.toReal e₂.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e₁ s, ← evalOp_eq M p e₂ s, ← numeric_lift]
  | .div h bc e₁ e₂ =>
    change evalOp M p (.div h bc e₁ e₂) s = Structural.evalOp M.lift p (.div h bc e₁.toReal e₂.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e₁ s, ← evalOp_eq M p e₂ s, ← numeric_lift]
  | .divRN bc e₁ e₂ =>
    change evalOp M p (.divRN bc e₁ e₂) s = Structural.evalOp M.lift p (.divRN bc e₁.toReal e₂.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e₁ s, ← evalOp_eq M p e₂ s] <;> rfl
  | .floorDiv h bc e₁ e₂ =>
    change evalOp M p (.floorDiv h bc e₁ e₂) s =
      Structural.evalOp M.lift p (.floorDiv h bc e₁.toReal e₂.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e₁ s, ← evalOp_eq M p e₂ s]
  | .mod h bc e₁ e₂ =>
    change evalOp M p (.mod h bc e₁ e₂) s = Structural.evalOp M.lift p (.mod h bc e₁.toReal e₂.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e₁ s, ← evalOp_eq M p e₂ s]
  | .bitAnd bc e₁ e₂ =>
    change evalOp M p (.bitAnd bc e₁ e₂) s =
      Structural.evalOp M.lift p (.bitAnd bc e₁.toReal e₂.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e₁ s, ← evalOp_eq M p e₂ s]
  | .bitOr bc e₁ e₂ =>
    change evalOp M p (.bitOr bc e₁ e₂) s = Structural.evalOp M.lift p (.bitOr bc e₁.toReal e₂.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e₁ s, ← evalOp_eq M p e₂ s]
  | .bitXor bc e₁ e₂ =>
    change evalOp M p (.bitXor bc e₁ e₂) s =
      Structural.evalOp M.lift p (.bitXor bc e₁.toReal e₂.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e₁ s, ← evalOp_eq M p e₂ s]
  | .shiftLeft bc e₁ e₂ =>
    change evalOp M p (.shiftLeft bc e₁ e₂) s =
      Structural.evalOp M.lift p (.shiftLeft bc e₁.toReal e₂.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e₁ s, ← evalOp_eq M p e₂ s]
  | .shiftRight bc e₁ e₂ =>
    change evalOp M p (.shiftRight bc e₁ e₂) s =
      Structural.evalOp M.lift p (.shiftRight bc e₁.toReal e₂.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e₁ s, ← evalOp_eq M p e₂ s]
  | .exp e =>
    change evalOp M p (.exp e) s = Structural.evalOp M.lift p (.exp e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s] <;> rfl
  | .libdeviceExp e =>
    change evalOp M p (.libdeviceExp e) s = Structural.evalOp M.lift p (.libdeviceExp e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s] <;> rfl
  | .libdeviceLog e =>
    change evalOp M p (.libdeviceLog e) s = Structural.evalOp M.lift p (.libdeviceLog e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s] <;> rfl
  | .libdeviceExpm1 e =>
    change evalOp M p (.libdeviceExpm1 e) s = Structural.evalOp M.lift p (.libdeviceExpm1 e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s] <;> rfl
  | .libdeviceLog1p e =>
    change evalOp M p (.libdeviceLog1p e) s = Structural.evalOp M.lift p (.libdeviceLog1p e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s] <;> rfl
  | .libdeviceTanh e =>
    change evalOp M p (.libdeviceTanh e) s = Structural.evalOp M.lift p (.libdeviceTanh e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s] <;> rfl
  | .exp2 e =>
    change evalOp M p (.exp2 e) s = Structural.evalOp M.lift p (.exp2 e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s] <;> rfl
  | .log e =>
    change evalOp M p (.log e) s = Structural.evalOp M.lift p (.log e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s] <;> rfl
  | .log2 e =>
    change evalOp M p (.log2 e) s = Structural.evalOp M.lift p (.log2 e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s] <;> rfl
  | .sigmoid e =>
    change evalOp M p (.sigmoid e) s = Structural.evalOp M.lift p (.sigmoid e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s] <;> rfl
  | .sqrt e =>
    change evalOp M p (.sqrt e) s = Structural.evalOp M.lift p (.sqrt e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s] <;> rfl
  | .rsqrt e =>
    change evalOp M p (.rsqrt e) s = Structural.evalOp M.lift p (.rsqrt e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s] <;> rfl
  | .libdeviceRsqrt e =>
    change evalOp M p (.libdeviceRsqrt e) s = Structural.evalOp M.lift p (.libdeviceRsqrt e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s] <;> rfl
  | .tanh e =>
    change evalOp M p (.tanh e) s = Structural.evalOp M.lift p (.tanh e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s] <;> rfl
  | .sin e =>
    change evalOp M p (.sin e) s = Structural.evalOp M.lift p (.sin e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s] <;> rfl
  | .cos e =>
    change evalOp M p (.cos e) s = Structural.evalOp M.lift p (.cos e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s] <;> rfl
  | .tan e =>
    change evalOp M p (.tan e) s = Structural.evalOp M.lift p (.tan e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s] <;> rfl
  | .atan e =>
    change evalOp M p (.atan e) s = Structural.evalOp M.lift p (.atan e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s] <;> rfl
  | .cosh e =>
    change evalOp M p (.cosh e) s = Structural.evalOp M.lift p (.cosh e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s] <;> rfl
  | .sinh e =>
    change evalOp M p (.sinh e) s = Structural.evalOp M.lift p (.sinh e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s] <;> rfl
  | .erf e =>
    change evalOp M p (.erf e) s = Structural.evalOp M.lift p (.erf e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s] <;> rfl
  | .minPosInf e =>
    change evalOp M p (.minPosInf e) s = Structural.evalOp M.lift p (.minPosInf e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s] <;> rfl
  | .lt h bc e₁ e₂ =>
    change evalOp M p (.lt h bc e₁ e₂) s = Structural.evalOp M.lift p (.lt h bc e₁.toReal e₂.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e₁ s, ← evalOp_eq M p e₂ s, ← numericLt_lift]
  | .le h bc e₁ e₂ =>
    change evalOp M p (.le h bc e₁ e₂) s = Structural.evalOp M.lift p (.le h bc e₁.toReal e₂.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e₁ s, ← evalOp_eq M p e₂ s, ← numericLe_lift]
  | .eq h bc e₁ e₂ =>
    change evalOp M p (.eq h bc e₁ e₂) s = Structural.evalOp M.lift p (.eq h bc e₁.toReal e₂.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e₁ s, ← evalOp_eq M p e₂ s, ← numericEq_lift]
  | .gt h bc e₁ e₂ =>
    change evalOp M p (.gt h bc e₁ e₂) s = Structural.evalOp M.lift p (.gt h bc e₁.toReal e₂.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e₁ s, ← evalOp_eq M p e₂ s, ← numericLt_lift]
  | .ge h bc e₁ e₂ =>
    change evalOp M p (.ge h bc e₁ e₂) s = Structural.evalOp M.lift p (.ge h bc e₁.toReal e₂.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e₁ s, ← evalOp_eq M p e₂ s, ← numericLe_lift]
  | .ne h bc e₁ e₂ =>
    change evalOp M p (.ne h bc e₁ e₂) s = Structural.evalOp M.lift p (.ne h bc e₁.toReal e₂.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e₁ s, ← evalOp_eq M p e₂ s, ← numericEq_lift]
  | .boolAnd bc e₁ e₂ =>
    change evalOp M p (.boolAnd bc e₁ e₂) s =
      Structural.evalOp M.lift p (.boolAnd bc e₁.toReal e₂.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e₁ s, ← evalOp_eq M p e₂ s]
  | .boolOr bc e₁ e₂ =>
    change evalOp M p (.boolOr bc e₁ e₂) s =
      Structural.evalOp M.lift p (.boolOr bc e₁.toReal e₂.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e₁ s, ← evalOp_eq M p e₂ s]
  | .boolNot e =>
    change evalOp M p (.boolNot e) s = Structural.evalOp M.lift p (.boolNot e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s]
  | .max2 bc e₁ e₂ =>
    change evalOp M p (.max2 bc e₁ e₂) s = Structural.evalOp M.lift p (.max2 bc e₁.toReal e₂.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e₁ s, ← evalOp_eq M p e₂ s] <;> rfl
  | .maximumFloat dtype bc e₁ e₂ =>
    change evalOp M p (.maximumFloat dtype bc e₁ e₂) s =
      Structural.evalOp M.lift p (.maximumFloat dtype bc e₁.toReal e₂.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e₁ s, ← evalOp_eq M p e₂ s] <;> rfl
  | .pow bc e₁ e₂ =>
    change evalOp M p (.pow bc e₁ e₂) s = Structural.evalOp M.lift p (.pow bc e₁.toReal e₂.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e₁ s, ← evalOp_eq M p e₂ s] <;> rfl
  | .libdevicePowi bc e₁ e₂ =>
    change evalOp M p (.libdevicePowi bc e₁ e₂) s =
      Structural.evalOp M.lift p (.libdevicePowi bc e₁.toReal e₂.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e₁ s, ← evalOp_eq M p e₂ s] <;> rfl
  | .where e₁ e₂ e₃ =>
    change evalOp M p (.where e₁ e₂ e₃) s =
      Structural.evalOp M.lift p (.where e₁.toReal e₂.toReal e₃.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e₁ s, ← evalOp_eq M p e₂ s, ← evalOp_eq M p e₃ s]
  | .ite e₁ e₂ e₃ =>
    change evalOp M p (.ite e₁ e₂ e₃) s =
      Structural.evalOp M.lift p (.ite e₁.toReal e₂.toReal e₃.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e₁ s, ← evalOp_eq M p e₂ s, ← evalOp_eq M p e₃ s]
  | .reduceMin axis keepDims e =>
    change evalOp M p (.reduceMin axis keepDims e) s =
      Structural.evalOp M.lift p (.reduceMin axis keepDims e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s] <;> rfl
  | .reduceMax axis keepDims e =>
    change evalOp M p (.reduceMax axis keepDims e) s =
      Structural.evalOp M.lift p (.reduceMax axis keepDims e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s] <;> rfl
  | .reduceMaxNat axis keepDims e =>
    change evalOp M p (.reduceMaxNat axis keepDims e) s =
      Structural.evalOp M.lift p (.reduceMaxNat axis keepDims e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s]
  | .reduceMaxInt axis keepDims e =>
    change evalOp M p (.reduceMaxInt axis keepDims e) s =
      Structural.evalOp M.lift p (.reduceMaxInt axis keepDims e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s]
  | .reduceSum axis keepDims e =>
    change evalOp M p (.reduceSum axis keepDims e) s =
      Structural.evalOp M.lift p (.reduceSum axis keepDims e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s] <;> rfl
  | .reduceSumBool axis keepDims e =>
    change evalOp M p (.reduceSumBool axis keepDims e) s =
      Structural.evalOp M.lift p (.reduceSumBool axis keepDims e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s]
  | .reduceSumTyped dtype axis keepDims e =>
    change evalOp M p (.reduceSumTyped dtype axis keepDims e) s =
      Structural.evalOp M.lift p (.reduceSumTyped dtype axis keepDims e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s] <;> rfl
  | .reduceMaxTyped dtype axis keepDims e =>
    change evalOp M p (.reduceMaxTyped dtype axis keepDims e) s =
      Structural.evalOp M.lift p (.reduceMaxTyped dtype axis keepDims e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s] <;> rfl
  | .scan op axis dir e =>
    change evalOp M p (.scan op axis dir e) s = Structural.evalOp M.lift p (.scan op axis dir e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s] <;> rfl
  | .argMax axis e =>
    change evalOp M p (.argMax axis e) s = Structural.evalOp M.lift p (.argMax axis e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s] <;> rfl
  | .argMaxInt axis e =>
    change evalOp M p (.argMaxInt axis e) s = Structural.evalOp M.lift p (.argMaxInt axis e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s]
  | .indexedMaxValue dtype axis e =>
    change evalOp M p (.indexedMaxValue dtype axis e) s =
      Structural.evalOp M.lift p (.indexedMaxValue dtype axis e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s] <;> rfl
  | .argMaxTyped dtype axis e precision =>
    change evalOp M p (.argMaxTyped dtype axis e precision) s =
      Structural.evalOp M.lift p (.argMaxTyped dtype axis e.toReal precision) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold]
    dsimp only
    rw [← evalOp_eq M _ e s] <;>
    rfl
  | .argMin axis e =>
    change evalOp M p (.argMin axis e) s = Structural.evalOp M.lift p (.argMin axis e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s]
  | .sort axis e =>
    change evalOp M p (.sort axis e) s = Structural.evalOp M.lift p (.sort axis e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s]
  | .dot e₁ e₂ options =>
    change evalOp M p (.dot e₁ e₂ options) s =
      Structural.evalOp M.lift p (.dot e₁.toReal e₂.toReal options) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e₁ s, ← evalOp_eq M p e₂ s] <;> rfl
  | .dotTyped left right out e₁ e₂ options =>
    change evalOp M p (.dotTyped left right out e₁ e₂ options) s =
      Structural.evalOp M.lift p (.dotTyped left right out e₁.toReal e₂.toReal options) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e₁ s, ← evalOp_eq M p e₂ s] <;> rfl
  | .dotAcc left right out e₁ e₂ e₃ options =>
    change evalOp M p (.dotAcc left right out e₁ e₂ e₃ options) s =
      Structural.evalOp M.lift p (.dotAcc left right out e₁.toReal e₂.toReal e₃.toReal options) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e₁ s, ← evalOp_eq M p e₂ s, ← evalOp_eq M p e₃ s] <;> rfl
  | .dotInt e₁ e₂ options =>
    change evalOp M p (.dotInt e₁ e₂ options) s =
      Structural.evalOp M.lift p (.dotInt e₁.toReal e₂.toReal options) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e₁ s, ← evalOp_eq M p e₂ s] <;> rfl
  | .transpose e =>
    change evalOp M p (.transpose e) s = Structural.evalOp M.lift p (.transpose e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s]
  | .reshape outShape e =>
    change evalOp M p (.reshape outShape e) s = Structural.evalOp M.lift p (.reshape outShape e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s]
  | .remap outShape f e =>
    change evalOp M p (.remap outShape f e) s = Structural.evalOp M.lift p (.remap outShape f e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s]
  | .join e₁ e₂ =>
    change evalOp M p (.join e₁ e₂) s = Structural.evalOp M.lift p (.join e₁.toReal e₂.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e₁ s, ← evalOp_eq M p e₂ s]
  | .split side e =>
    change evalOp M p (.split side e) s = Structural.evalOp M.lift p (.split side e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s]
  | .expandDim axis e =>
    change evalOp M p (.expandDim axis e) s = Structural.evalOp M.lift p (.expandDim axis e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s]
  | .ptrBase r =>
    change evalOp M p (.ptrBase r) s = Structural.evalOp M.lift p (.ptrBase r) s.lift
    rfl
  | .ptrAdd bc e₁ e₂ =>
    change evalOp M p (.ptrAdd bc e₁ e₂) s =
      Structural.evalOp M.lift p (.ptrAdd bc e₁.toReal e₂.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e₁ s, ← evalOp_eq M p e₂ s]
  | .ptrSub bc e₁ e₂ =>
    change evalOp M p (.ptrSub bc e₁ e₂) s =
      Structural.evalOp M.lift p (.ptrSub bc e₁.toReal e₂.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e₁ s, ← evalOp_eq M p e₂ s]
  | .makeBlockPtr r baseOffset parentShape blockShape strides offsets =>
    change evalOp M p (.makeBlockPtr r baseOffset parentShape blockShape strides offsets) s =
      Structural.evalOp M.lift p (.makeBlockPtr r baseOffset parentShape blockShape strides offsets) s.lift
    rfl
  | .makeBlockPtrDyn r baseOffset parentShape blockShape strides offsets =>
    change evalOp M p (.makeBlockPtrDyn r baseOffset parentShape blockShape strides offsets) s =
      Structural.evalOp M.lift p (.makeBlockPtrDyn r baseOffset.toReal parentShape blockShape strides offsets) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p baseOffset s]
  | .makeBlockPtrDynOffsets r baseOffset parentShape blockShape strides offsets =>
    change evalOp M p (.makeBlockPtrDynOffsets r baseOffset parentShape blockShape strides offsets) s =
      Structural.evalOp M.lift p (.makeBlockPtrDynOffsets r baseOffset.toReal parentShape blockShape strides (QOp.toRealNats offsets)) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p baseOffset s, ← evalOffsets_eq M p offsets s]
  | .makeBlockPtrDynMetadata r baseOffset parentShape blockShape strides offsets =>
    change evalOp M p (.makeBlockPtrDynMetadata r baseOffset parentShape blockShape strides offsets) s =
      Structural.evalOp M.lift p (.makeBlockPtrDynMetadata r baseOffset.toReal parentShape blockShape (QOp.toRealNats strides) (QOp.toRealNats offsets)) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p baseOffset s, ← evalOffsets_eq M p strides s,
      ← evalOffsets_eq M p offsets s]
  | .advanceBlockPtr e offsetDeltas =>
    change evalOp M p (.advanceBlockPtr e offsetDeltas) s =
      Structural.evalOp M.lift p (.advanceBlockPtr e.toReal offsetDeltas) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s]
  | .advanceBlockPtrDyn e deltas =>
    change evalOp M p (.advanceBlockPtrDyn e deltas) s =
      Structural.evalOp M.lift p (.advanceBlockPtrDyn e.toReal (QOp.toRealInts deltas)) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s, ← evalDeltas_eq M p deltas s]
  | .load dtype mem msk padding =>
    change evalOp M p (.load dtype mem msk padding) s =
      Structural.evalOp M.lift p (.load dtype mem.toReal msk.toReal padding) s.lift
    match mem, msk with
    | .region r off, .none =>
      rw [evalOp_unfold, Structural.evalOp_unfold] <;>
      simp only [QMemAccess.toReal, QMaskOpt.toReal, ← evalOp_eq M p off s, ← boundaryZero_eq,
      State.lift_mem, State.lift_undef]
    | .region r off, .mask active =>
      rw [evalOp_unfold, Structural.evalOp_unfold] <;>
      simp only [QMemAccess.toReal, QMaskOpt.toReal, ← evalOp_eq M p off s, ← evalOp_eq M p active s,
      ← boundaryZero_eq, State.lift_mem, State.lift_undef]
    | .region r off, .maskOther active other =>
      rw [evalOp_unfold, Structural.evalOp_unfold] <;>
      simp only [QMemAccess.toReal, QMaskOpt.toReal, ← evalOp_eq M p off s, ← evalOp_eq M p active s,
      ← evalOp_eq M p other s, ← boundaryZero_eq, State.lift_mem, State.lift_undef]
    | .ptr ptr, .none =>
      rw [evalOp_unfold, Structural.evalOp_unfold] <;>
      simp only [QMemAccess.toReal, QMaskOpt.toReal, ← evalOp_eq M p ptr s, ← boundaryZero_eq,
      State.lift_mem, State.lift_undef]
    | .ptr ptr, .mask active =>
      rw [evalOp_unfold, Structural.evalOp_unfold] <;>
      simp only [QMemAccess.toReal, QMaskOpt.toReal, ← evalOp_eq M p ptr s, ← evalOp_eq M p active s,
      ← boundaryZero_eq, State.lift_mem, State.lift_undef]
    | .ptr ptr, .maskOther active other =>
      rw [evalOp_unfold, Structural.evalOp_unfold] <;>
      simp only [QMemAccess.toReal, QMaskOpt.toReal, ← evalOp_eq M p ptr s, ← evalOp_eq M p active s,
      ← evalOp_eq M p other s, ← boundaryZero_eq, State.lift_mem, State.lift_undef]
    | .blockPtr ptr checks, .none =>
      rw [evalOp_unfold, Structural.evalOp_unfold] <;>
      simp only [QMemAccess.toReal, QMaskOpt.toReal, ← evalOp_eq M p ptr s, ← boundaryZero_eq,
      State.lift_mem, State.lift_undef] <;> rfl
    | .blockPtr ptr checks, .mask active =>
      rw [evalOp_unfold, Structural.evalOp_unfold] <;>
      simp only [QMemAccess.toReal, QMaskOpt.toReal, ← evalOp_eq M p ptr s, ← evalOp_eq M p active s,
      ← boundaryZero_eq, State.lift_mem, State.lift_undef] <;> rfl
    | .blockPtr ptr checks, .maskOther active other =>
      rw [evalOp_unfold, Structural.evalOp_unfold] <;>
      simp only [QMemAccess.toReal, QMaskOpt.toReal, ← evalOp_eq M p ptr s, ← evalOp_eq M p active s,
      ← evalOp_eq M p other s, ← boundaryZero_eq, State.lift_mem, State.lift_undef] <;> rfl
  | .natToReal e =>
    change evalOp M p (.natToReal e) s = Structural.evalOp M.lift p (.natToReal e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s] <;> rfl
  | .intToReal e =>
    change evalOp M p (.intToReal e) s = Structural.evalOp M.lift p (.intToReal e.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e s] <;> rfl
  | .minimumFloat dtype bc e₁ e₂ =>
    change evalOp M p (.minimumFloat dtype bc e₁ e₂) s =
      Structural.evalOp M.lift p (.minimumFloat dtype bc e₁.toReal e₂.toReal) s.lift
    rw [evalOp_unfold, Structural.evalOp_unfold] <;>
    simp only [← evalOp_eq M p e₁ s, ← evalOp_eq M p e₂ s] <;> rfl
termination_by sizeOf e
decreasing_by all_goals simp_wf <;> omega

theorem evalOffsets_eq (M : CAlgebra α) (p : Option ComputeDType)
    (es : List (QOp .nat [])) (s : State α) :
    evalOffsets M p es s = Structural.evalOffsets M.lift p (QOp.toRealNats es) s.lift := by
  cases es with
  | nil => rfl
  | cons e es =>
    simp only [evalOffsets, QOp.toRealNats, Structural.evalOffsets,
      ← evalOp_eq M p e s, ← evalOffsets_eq M p es s]
termination_by sizeOf es
decreasing_by all_goals simp_wf <;> omega

theorem evalDeltas_eq (M : CAlgebra α) (p : Option ComputeDType)
    (es : List (QOp .int [])) (s : State α) :
    evalDeltas M p es s = Structural.evalDeltas M.lift p (QOp.toRealInts es) s.lift := by
  cases es with
  | nil => rfl
  | cons e es =>
    simp only [evalDeltas, QOp.toRealInts, Structural.evalDeltas,
      ← evalOp_eq M p e s, ← evalDeltas_eq M p es s]
termination_by sizeOf es
decreasing_by all_goals simp_wf <;> omega
end

theorem evalComputeOp_eq (M : CAlgebra α) (e : QComputeOp d sh) (s : State α) :
    evalComputeOp M e s = Structural.evalComputeOp M.lift e.toReal s.lift := by
  match e with
  | .alg d e => exact evalOp_eq M (some d) e s
  | .const (dtype := d) c => cases d <;> rfl
  | .full sh e =>
    simp only [evalComputeOp, QComputeOp.toReal, Structural.evalComputeOp,
      ← evalComputeOp_eq M e s]
  | .load d mem mask padding =>
    cases d <;>
      simp only [evalComputeOp, QComputeOp.toReal, Structural.evalComputeOp]
    all_goals
      rw [evalOp_eq]
      rfl
  | .bitcast src dst h e =>
    have ih := evalComputeOp_eq M e s
    cases src <;> cases dst <;>
      simp only [evalComputeOp, QComputeOp.toReal, Structural.evalComputeOp,
        ← ih] <;> rfl
termination_by sizeOf e
decreasing_by all_goals simp_wf

theorem evalExpr_eq (M : CAlgebra α) (e : QExpr d sh) (s : State α) :
    evalExpr M e s = Structural.evalExpr M.lift e.toReal s.lift := by
  match e with
  | .alg e => exact evalOp_eq M none e s
  | .compute e => exact evalComputeOp_eq M e s
  | .opaque _ | .llrint _ => rfl
  | .isFinite p e =>
    simp only [evalExpr, QExpr.toReal, Structural.evalExpr, ← evalExpr_eq M e s]
    rfl
  | .fma p bc₁ bc₂ a b c =>
    simp only [evalExpr, QExpr.toReal, Structural.evalExpr,
      ← evalExpr_eq M a s, ← evalExpr_eq M b s, ← evalExpr_eq M c s]
    rfl
  | .fmaTyped d bc₁ bc₂ a b c =>
    simp only [evalExpr, QExpr.toReal, Structural.evalExpr,
      ← evalExpr_eq M a s, ← evalExpr_eq M b s, ← evalExpr_eq M c s]
    rfl
termination_by sizeOf e
decreasing_by all_goals simp_wf <;> omega

/-! Negative examples: changing the state (or the algebra) invalidates the
agreement. Each public theorem is exercised at its own entrypoint. -/

private def emptyState : State α where
  init := fun _ _ => .mk .nat 0
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => Structural.defaultValue d

private def withNat (n : Nat) : State α :=
  emptyState.setReg "x" .nat [] fun _ => n

private def withInt (n : Int) : State α :=
  emptyState.setReg "x" .int [] fun _ => n

example (M : CAlgebra ℚ) :
    boundaryZero { M with literal := fun _ _ q => q } none .real ≠
      Structural.boundaryZero { M with literal := fun _ _ q => q + 1 }.lift none .real := by
  simp [boundaryZero, Structural.boundaryZero, CAlgebra.lift_literal_zero]

example (M : CAlgebra α) :
    evalOp M none (.ref .nat [] "x") (withNat 0) ≠
      Structural.evalOp M.lift none (.ref .nat [] "x") (withNat 1).lift := by
  rw [evalOp_unfold, Structural.evalOp_unfold]
  intro h
  have := congrArg (fun v => v.map (· ())) h
  simp [withNat, State.getReg, State.setReg, State.lift, ComputeDType.eraseDType] at this

example (M : CAlgebra α) :
    evalOffsets M none [.ref .nat [] "x"] (withNat 0) ≠
      Structural.evalOffsets M.lift none [.ref .nat [] "x"] (withNat 1).lift := by
  simp only [evalOffsets, Structural.evalOffsets, evalOp_unfold, Structural.evalOp_unfold]
  simp [withNat, State.getReg, State.setReg, State.lift, ComputeDType.eraseDType]

example (M : CAlgebra α) :
    evalDeltas M none [.ref .int [] "x"] (withInt 0) ≠
      Structural.evalDeltas M.lift none [.ref .int [] "x"] (withInt 1).lift := by
  simp only [evalDeltas, Structural.evalDeltas, evalOp_unfold, Structural.evalOp_unfold]
  simp [withInt, State.getReg, State.setReg, State.lift]

example (M : CAlgebra α) :
    evalComputeOp M (.alg .uint32 (.ref .nat [] "x")) (withNat 0) ≠
      Structural.evalComputeOp M.lift (.alg .uint32 (.ref .nat [] "x")) (withNat 1).lift := by
  simp only [evalComputeOp, Structural.evalComputeOp, evalOp_unfold, Structural.evalOp_unfold]
  intro h
  have := congrArg (fun v => v.map (· ())) h
  simp [withNat, State.getReg, State.setReg, State.lift, ComputeDType.eraseDType] at this

example (M : CAlgebra α) :
    evalExpr M (.alg (.ref .nat [] "x")) (withNat 0) ≠
      Structural.evalExpr M.lift (.alg (.ref .nat [] "x")) (withNat 1).lift := by
  simp only [evalExpr, Structural.evalExpr, evalOp_unfold, Structural.evalOp_unfold]
  intro h
  have := congrArg (fun v => v.map (· ())) h
  simp [withNat, State.getReg, State.setReg, State.lift, ComputeDType.eraseDType] at this

end VeriTile.Interp
