/-
Shared rank-indexed real tensors and reduction orders for the specification
library. The signed definitions move unchanged from SpecLibSpec for S7.3;
the generic evaluation lemma uses the laws of a commutative additive monoid.
-/
import Mathlib.Data.Real.Basic
import Mathlib.Algebra.BigOperators.Group.List.Basic

namespace VeriTile.Dist

/-- Owner: S7.2. One real tensor per rank: `x r i` is the element at
index `i` of the tensor of rank `r`. -/
abbrev RankTensor (numRanks : Nat) (ι : Type) : Type := Fin numRanks → ι → ℝ

/-- Owner: S7.2. The order of a sum over operands named by `ι`:
`zero` is the empty sum, `input i` is operand `i`, `add l r` adds the value of
`l` and the value of `r`. -/
inductive ReduceOrder (ι : Type) where
  | zero : ReduceOrder ι
  | input : ι → ReduceOrder ι
  | add : ReduceOrder ι → ReduceOrder ι → ReduceOrder ι

/-- Owner: S7.2. The value of a sum in the given order, with `f i`
as operand `i`, in any carrier `α` that has a zero and an addition. -/
def ReduceOrder.eval {ι α : Type} [Zero α] [Add α] (f : ι → α) : ReduceOrder ι → α
  | .zero => 0
  | .input i => f i
  | .add l r => l.eval f + r.eval f

/-- Owner: S7.2. The operands of an order, from left to right, each
as often as it occurs. -/
def ReduceOrder.leaves {ι : Type} : ReduceOrder ι → List ι
  | .zero => []
  | .input i => [i]
  | .add l r => l.leaves ++ r.leaves

/-- Evaluation is the sum of the leaf values whenever addition satisfies
the commutative monoid laws. -/
theorem ReduceOrder.eval_eq_sum_leaves {ι α : Type} [AddCommMonoid α]
    (f : ι → α) (order : ReduceOrder ι) :
    order.eval f = (order.leaves.map f).sum := by
  induction order with
  | zero => simp [eval, leaves]
  | input i => simp [eval, leaves]
  | add l r ihl ihr => simp [eval, leaves, ihl, ihr]

end VeriTile.Dist
