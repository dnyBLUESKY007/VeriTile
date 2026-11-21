/-
Allreduce over real tensors (S7.3). The definition and signed statement move
unchanged from SpecLibSpec. Reordering uses the laws of real addition and
makes no claim about floating-point addition.
-/
import VeriTile.Dist.Spec.RankTensor
import Mathlib.Tactic.NormNum

namespace VeriTile.Dist

/-- Owner: S7.3. Every rank holds a tensor. Afterwards rank `r`
holds the sum over ranks; its element at index `i` is added in the order
`order r i`. -/
noncomputable def allReduce {numRanks : Nat} {ι : Type}
    (order : Fin numRanks → ι → ReduceOrder (Fin numRanks)) (x : RankTensor numRanks ι) :
    RankTensor numRanks ι :=
  fun r i => (order r i).eval fun q => x q i

/-- Owner: S7.3. Two orders that add the same operands equally often
give the same allreduce result over `ℝ`. -/
theorem allReduce_eq_of_leaves_perm {numRanks : Nat} {ι : Type}
    (order₁ order₂ : Fin numRanks → ι → ReduceOrder (Fin numRanks))
    (h : ∀ r i, (order₁ r i).leaves.Perm (order₂ r i).leaves) (x : RankTensor numRanks ι) :
    allReduce order₁ x = allReduce order₂ x := by
  funext r i
  change (order₁ r i).eval (fun q => x q i) = (order₂ r i).eval (fun q => x q i)
  rw [ReduceOrder.eval_eq_sum_leaves, ReduceOrder.eval_eq_sum_leaves]
  exact ((h r i).map (fun q => x q i)).sum_eq

namespace AllReduceExamples

private def leftOrder : ReduceOrder (Fin 3) :=
  .add (.add (.add .zero (.input 0)) (.input 1)) (.input 2)

private def rightOrder : ReduceOrder (Fin 3) :=
  .add (.input 0) (.add (.input 1) (.input 2))

/-- The trees have different nesting. -/
example : leftOrder ≠ rightOrder := by
  intro h
  cases h

/-- Both trees give the same real tensor, for every input tensor. -/
example (x : RankTensor 3 Unit) :
    allReduce (fun _ _ => leftOrder) x = allReduce (fun _ _ => rightOrder) x := by
  apply allReduce_eq_of_leaves_perm
  intro r i
  exact List.Perm.refl _

private def droppedOrder : ReduceOrder (Fin 3) :=
  .add (.input 0) (.input 1)

/-- Dropping an operand violates the theorem's permutation premise. -/
example : ¬ leftOrder.leaves.Perm droppedOrder.leaves := by
  intro h
  have hlen := h.length_eq
  norm_num [leftOrder, droppedOrder, ReduceOrder.leaves] at hlen

private def droppedTensor : RankTensor 3 Unit :=
  fun q _ => if q = 2 then 1 else 0

/-- Negative example: the dropped operand is nonzero, so equality fails. -/
example :
    allReduce (fun _ _ => leftOrder) droppedTensor ≠
      allReduce (fun _ _ => droppedOrder) droppedTensor := by
  intro h
  have hcell := congrFun (congrFun h 0) ()
  norm_num [allReduce, leftOrder, droppedOrder, droppedTensor, ReduceOrder.eval] at hcell

end AllReduceExamples
end VeriTile.Dist
