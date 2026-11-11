/-
VeriTile.Staging.Dist.SpecLibSpec

Interface and statements of the specification library, frozen by ticket S7.1.
UNSIGNED.

The library states what a collective computes, as a mathematical function or
relation over one tensor per rank. It is independent of any front end and of
any schedule. Tickets S7.2–S7.5 move each declaration to `VeriTile/Dist/Spec/`
under the same name and type, prove the theorems and delete the stubs here.

Ranks. A rank is an element of `Fin numRanks` for a parameter `numRanks : Nat`.
The `Rank` type of the location model (ticket S1.1) is drafted in parallel and
is not used here; the two drafts are joined at review.

Arithmetic. Every value is a real number. No floating-point type and no
rounding appears.

Reduction order. A sum over ranks is the value of a `ReduceOrder`: a tree that
fixes which operands are added and in what nesting. The specifications use no
associativity or commutativity; `allReduce_eq_of_leaves_perm` is the only
statement that relates two orders, and it is proved over `ℝ`.

Layout. An axis that holds one chunk of `n` rows per rank has `numRanks * n`
rows, and row `q * n + i` is row `i` of the chunk of rank `q`
(`finProdFinEquiv (q, i)`).

Decisions that the tickets left open:
 1. `RankTensor numRanks ι` is `Fin numRanks → ι → ℝ` for any index type `ι`;
    `TileIndex shape` of the Triton core is one instance.
 2. Issue #17: the reduction order is the tree with `zero`, `input` and `add`;
    a left fold over a list of operands is `ReduceOrder.ofList`.
 3. The order of a reduction may differ per destination rank and per element.
 4. Every reduction is a sum.
 5. `ReduceOrder` belongs to S7.2, the first ticket that needs it in a main
    target, not to S7.3.
 6. Chunks have the same number of rows on every rank.
 7. `allGatherGemm` and `gemmReduceScatter` take the upstream `gemmSum` as the
    matrix product, whose inner sum has no order, as S7.2 asks; the sum over
    ranks in `gemmReduceScatter` keeps its order.
 8. In `allGatherGemm` the left operand is gathered and the right operand is
    local to each rank; in `gemmReduceScatter` both operands are local.
 9. Variable-length all-to-all and MoE dispatch are relations between send and
    receive buffers, because they fix only the rows that a table names. Their
    buffers are indexed by a row in `Nat`; capacity is left to the kernel
    contract.
10. `AllToAllLayout.WellFormed` asks only that no receive row gets rows from
    two sources.
11. An MoE token has a fixed number of copies; each copy goes to one rank and
    one row. The weights are an argument of `moeCombine`, not of the table.
12. An attention partial result is the softmax denominator and the normalized
    output of one query row, with the denominator not in logarithmic form.
13. Negative examples are concrete instances; S7.2–S7.5 choose them.
-/
import VeriTile.Triton.KernelLemmas.Matmul

namespace VeriTile.Dist

open VeriTile.Triton

/-! ## Rank-indexed tensors and reduction orders -/

/-- UNSIGNED. Owner: S7.2. One real tensor per rank: `x r i` is the element at
index `i` of the tensor of rank `r`. -/
abbrev RankTensor (numRanks : Nat) (ι : Type) : Type := Fin numRanks → ι → ℝ

/-- UNSIGNED. Owner: S7.2. The order of a sum over operands named by `ι`:
`zero` is the empty sum, `input i` is operand `i`, `add l r` adds the value of
`l` and the value of `r`. -/
inductive ReduceOrder (ι : Type) where
  | zero : ReduceOrder ι
  | input : ι → ReduceOrder ι
  | add : ReduceOrder ι → ReduceOrder ι → ReduceOrder ι

/-- UNSIGNED. Owner: S7.2. The value of a sum in the given order, with `f i`
as operand `i`. -/
noncomputable def ReduceOrder.eval {ι : Type} (f : ι → ℝ) : ReduceOrder ι → ℝ
  | .zero => 0
  | .input i => f i
  | .add l r => l.eval f + r.eval f

/-- UNSIGNED. Owner: S7.2. The operands of an order, from left to right, each
as often as it occurs. -/
def ReduceOrder.leaves {ι : Type} : ReduceOrder ι → List ι
  | .zero => []
  | .input i => [i]
  | .add l r => l.leaves ++ r.leaves

/-- UNSIGNED. Owner: S7.2. The left fold from zero over a list of operands:
`ofList [a, b, c]` is `((0 + a) + b) + c`. -/
def ReduceOrder.ofList {ι : Type} (operands : List ι) : ReduceOrder ι :=
  operands.foldl (fun order i => .add order (.input i)) .zero

/-! ## Allgather and reduce-scatter (S7.2) -/

/-- UNSIGNED. Owner: S7.2. Each rank holds a chunk of `n` rows. Afterwards
every rank holds all chunks in rank order. -/
def allGather {numRanks n : Nat} {ι : Type} (x : RankTensor numRanks (Fin n × ι)) :
    RankTensor numRanks (Fin (numRanks * n) × ι) :=
  fun _ i => x i.1.divNat (i.1.modNat, i.2)

/-- UNSIGNED. Owner: S7.2. Each rank holds a full tensor of `numRanks * n`
rows. Afterwards rank `r` holds chunk `r` of the sum over ranks; the element at
index `i` of that chunk is added in the order `order r i`. -/
noncomputable def reduceScatter {numRanks n : Nat} {ι : Type}
    (order : Fin numRanks → Fin n × ι → ReduceOrder (Fin numRanks))
    (x : RankTensor numRanks (Fin (numRanks * n) × ι)) :
    RankTensor numRanks (Fin n × ι) :=
  fun r i => (order r i).eval fun q => x q (finProdFinEquiv (r, i.1), i.2)

/-- UNSIGNED. Owner: S7.2. The matrix product of the upstream matmul
specification `gemmSum`, on tensors: entry `(i, j)` is `∑ k, A i k * B k j`. -/
noncomputable def gemm {M K N : Nat} (A : TileIndex [M, K] → ℝ) (B : TileIndex [K, N] → ℝ) :
    TileIndex [M, N] → ℝ :=
  fun i =>
    gemmSum (fun k => if h : k < K then A (i.1, ⟨k, h⟩, PUnit.unit) else 0)
      (fun k => if h : k < K then B (⟨k, h⟩, i.2.1, PUnit.unit) else 0) K

/-- UNSIGNED. Owner: S7.2. Allgather followed by a matrix product: rank `r`
holds the product of the gathered `A` with its own `B r`. -/
noncomputable def allGatherGemm {numRanks m K N : Nat}
    (A : RankTensor numRanks (TileIndex [m, K])) (B : RankTensor numRanks (TileIndex [K, N])) :
    RankTensor numRanks (TileIndex [numRanks * m, N]) :=
  fun r => gemm (allGather A r) (B r)

/-- UNSIGNED. Owner: S7.2. A matrix product followed by reduce-scatter: rank
`r` holds chunk `r` of the sum over ranks `q` of the products `A q * B q`. -/
noncomputable def gemmReduceScatter {numRanks m K N : Nat}
    (order : Fin numRanks → TileIndex [m, N] → ReduceOrder (Fin numRanks))
    (A : RankTensor numRanks (TileIndex [numRanks * m, K]))
    (B : RankTensor numRanks (TileIndex [K, N])) :
    RankTensor numRanks (TileIndex [m, N]) :=
  reduceScatter order fun q => gemm (A q) (B q)

/-- UNSIGNED. Owner: S7.2. On every rank, row `q * n + i` of the allgather
result is row `i` of the chunk of rank `q`. -/
theorem allGather_apply {numRanks n : Nat} {ι : Type} (x : RankTensor numRanks (Fin n × ι))
    (r q : Fin numRanks) (i : Fin n) (j : ι) :
    allGather x r (finProdFinEquiv (q, i), j) = x q (i, j) := sorry

/-! ## Allreduce (S7.3) -/

/-- UNSIGNED. Owner: S7.3. Every rank holds a tensor. Afterwards rank `r`
holds the sum over ranks; its element at index `i` is added in the order
`order r i`. -/
noncomputable def allReduce {numRanks : Nat} {ι : Type}
    (order : Fin numRanks → ι → ReduceOrder (Fin numRanks)) (x : RankTensor numRanks ι) :
    RankTensor numRanks ι :=
  fun r i => (order r i).eval fun q => x q i

/-- UNSIGNED. Owner: S7.3. Two orders that add the same operands equally often
give the same allreduce result over `ℝ`. -/
theorem allReduce_eq_of_leaves_perm {numRanks : Nat} {ι : Type}
    (order₁ order₂ : Fin numRanks → ι → ReduceOrder (Fin numRanks))
    (h : ∀ r i, (order₁ r i).leaves.Perm (order₂ r i).leaves) (x : RankTensor numRanks ι) :
    allReduce order₁ x = allReduce order₂ x := sorry

/-! ## All-to-all (S7.4) -/

/-- UNSIGNED. Owner: S7.4. Fixed-size all-to-all. The buffer of each rank has
one chunk of `n` rows per peer. Afterwards chunk `s` of rank `d` is what chunk
`d` of rank `s` was. -/
def allToAll {numRanks n : Nat} {ι : Type} (x : RankTensor numRanks (Fin (numRanks * n) × ι)) :
    RankTensor numRanks (Fin (numRanks * n) × ι) :=
  fun d i => x i.1.divNat (finProdFinEquiv (d, i.1.modNat), i.2)

/-- UNSIGNED. Owner: S7.4. The tables of a variable-length all-to-all.
`count s d` is the number of rows that rank `s` sends to rank `d`;
`sendOffset s d` is the first of these rows in the send buffer of `s`;
`recvOffset d s` is the first row in the receive buffer of `d` that they
fill. -/
structure AllToAllLayout (numRanks : Nat) where
  count : Fin numRanks → Fin numRanks → Nat
  sendOffset : Fin numRanks → Fin numRanks → Nat
  recvOffset : Fin numRanks → Fin numRanks → Nat

/-- UNSIGNED. Owner: S7.4. Row `p` of the receive buffer of rank `d` is filled
from rank `s`. -/
def AllToAllLayout.Receives {numRanks : Nat} (L : AllToAllLayout numRanks)
    (d s : Fin numRanks) (p : Nat) : Prop :=
  L.recvOffset d s ≤ p ∧ p < L.recvOffset d s + L.count s d

/-- UNSIGNED. Owner: S7.4. The precondition of a variable-length all-to-all:
no row of a receive buffer is filled from two sources. -/
def AllToAllLayout.WellFormed {numRanks : Nat} (L : AllToAllLayout numRanks) : Prop :=
  ∀ d s₁ s₂ p, L.Receives d s₁ p → L.Receives d s₂ p → s₁ = s₂

/-- UNSIGNED. Owner: S7.4. Variable-length all-to-all. Row `i` of the rows
that rank `s` sends to rank `d` arrives at row `recvOffset d s + i` of the
receive buffer of `d`. Rows that no source fills are not constrained. -/
def AllToAllV {numRanks : Nat} {ι : Type} (L : AllToAllLayout numRanks)
    (send recv : RankTensor numRanks (Nat × ι)) : Prop :=
  ∀ s d i, i < L.count s d → ∀ j,
    recv d (L.recvOffset d s + i, j) = send s (L.sendOffset s d + i, j)

/-! ## MoE dispatch and combine, attention chunk merge (S7.5) -/

/-- UNSIGNED. Owner: S7.5. The routing table of an MoE dispatch. Each rank has
`numTokens` tokens and each token has `numCopies` copies. Copy `k` of token `t`
of rank `s` goes to row `row s t k` of the buffer of rank `dest s t k`. -/
structure MoERouting (numRanks numTokens numCopies : Nat) where
  dest : Fin numRanks → Fin numTokens → Fin numCopies → Fin numRanks
  row : Fin numRanks → Fin numTokens → Fin numCopies → Nat

/-- UNSIGNED. Owner: S7.5. The precondition of an MoE dispatch: no token is
sent to one rank twice, and no row of a rank receives two copies. -/
structure MoERouting.WellFormed {numRanks numTokens numCopies : Nat}
    (R : MoERouting numRanks numTokens numCopies) : Prop where
  dest_injective : ∀ s t, Function.Injective (R.dest s t)
  slot_injective : Function.Injective fun c : Fin numRanks × Fin numTokens × Fin numCopies =>
    (R.dest c.1 c.2.1 c.2.2, R.row c.1 c.2.1 c.2.2)

/-- UNSIGNED. Owner: S7.5. MoE dispatch. Every copy of a token arrives at the
row and rank that the routing table names. Other rows are not constrained. -/
def MoEDispatch {numRanks numTokens numCopies : Nat} {ι : Type}
    (R : MoERouting numRanks numTokens numCopies)
    (tokens : RankTensor numRanks (Fin numTokens × ι)) (recv : RankTensor numRanks (Nat × ι)) :
    Prop :=
  ∀ s t k j, recv (R.dest s t k) (R.row s t k, j) = tokens s (t, j)

/-- UNSIGNED. Owner: S7.5. MoE combine. Token `t` of rank `s` becomes the
weighted sum of its copies, read back from the ranks that hold them; copy `k`
has weight `weight s t k` and the copies are added in the order `order s t`. -/
noncomputable def moeCombine {numRanks numTokens numCopies : Nat} {ι : Type}
    (R : MoERouting numRanks numTokens numCopies)
    (weight : Fin numRanks → Fin numTokens → Fin numCopies → ℝ)
    (order : Fin numRanks → Fin numTokens → ReduceOrder (Fin numCopies))
    (results : RankTensor numRanks (Nat × ι)) : RankTensor numRanks (Fin numTokens × ι) :=
  fun s i => (order s i.1).eval fun k =>
    weight s i.1 k * results (R.dest s i.1 k) (R.row s i.1 k, i.2)

/-- UNSIGNED. Owner: S7.5. With identity experts, combine after dispatch gives
each token times its weights, added in the order of the combine. -/
theorem moeCombine_of_moeDispatch {numRanks numTokens numCopies : Nat} {ι : Type}
    (R : MoERouting numRanks numTokens numCopies)
    (weight : Fin numRanks → Fin numTokens → Fin numCopies → ℝ)
    (order : Fin numRanks → Fin numTokens → ReduceOrder (Fin numCopies))
    (tokens : RankTensor numRanks (Fin numTokens × ι)) (recv : RankTensor numRanks (Nat × ι))
    (h : MoEDispatch R tokens recv) :
    moeCombine R weight order recv =
      fun s i => (order s i.1).eval fun k => weight s i.1 k * tokens s i := sorry

/-- UNSIGNED. Owner: S7.5. The attention result of one query row over one
chunk of keys: `weight` is the softmax denominator, the sum of the
exponentiated scores of the chunk; `out` is the output row, the average of the
chunk's values with these weights. -/
structure AttentionPartial (ι : Type) where
  weight : ℝ
  out : ι → ℝ

/-- UNSIGNED. Owner: S7.5. The merge of the partial results of
`numChunks` chunks: the denominators are added, and the outputs are averaged
with the denominators as weights. Both sums use the order `order`. -/
noncomputable def attentionMerge {numChunks : Nat} {ι : Type}
    (order : ReduceOrder (Fin numChunks)) (parts : Fin numChunks → AttentionPartial ι) :
    AttentionPartial ι where
  weight := order.eval fun c => (parts c).weight
  out := fun d =>
    (order.eval fun c => (parts c).weight * (parts c).out d) /
      order.eval fun c => (parts c).weight

end VeriTile.Dist
