/-
VeriTile.Staging.Dist.SpecLibSpec

Interface and statements of the specification library, frozen by ticket S7.1.
Signed on 2026-11-11: `allGather_apply` and `allReduce_eq_of_leaves_perm` with the
definitions under them. Everything else is UNSIGNED.

The library states what a collective computes, as a mathematical function or
relation over one tensor per rank. It is independent of any front end and of
any schedule. Tickets S7.2–S7.5 move each declaration to `VeriTile/Dist/Spec/`
under the same name and type, prove the theorems and delete the stubs here.

Ranks. A rank is an element of `Fin numRanks` for a parameter `numRanks : Nat`.
The location model (`VeriTile/Staging/Dist/LocationSpec.lean`, ticket S1.1)
keeps `Rank := Nat`. The bridge between the two is `Fin.val`. This file does
not import the location model.

Arithmetic. Every value is a real number. No floating-point type and no
rounding appears. Only `ReduceOrder.eval` is generic: it takes any carrier
with a zero and an addition, and every definition here calls it at `ℝ`.

Reduction order. A sum over ranks is the value of a `ReduceOrder`: a tree that
fixes which operands are added and in what nesting. The specifications use no
associativity or commutativity; `allReduce_eq_of_leaves_perm` is the only
statement that relates two orders, and it is a statement over `ℝ`.

MoE source. The MoE declarations follow the deduplicated layout
(`do_expand=False`) of the DeepEP port in TileScale, branch
`deepep-v2-intranode` at commit `c0f80cc5a5e386ae77075af9574701d3e0a99b00`,
directory `examples/distributed/deepep_v2/`.

Layout. An axis that holds one chunk of `n` rows per rank has `numRanks * n`
rows, and row `q * n + i` is row `i` of the chunk of rank `q`
(`finProdFinEquiv (q, i)`).

Decisions that the tickets left open:
 1. `RankTensor numRanks ι` is `Fin numRanks → ι → ℝ` for any index type `ι`;
    `TileIndex shape` of the Triton core is one instance.
 2. Issue #17: the reduction order is the tree with `zero`, `input` and `add`;
    a left fold over a list of operands is `ReduceOrder.ofList`.
    `ReduceOrder.eval` is generic in the carrier of the sum.
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
 9. Variable-length all-to-all, MoE dispatch and MoE store-back are relations
    between buffers, because they do not fix every row. Their buffers are
    indexed by a row in `Nat`; capacity is left to the kernel contract.
10. `AllToAllLayout.WellFormed` asks only that no receive row gets rows from
    two sources.
11. An MoE routing table gives each token a set of destination ranks, which
    may be empty, and one weight per destination. A token is sent once to each
    destination, also when it selects several experts there.
12. The row that a token gets at a destination is not in the routing table,
    because it depends on the schedule. A row assignment `MoERows` says which
    token each receive row holds; `MoERouting.Assigns` says which assignments
    a table allows; `MoEDispatch` takes the assignment as an argument.
    Which rows of a buffer are used, and their grouping by source rank, are
    not specified.
13. MoE combine is two steps: the relation `MoEStoreBack` and the sum
    `moeCombine`. Neither applies a weight; the weight is applied by the
    expert step, between dispatch and combine. The expanded layout, the bias
    of the combine and the gate weights that a dispatch delivers next to a
    row are not specified.
14. Ranks are `Fin numRanks` here and `Nat` in the location model.
15. An attention partial result is the softmax denominator and the normalized
    output of one query row, with the denominator not in logarithmic form.
16. Negative examples are concrete instances; S7.2–S7.5 choose them.
-/
import VeriTile.Triton.KernelLemmas.Matmul

namespace VeriTile.Dist

open VeriTile.Triton

/-! ## Rank-indexed tensors and reduction orders -/

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

/-- UNSIGNED. Owner: S7.2. The left fold from zero over a list of operands:
`ofList [a, b, c]` is `((0 + a) + b) + c`. -/
def ReduceOrder.ofList {ι : Type} (operands : List ι) : ReduceOrder ι :=
  operands.foldl (fun order i => .add order (.input i)) .zero

/-! ## Allgather and reduce-scatter (S7.2) -/

/-- Owner: S7.2. Each rank holds a chunk of `n` rows. Afterwards
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

/-- Owner: S7.2. On every rank, row `q * n + i` of the allgather
result is row `i` of the chunk of rank `q`. -/
theorem allGather_apply {numRanks n : Nat} {ι : Type} (x : RankTensor numRanks (Fin n × ι))
    (r q : Fin numRanks) (i : Fin n) (j : ι) :
    allGather x r (finProdFinEquiv (q, i), j) = x q (i, j) := sorry

/-! ## Allreduce (S7.3) -/

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
`numTokens` tokens. Token `t` of rank `s` is sent once to each rank in
`dest s t`: the ranks that own an expert the token selects. The set may be
empty. `weight s t d` is the sum of the gate weights of the experts on rank `d`
that the token selects. -/
structure MoERouting (numRanks numTokens : Nat) where
  dest : Fin numRanks → Fin numTokens → Finset (Fin numRanks)
  weight : Fin numRanks → Fin numTokens → Fin numRanks → ℝ

/-- UNSIGNED. Owner: S7.5. A row assignment of an MoE dispatch: which token
each row of each receive buffer holds. `rows d p = some (s, t)` says that row
`p` of rank `d` holds token `t` of rank `s`; `none` says that the row holds no
token. A row holds at most one token. The assignment is not fixed in advance:
it depends on the schedule. -/
abbrev MoERows (numRanks numTokens : Nat) : Type :=
  Fin numRanks → Nat → Option (Fin numRanks × Fin numTokens)

/-- UNSIGNED. Owner: S7.5. The row assignments that a routing table allows: a
token has exactly one row on each rank it is sent to, and no row on any other
rank. An assignment that holds one token in two rows of one rank is not
allowed. -/
structure MoERouting.Assigns {numRanks numTokens : Nat} (R : MoERouting numRanks numTokens)
    (rows : MoERows numRanks numTokens) : Prop where
  existsUnique_row : ∀ s t, ∀ d ∈ R.dest s t, ∃! p, rows d p = some (s, t)
  mem_dest : ∀ d p s t, rows d p = some (s, t) → d ∈ R.dest s t

/-- UNSIGNED. Owner: S7.5. MoE dispatch with the row assignment `rows`. The
routing table allows the assignment, and every row that holds a token holds
the values of that token. Rows that hold no token are not constrained. A
dispatch guarantees this for some `rows`, not for a given one. -/
def MoEDispatch {numRanks numTokens : Nat} {ι : Type} (R : MoERouting numRanks numTokens)
    (rows : MoERows numRanks numTokens) (tokens : RankTensor numRanks (Fin numTokens × ι))
    (recv : RankTensor numRanks (Nat × ι)) : Prop :=
  R.Assigns rows ∧ ∀ d p s t, rows d p = some (s, t) → ∀ j, recv d (p, j) = tokens s (t, j)

/-- UNSIGNED. Owner: S7.5. The first step of an MoE combine. Every rank
returns the result in each row that holds a token to the rank the token came
from: `back s d (t, j)` is what rank `d` returns for token `t` of rank `s`.
Entries for a rank that holds no row of the token are not constrained. -/
def MoEStoreBack {numRanks numTokens : Nat} {ι : Type} (rows : MoERows numRanks numTokens)
    (results : RankTensor numRanks (Nat × ι))
    (back : Fin numRanks → RankTensor numRanks (Fin numTokens × ι)) : Prop :=
  ∀ d p s t, rows d p = some (s, t) → ∀ j, back s d (t, j) = results d (p, j)

/-- UNSIGNED. Owner: S7.5. The second step of an MoE combine. Token `t` of
rank `s` becomes the sum of what the ranks returned for it, added in the order
`order s t`. The order names the ranks to add: those in `dest s t`. No weight
is applied here; a returned row already carries its weight. -/
noncomputable def moeCombine {numRanks numTokens : Nat} {ι : Type}
    (order : Fin numRanks → Fin numTokens → ReduceOrder (Fin numRanks))
    (back : Fin numRanks → RankTensor numRanks (Fin numTokens × ι)) :
    RankTensor numRanks (Fin numTokens × ι) :=
  fun s i => (order s i.1).eval fun d => back s d i

/-- UNSIGNED. Owner: S7.5. With identity experts, combine after dispatch gives
each token times its weights, added in the order of the combine, for every row
assignment of the dispatch. An identity expert returns a received row times
the weight of its token on that rank (`hexpert`). The order adds only ranks
that the token is sent to (`horder`). -/
theorem moeCombine_of_moeDispatch {numRanks numTokens : Nat} {ι : Type}
    (R : MoERouting numRanks numTokens) (rows : MoERows numRanks numTokens)
    (order : Fin numRanks → Fin numTokens → ReduceOrder (Fin numRanks))
    (tokens : RankTensor numRanks (Fin numTokens × ι))
    (recv results : RankTensor numRanks (Nat × ι))
    (back : Fin numRanks → RankTensor numRanks (Fin numTokens × ι))
    (hdispatch : MoEDispatch R rows tokens recv)
    (hexpert : ∀ d p s t, rows d p = some (s, t) → ∀ j,
      results d (p, j) = R.weight s t d * recv d (p, j))
    (hback : MoEStoreBack rows results back)
    (horder : ∀ s t, ∀ d ∈ (order s t).leaves, d ∈ R.dest s t) :
    moeCombine order back =
      fun s i => (order s i.1).eval fun d => R.weight s i.1 d * tokens s i := sorry

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
