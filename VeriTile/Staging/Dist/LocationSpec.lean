/-
VeriTile.Staging.Dist.LocationSpec

Location and agent model of the Dist line, frozen by ticket S1.1. Signed on
2026-11-11, except `DistState`, its projections and its two statements, which stay
unsigned until the multi-rank state is designed.

Tickets S1.2 to S1.5 moved their parts to the main target with proofs: ranks,
CTAs, spaces, addresses and lifetimes are in `VeriTile/Dist/Space.lean`;
agents and scopes are in `VeriTile/Dist/Agent.lean`; `Accessible` is in
`VeriTile/Dist/Access.lean`; `symmAt` and `AccessibleAt` are in
`VeriTile/Dist/SymmHeap.lean`. What stays here is the multi-rank state (S1.5)
and thread guards (S1.6).

Spaces tag the upstream `RegionName`; no `Stmt` or `Op` constructor is added.
The data types and the small functions over them are real definitions.
`DistState` and its projections are stubs. Each theorem is a statement with
`sorry`. The ticket named in a docstring moves that declaration to
`VeriTile/Dist/` under the same name and type, proves it and deletes it here in
the same change.

Decisions that the tickets left open:

- `Rank := Nat`; `numRanks` is a projection of `DistState`, not a constant or a bound in `Rank`. The specification library indexes ranks by `Fin numRanks`; `Fin.val` is the bridge.
- `Cta` is a rank and per-axis program ids, so `shared c` names the memory of one CTA of one rank.
- A location is `Addr` (space tag, region, offset); no table maps region names to spaces.
- `Space.local` is declared as `«local»`, because `local` is a Lean keyword.
- Lifetimes: `global` kernel; `shared`, `fragment`, `barrier`, `tmem` CTA; `local` thread.
- Liveness is relative to the running threads; a CTA is alive while one of its threads runs.
- A thread agent is `thread c g t`: its identity contains its group; `t` is its id in the CTA.
- `Scope.covers s a b` is a relation, not the function that returns the smallest scope.
- `cta` covers two agents only inside one CTA; a whole-rank agent is covered by `gpu` and `sys`.
- `Accessible a (global r)` holds from every rank; `AccessibleAt` adds that a cell of another rank must be on the symmetric heap.
- `symmAt` takes the set of symmetric regions as an argument and does not consult `numRanks`.
- `DistState` is opaque, because its synchronization-object part is defined by the event tickets.
- `DistState.block` takes a rank and program ids; `DistState.single` embeds an upstream state.
- A thread guard is a condition on the thread id; the CTA's thread count is an argument of `Runs`.
- Lemma names other than `covers_ne_rank_eq_sys`, `symmAt_self`, `symmAt_symmAt` are chosen here.

Not stated here, because both need the execution semantics: that a rejected
access is an execution failure, and that two threads storing to one address
are a data race.
-/
import VeriTile.Dist.Space

namespace VeriTile.Dist

open VeriTile.Triton (BlockState)

/-! ## Multi-rank state (S1.5) -/

/-- UNSIGNED stub (S1.5). The state of a multi-rank launch: one store per rank
and the state of the synchronization objects. -/
opaque DistState : Type

/-- UNSIGNED stub (S1.5). The number of ranks of the launch. -/
opaque DistState.numRanks : DistState → Nat

/-- UNSIGNED stub (S1.5). The upstream state that the CTA with program ids
`pid` on rank `r` executes on. -/
opaque DistState.block : DistState → Rank → List Nat → BlockState

/-- UNSIGNED stub (S1.5). The launch with one rank, no synchronization object,
and the upstream state `s` as the state of rank 0. -/
def DistState.single (s : BlockState) : DistState := sorry

/-- UNSIGNED. Proved by S1.5. The embedding of an upstream state has one
rank. -/
theorem DistState.numRanks_single (s : BlockState) : (DistState.single s).numRanks = 1 := sorry

/-- UNSIGNED. Proved by S1.5. A single-rank state projects to the upstream
state it was built from, unchanged, for every CTA. -/
theorem DistState.block_single (s : BlockState) (pid : List Nat) :
    (DistState.single s).block 0 pid = s := sorry

/-! ## Thread guards (S1.6) -/

/-- A guard on the thread id, such as `tid == 0`: `pass t` says
whether thread `t` of a CTA runs the guarded region. -/
structure ThreadGuard where
  pass : Nat → Bool

/-- No guard: every thread runs the region. -/
def ThreadGuard.all : ThreadGuard := ⟨fun _ => true⟩

/-- The guard `tid == k`. -/
def ThreadGuard.tidEq (k : Nat) : ThreadGuard := ⟨fun t => t == k⟩

/-- `g.Runs n t`: thread `t` of a CTA with `n` threads runs a region
guarded by `g`. -/
def ThreadGuard.Runs (g : ThreadGuard) (n t : Nat) : Prop :=
  t < n ∧ g.pass t = true

/-- The sequential view of thread `t` of one interval between two
`__syncthreads` points: the guarded regions of `body` that `t` runs, in program
order. -/
def ThreadGuard.view {α : Type} (body : List (ThreadGuard × α)) (t : Nat) : List α :=
  (body.filter fun region => region.1.pass t).map Prod.snd

/-- Proved by S1.6. A region guarded by `tid == k` is run by exactly
one thread of a CTA that has a thread `k`; `k = 0` is the guard `tid == 0`. -/
theorem ThreadGuard.existsUnique_runs_tidEq {n k : Nat} (h : k < n) :
    ∃! t, (ThreadGuard.tidEq k).Runs n t := sorry

end VeriTile.Dist
