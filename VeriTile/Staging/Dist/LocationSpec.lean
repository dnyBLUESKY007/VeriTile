/-
VeriTile.Staging.Dist.LocationSpec

Location and agent model of the Dist line, frozen by ticket S1.1. Signed on
2026-11-11, except `DistState`, its projections and its two statements, which stay
unsigned until ticket S1.5 designs the multi-rank state.

Ticket S1.2 moved its part to the main target with proofs: ranks, CTAs,
spaces, addresses and lifetimes are in `VeriTile/Dist/Space.lean`. What stays
here is agents and scopes (S1.3), accessibility (S1.4), the symmetric heap and
the multi-rank state (S1.5) and thread guards (S1.6).

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

open VeriTile.Triton (RegionName BlockState)

/-! ## Agents and scopes (S1.3) -/

/-- Who executes: a whole rank, one CTA, one role or warp group `g`
of a CTA, or one thread. In `thread c g t`, `t` is the thread id within CTA `c`
and `g` is the group the thread belongs to. -/
inductive Agent where
  | rank (r : Rank)
  | cta (c : Cta)
  | group (c : Cta) (g : Nat)
  | thread (c : Cta) (g : Nat) (t : Nat)
  deriving DecidableEq, Repr

/-- The rank an agent runs on. -/
def Agent.rankOf : Agent → Rank
  | .rank r => r
  | .cta c | .group c _ | .thread c _ _ => c.rank

/-- The CTA an agent is or lies in; `none` for a whole rank. -/
def Agent.cta? : Agent → Option Cta
  | .rank _ => none
  | .cta c | .group c _ | .thread c _ _ => some c

/-- `a.within b`: `a` is `b` or lies inside `b` in the hierarchy
rank → CTA → group → thread. -/
def Agent.within : Agent → Agent → Prop
  | a, .rank r => a.rankOf = r
  | a, .cta c => a.cta? = some c
  | .group c g, .group c' g' => c = c' ∧ g = g'
  | .thread c g _, .group c' g' => c = c' ∧ g = g'
  | _, .group _ _ => False
  | a, .thread c g t => a = .thread c g t

/-- `Agent.within` is decidable. -/
instance Agent.within.decidable (a b : Agent) : Decidable (a.within b) := by
  cases b <;> cases a <;> simp only [Agent.within] <;> infer_instance

/-- The scope of a synchronizing operation: one CTA, one GPU, or the
whole system. -/
inductive Scope where
  | cta
  | gpu
  | sys
  deriving DecidableEq, Repr

/-- Position of a scope in the order `cta < gpu < sys`. -/
def Scope.level : Scope → Nat
  | .cta => 0
  | .gpu => 1
  | .sys => 2

/-- The order `cta < gpu < sys`. -/
instance Scope.instLinearOrder : LinearOrder Scope :=
  LinearOrder.lift' Scope.level (by intro s s'; cases s <;> cases s' <;> simp [Scope.level])

/-- `s.covers a b`: one instance of scope `s` contains both agents.
`cta` needs both in the same CTA, `gpu` needs both on the same rank, `sys`
contains every pair. -/
def Scope.covers : Scope → Agent → Agent → Prop
  | .cta, a, b => a.cta? ≠ none ∧ a.cta? = b.cta?
  | .gpu, a, b => a.rankOf = b.rankOf
  | .sys, _, _ => True

/-- `Scope.covers` is decidable. -/
instance Scope.covers.decidable (s : Scope) (a b : Agent) : Decidable (s.covers a b) := by
  cases s <;> simp only [Scope.covers] <;> infer_instance

/-- Proved by S1.3. A wider scope covers every pair that a narrower
scope covers. -/
theorem Scope.covers_mono {s s' : Scope} {a b : Agent}
    (h : s ≤ s') (hc : s.covers a b) : s'.covers a b := sorry

/-- Proved by S1.3. Agents on different ranks are covered by `sys`
only. -/
theorem Scope.covers_ne_rank_eq_sys {s : Scope} {a b : Agent}
    (hc : s.covers a b) (hr : a.rankOf ≠ b.rankOf) : s = .sys := sorry

/-! ## Accessibility (S1.4) -/

/-- `Accessible a s`: agent `a` may read and write space `s`. The
execution semantics turns any other access into a failure. Device memory of
every rank passes here; whether a cell has a counterpart on another rank is
decided by `symmAt`. -/
def Accessible : Agent → Space → Prop
  | _, .global _ => True
  | a, .shared c | a, .fragment c | a, .barrier c | a, .tmem c => a.within (.cta c)
  | .thread c' _ t', .local c t => c' = c ∧ t' = t
  | _, .local _ _ => False

/-- `Accessible` is decidable. -/
instance Accessible.decidable (a : Agent) (s : Space) : Decidable (Accessible a s) := by
  cases s <;> cases a <;> simp only [Accessible] <;> infer_instance

/-- Proved by S1.4. Device memory is accessible from the agents of
every rank. -/
theorem accessible_global (a : Agent) (r : Rank) : Accessible a (.global r) := sorry

/-- Proved by S1.4. Shared memory is accessible exactly from its CTA
and from the groups and threads of that CTA. -/
theorem accessible_shared_iff {a : Agent} {c : Cta} :
    Accessible a (.shared c) ↔ a.cta? = some c := sorry

/-- Proved by S1.4. Fragment tiles are accessible exactly from their
CTA and from the groups and threads of that CTA. -/
theorem accessible_fragment_iff {a : Agent} {c : Cta} :
    Accessible a (.fragment c) ↔ a.cta? = some c := sorry

/-- Proved by S1.4. Local memory is accessible exactly from the
thread that owns it. -/
theorem accessible_local_iff {a : Agent} {c : Cta} {t : Nat} :
    Accessible a (.local c t) ↔ ∃ g, a = .thread c g t := sorry

/-- Proved by S1.4. Barrier objects are accessible exactly from their
CTA and from the groups and threads of that CTA. -/
theorem accessible_barrier_iff {a : Agent} {c : Cta} :
    Accessible a (.barrier c) ↔ a.cta? = some c := sorry

/-- Proved by S1.4. Tensor memory is accessible exactly from its CTA
and from the groups and threads of that CTA. -/
theorem accessible_tmem_iff {a : Agent} {c : Cta} :
    Accessible a (.tmem c) ↔ a.cta? = some c := sorry

/-! ## Symmetric heap and multi-rank state (S1.5) -/

/-- `symmAt symm p r'` is the cell at the same region and offset as
`p` on rank `r'`. `symm` says which regions lie on the symmetric heap. The
result is `none` when `p` is not in device memory or its region is not
symmetric. -/
def symmAt (symm : RegionName → Bool) (p : Addr) (r' : Rank) : Option Addr :=
  match p.space with
  | .global _ => if symm p.region then some { p with space := .global r' } else none
  | _ => none

/-- `AccessibleAt symm a p`: agent `a` may access the cell `p`. The
space of `p` must be accessible to `a`, and a cell in the device memory of
another rank must lie on the symmetric heap. -/
def AccessibleAt (symm : RegionName → Bool) (a : Agent) (p : Addr) : Prop :=
  Accessible a p.space ∧
    match p.space with
    | .global r => r = a.rankOf ∨ symm p.region = true
    | _ => True

/-- `AccessibleAt` is decidable. -/
instance AccessibleAt.decidable (symm : RegionName → Bool) (a : Agent) (p : Addr) :
    Decidable (AccessibleAt symm a p) := by
  unfold AccessibleAt
  split <;> infer_instance

/-- Proved by S1.5. A cell in the device memory of another rank that
is not on the symmetric heap is not accessible. -/
theorem not_accessibleAt_of_remote_not_symm {symm : RegionName → Bool} {a : Agent} {p : Addr}
    {r : Rank} (hp : p.space = .global r) (hr : r ≠ a.rankOf) (hs : symm p.region = false) :
    ¬ AccessibleAt symm a p := sorry

/-- Proved by S1.5. Every symmetric counterpart is accessible, from
every agent. -/
theorem accessibleAt_of_symmAt {symm : RegionName → Bool} {a : Agent} {p q : Addr} {r' : Rank}
    (h : symmAt symm p r' = some q) : AccessibleAt symm a q := sorry

/-- Proved by S1.5. The symmetric counterpart of a cell on its own
rank is the cell itself. -/
theorem symmAt_self {symm : RegionName → Bool} {p : Addr} {r : Rank}
    (hs : symm p.region = true) (hp : p.space = .global r) : symmAt symm p r = some p := sorry

/-- Proved by S1.5. Moving to rank `r'` and then to rank `r''` is
moving to rank `r''`. -/
theorem symmAt_symmAt {symm : RegionName → Bool} {p q : Addr} {r' : Rank}
    (h : symmAt symm p r' = some q) (r'' : Rank) :
    symmAt symm q r'' = symmAt symm p r'' := sorry

/-- Proved by S1.5. `symmAt` succeeds exactly on a device-memory cell
of a symmetric region, and then keeps the region and the offset. -/
theorem symmAt_eq_some_iff {symm : RegionName → Bool} {p q : Addr} {r' : Rank} :
    symmAt symm p r' = some q ↔
      symm p.region = true ∧ (∃ r, p.space = .global r) ∧
        q = { space := .global r', region := p.region, offset := p.offset } := sorry

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
