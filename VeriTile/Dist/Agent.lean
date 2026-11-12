/-
VeriTile.Dist.Agent

Agent hierarchy and scope order of the Dist line (ticket S1.3): who executes
(rank → CTA → group → thread), `Agent.within`, the scopes `cta < gpu < sys`
and `Scope.covers`. Frozen by ticket S1.1 in
`VeriTile/Staging/Dist/LocationSpec.lean` and moved here with proofs, under
the same names and types.
-/
import VeriTile.Dist.Space

namespace VeriTile.Dist

/-! ## Agents and scopes -/

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

/-- A wider scope covers every pair that a narrower scope covers. -/
theorem Scope.covers_mono {s s' : Scope} {a b : Agent}
    (h : s ≤ s') (hc : s.covers a b) : s'.covers a b := by
  cases s <;> cases s' <;> cases a <;> cases b <;>
    first
      | exact absurd h (by decide)
      | simp_all [Scope.covers, Agent.cta?, Agent.rankOf]

/-- Agents on different ranks are covered by `sys` only. -/
theorem Scope.covers_ne_rank_eq_sys {s : Scope} {a b : Agent}
    (hc : s.covers a b) (hr : a.rankOf ≠ b.rankOf) : s = .sys := by
  cases s
  · cases a <;> cases b <;> simp_all [Scope.covers, Agent.cta?, Agent.rankOf]
  · exact absurd hc hr
  · rfl

/-- Negative example of S1.3: two ranks are not covered by `gpu`. -/
example : ¬ Scope.gpu.covers (.rank 0) (.rank 1) := by decide

end VeriTile.Dist
