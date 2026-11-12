/-
VeriTile.Dist.Access

Accessibility of the Dist line (ticket S1.4): which agent may read and write
which space. Frozen by ticket S1.1 in
`VeriTile/Staging/Dist/LocationSpec.lean` and moved here with proofs, under
the same names and types. Accessibility of a single cell, `AccessibleAt`,
depends on the symmetric heap and is in `VeriTile/Dist/SymmHeap.lean`.

That a rejected access is an execution failure is not stated here: it needs
the execution semantics.
-/
import VeriTile.Dist.Agent

namespace VeriTile.Dist

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

/-- Device memory is accessible from the agents of every rank. -/
theorem accessible_global (a : Agent) (r : Rank) : Accessible a (.global r) := by
  simp [Accessible]

/-- Shared memory is accessible exactly from its CTA and from the groups and
threads of that CTA. -/
theorem accessible_shared_iff {a : Agent} {c : Cta} :
    Accessible a (.shared c) ↔ a.cta? = some c := by
  simp [Accessible, Agent.within]

/-- Fragment tiles are accessible exactly from their CTA and from the groups
and threads of that CTA. -/
theorem accessible_fragment_iff {a : Agent} {c : Cta} :
    Accessible a (.fragment c) ↔ a.cta? = some c := by
  simp [Accessible, Agent.within]

/-- Local memory is accessible exactly from the thread that owns it. -/
theorem accessible_local_iff {a : Agent} {c : Cta} {t : Nat} :
    Accessible a (.local c t) ↔ ∃ g, a = .thread c g t := by
  cases a <;> simp [Accessible]

/-- Barrier objects are accessible exactly from their CTA and from the groups
and threads of that CTA. -/
theorem accessible_barrier_iff {a : Agent} {c : Cta} :
    Accessible a (.barrier c) ↔ a.cta? = some c := by
  simp [Accessible, Agent.within]

/-- Tensor memory is accessible exactly from its CTA and from the groups and
threads of that CTA. -/
theorem accessible_tmem_iff {a : Agent} {c : Cta} :
    Accessible a (.tmem c) ↔ a.cta? = some c := by
  simp [Accessible, Agent.within]

/-- Negative example of S1.4: a CTA may not access the shared memory of
another CTA, so the execution semantics turns such a read into a failure. -/
example : ¬ Accessible (.cta ⟨0, [0]⟩) (.shared ⟨0, [1]⟩) := by decide

end VeriTile.Dist
