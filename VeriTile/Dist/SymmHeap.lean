/-
VeriTile.Dist.SymmHeap

Symmetric heap of the Dist line (ticket S1.5): `symmAt` maps a cell of device
memory to the cell at the same region and offset on another rank, and
`AccessibleAt` says which cell an agent may access. Frozen by ticket S1.1 in
`VeriTile/Staging/Dist/LocationSpec.lean` and moved here with proofs, under
the same names and types.

`AccessibleAt` is declared after `symmAt` in this module, as in the staging
file: its elaborated body reuses the matcher `symmAt.match_1`, which is part of
its recorded fingerprint.

`Rank` is in `VeriTile/Dist/Space.lean`. The multi-rank state `DistState`
stays an unsigned stub in the staging file.
-/
import VeriTile.Dist.Access

namespace VeriTile.Dist

open VeriTile.Triton (RegionName)

/-! ## Symmetric heap -/

/-- `symmAt symm p r'` is the cell at the same region and offset as
`p` on rank `r'`. `symm` says which regions lie on the symmetric heap. The
result is `none` when `p` is not in device memory or its region is not
symmetric. -/
def symmAt (symm : RegionName → Bool) (p : Addr) (r' : Rank) : Option Addr :=
  match p.space with
  | .global _ => if symm p.region then some { p with space := .global r' } else none
  | _ => none

/-- `symmAt` succeeds exactly on a device-memory cell of a symmetric region,
and then keeps the region and the offset. -/
theorem symmAt_eq_some_iff {symm : RegionName → Bool} {p q : Addr} {r' : Rank} :
    symmAt symm p r' = some q ↔
      symm p.region = true ∧ (∃ r, p.space = .global r) ∧
        q = { space := .global r', region := p.region, offset := p.offset } := by
  obtain ⟨space, region, offset⟩ := p
  cases space <;> cases hs : symm region <;> simp [symmAt, hs, eq_comm]

/-- The symmetric counterpart of a cell on its own rank is the cell itself. -/
theorem symmAt_self {symm : RegionName → Bool} {p : Addr} {r : Rank}
    (hs : symm p.region = true) (hp : p.space = .global r) : symmAt symm p r = some p := by
  obtain ⟨space, region, offset⟩ := p
  subst hp
  simp_all [symmAt]

/-- Moving to rank `r'` and then to rank `r''` is moving to rank `r''`. -/
theorem symmAt_symmAt {symm : RegionName → Bool} {p q : Addr} {r' : Rank}
    (h : symmAt symm p r' = some q) (r'' : Rank) :
    symmAt symm q r'' = symmAt symm p r'' := by
  obtain ⟨hs, ⟨r, hp⟩, rfl⟩ := symmAt_eq_some_iff.mp h
  obtain ⟨space, region, offset⟩ := p
  subst hp
  simp_all [symmAt]

/-! ## Accessibility of a cell -/

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

/-- A cell in the device memory of another rank that is not on the symmetric
heap is not accessible. -/
theorem not_accessibleAt_of_remote_not_symm {symm : RegionName → Bool} {a : Agent} {p : Addr}
    {r : Rank} (hp : p.space = .global r) (hr : r ≠ a.rankOf) (hs : symm p.region = false) :
    ¬ AccessibleAt symm a p := by
  simp [AccessibleAt, hp, hr, hs]

/-- Every symmetric counterpart is accessible, from every agent. -/
theorem accessibleAt_of_symmAt {symm : RegionName → Bool} {a : Agent} {p q : Addr} {r' : Rank}
    (h : symmAt symm p r' = some q) : AccessibleAt symm a q := by
  obtain ⟨hs, -, rfl⟩ := symmAt_eq_some_iff.mp h
  exact ⟨accessible_global a r', Or.inr hs⟩

/-- Negative example of S1.5: a cell of device memory whose region is not on
the symmetric heap has no counterpart on another rank. -/
example (region : RegionName) : symmAt (fun _ => false) ⟨.global 0, region, 0⟩ 1 = none := rfl

/-- Negative example of S1.5: a cell of another rank outside the symmetric
heap is not accessible. -/
example (region : RegionName) :
    ¬ AccessibleAt (fun _ => false) (.rank 0) ⟨.global 1, region, 0⟩ :=
  not_accessibleAt_of_remote_not_symm rfl (by decide) rfl

end VeriTile.Dist
