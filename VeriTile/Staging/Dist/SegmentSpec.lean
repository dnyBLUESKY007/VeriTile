/-
VeriTile.Staging.Dist.SegmentSpec

Segment interface, slicing signature and bridge statements of the Dist line,
drafted by ticket S3.1. Nothing in this file is signed.

A kernel instance is sequential tensor code between synchronization or
communication points. A segment is one such piece of code, seen as a
transformer of the distributed state together with the cells it may read and
may write. Slicing cuts the body of a kernel instance at its events. The
bridge statements say that a segment made from upstream statements runs the
upstream `stepStmts` on the block state of its CTA and writes the result back.

`Segment`, `Segment.FootprintSound`, `DistState.AgreesOn` and
`DistState.SameLocal` are real definitions. `DistState.cell`,
`DistState.setBlock`, `Segment.ofStmts` and `slice` are stubs. Each theorem is
a statement with `sorry`. The ticket named in a docstring moves that
declaration to `VeriTile/Dist/` under the same name and type, proves it and
deletes it here in the same change.

The event type is a parameter `ε` of `slice` and of the statements about it,
because ticket S2.1 freezes `Event` at the same time. The body of a kernel
instance carries, at each cut, the event fired there.

Provisional choices, each open until the user signs (the pull request of
ticket S3.1 lists the options and their costs):

1. A segment is a partial function `DistState → Option DistState`; `none` is
   the failure of the upstream sequential execution. Access violations and
   data races are not failures of `run`; the execution semantics (S4) judges
   them from the footprints.
2. A footprint is a set of cells, `Set Addr`, not a set of regions.
3. A footprint is a function of the state in which the segment starts.
4. `FootprintSound` is two-sided: a frame clause for the write footprint and a
   determinacy clause for the read footprint. A written cell whose final
   content depends on its old content belongs to the read footprint too.
5. A segment records the CTA that runs it, because registers are not cells and
   `DistState.block` is keyed by rank and program ids.
6. Slicing consumes the body of one kernel instance as a flat list of upstream
   statements and events. Control flow that contains an event cannot be
   written in this form: it is unrolled or rejected before slicing.
7. Slicing is total and returns a flat list. A body with `n` events gives
   `n + 1` segments, some of which may hold no statement.
8. `DistState.cell` and `DistState.setBlock` are declared here because the
   statements need them; they belong to the multi-rank state of ticket S1.5.
-/
import VeriTile.Triton.Semantics.Step
import VeriTile.Staging.Dist.LocationSpec

namespace VeriTile.Dist

open VeriTile.Triton (BlockState MemCell Stmt Kernel stepStmts exec)

/-! ## State access that the statements need (stubs for S1.5) -/

/-- UNSIGNED stub (S1.5). The content of cell `p` in the distributed state. -/
noncomputable opaque DistState.cell : DistState → Addr → MemCell := fun _ _ => MemCell.real 0

/-- UNSIGNED stub (S1.5). `σ.setBlock r pid b` writes the upstream state `b`
back as the state of the CTA with program ids `pid` on rank `r`: the memory
that this CTA can reach and its registers. It is the inverse direction of
`DistState.block`. -/
opaque DistState.setBlock : DistState → Rank → List Nat → BlockState → DistState :=
  fun σ _ _ _ => σ

/-- UNSIGNED. The two states hold the same content in every cell of `P`. -/
def DistState.AgreesOn (P : Set Addr) (σ₁ σ₂ : DistState) : Prop :=
  ∀ p ∈ P, σ₁.cell p = σ₂.cell p

/-- UNSIGNED. The two states give CTA `c` the same state apart from memory:
the same registers, program ids, grid dimensions and unspecified values. -/
def DistState.SameLocal (c : Cta) (σ₁ σ₂ : DistState) : Prop :=
  { σ₁.block c.rank c.pid with mem := (σ₂.block c.rank c.pid).mem } = σ₂.block c.rank c.pid

/-! ## Segments and footprint soundness (S3.2) -/

/-- UNSIGNED. Owner: S3.2. Sequential code between two synchronization or
communication points. `cta` is the CTA that runs it. `run` is its effect on
the distributed state; `none` is a failed execution. `reads σ` and `writes σ`
are the cells it may read and may write when it starts in `σ`. -/
structure Segment where
  cta : Cta
  run : DistState → Option DistState
  reads : DistState → Set Addr
  writes : DistState → Set Addr

/-- UNSIGNED. Owner: S3.2. The footprints of a segment are sound.

`frame`: a run changes no cell outside the write footprint.

`confined`: a run changes the distributed state only by writing back the block
state of its own CTA, so it changes neither the registers of another CTA nor a
synchronization object.

`reads_determine`: two states that agree on the read footprint and give the
CTA the same local state have the same footprints, and if the run succeeds in
one it succeeds in the other, with results that agree on the write footprint
and on the local state. A written cell whose final content depends on its old
content therefore has to be in the read footprint too. -/
structure Segment.FootprintSound (seg : Segment) : Prop where
  frame : ∀ σ σ', seg.run σ = some σ' → ∀ p, p ∉ seg.writes σ → σ'.cell p = σ.cell p
  confined : ∀ σ σ', seg.run σ = some σ' → ∃ b, σ' = σ.setBlock seg.cta.rank seg.cta.pid b
  reads_determine : ∀ σ₁ σ₂, DistState.AgreesOn (seg.reads σ₁) σ₁ σ₂ →
    DistState.SameLocal seg.cta σ₁ σ₂ →
      seg.reads σ₂ = seg.reads σ₁ ∧ seg.writes σ₂ = seg.writes σ₁ ∧
        ∀ σ₁', seg.run σ₁ = some σ₁' →
          ∃ σ₂', seg.run σ₂ = some σ₂' ∧ DistState.AgreesOn (seg.writes σ₁) σ₁' σ₂' ∧
            DistState.SameLocal seg.cta σ₁' σ₂'

/-- UNSIGNED. Proved by S3.2. The frame lemma: a segment with sound footprints
leaves every cell outside its write footprint as it was. -/
theorem Segment.frame {seg : Segment} (h : seg.FootprintSound) {σ σ' : DistState}
    (hrun : seg.run σ = some σ') {p : Addr} (hp : p ∉ seg.writes σ) :
    σ'.cell p = σ.cell p := sorry

/-! ## Segments of upstream statements and slicing (S3.3) -/

/-- UNSIGNED stub (S3.3). The segment that runs the upstream statements `code`
on CTA `c`, with the footprints computed from the statements. -/
opaque Segment.ofStmts (c : Cta) (code : List Stmt) : Segment :=
  ⟨c, fun _ => none, fun _ => ∅, fun _ => ∅⟩

/-- UNSIGNED. Proved by S3.3. A segment made from statements is run by the CTA
it was made for. -/
theorem Segment.ofStmts_cta (c : Cta) (code : List Stmt) :
    (Segment.ofStmts c code).cta = c := sorry

/-- UNSIGNED. Proved by S3.3. The bridge for one segment: running it on the
distributed state is running the upstream `stepStmts` of its statements on the
block state of its CTA and writing the result back. It fails exactly when the
upstream execution fails. -/
theorem Segment.ofStmts_run (c : Cta) (code : List Stmt) (σ : DistState) :
    (Segment.ofStmts c code).run σ =
      (stepStmts code (σ.block c.rank c.pid)).map (σ.setBlock c.rank c.pid) := sorry

/-- UNSIGNED. Proved by S3.3. The footprints computed from the statements are
sound. -/
theorem Segment.ofStmts_footprintSound (c : Cta) (code : List Stmt) :
    (Segment.ofStmts c code).FootprintSound := sorry

/-- UNSIGNED stub (S3.3). Slicing. `body` is the body of the kernel instance
that CTA `c` runs: upstream statements and, at each synchronization or
communication point, the event `e : ε` fired there. The result lists, in
program order, the segments between the events and the events. -/
opaque slice {ε : Type} (c : Cta) (body : List (Stmt ⊕ ε)) : List (Segment ⊕ ε)

/-- UNSIGNED. Proved by S3.3. A body with no event is one segment. -/
theorem slice_noSync {ε : Type} (c : Cta) (code : List Stmt) :
    slice (ε := ε) c (code.map Sum.inl) = [Sum.inl (Segment.ofStmts c code)] := sorry

/-- UNSIGNED. Proved by S3.3. Slicing cuts at the first event: the statements
before it are one segment, and the rest of the body is sliced in the same
way. -/
theorem slice_sync {ε : Type} (c : Cta) (code : List Stmt) (e : ε) (rest : List (Stmt ⊕ ε)) :
    slice c (code.map Sum.inl ++ Sum.inr e :: rest) =
      Sum.inl (Segment.ofStmts c code) :: Sum.inr e :: slice c rest := sorry

/-! ## Bridge to the upstream execution (S3.4) -/

/-- UNSIGNED. Proved by S3.4. A kernel with no synchronization point is one
segment, and running that segment is running the upstream `exec` on the block
state of the CTA and writing the result back. `body` is the body of the
instance that CTA `c` runs; `hbody` says that its statements are those of the
upstream kernel `k`, and `hsync` that it holds no event. -/
theorem exec_noSync_eq_exec {ε : Type} (c : Cta) (k : Kernel) (body : List (Stmt ⊕ ε))
    (hbody : body.filterMap Sum.getLeft? = k.body) (hsync : ∀ e, Sum.inr e ∉ body)
    (σ : DistState) :
    ∃ seg : Segment, slice c body = [Sum.inl seg] ∧
      seg.run σ = (exec k (σ.block c.rank c.pid)).map (σ.setBlock c.rank c.pid) := sorry

end VeriTile.Dist
