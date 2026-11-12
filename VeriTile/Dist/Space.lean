/-
VeriTile.Dist.Space

Spaces and lifetimes of the Dist line (ticket S1.2): ranks, CTAs, the space
tag of a region, addresses, and when a space is live. Frozen by ticket S1.1 in
`VeriTile/Staging/Dist/LocationSpec.lean` and moved here with proofs, under
the same names and types.

Spaces tag the upstream `RegionName`; no `Stmt` or `Op` constructor is added.
`tmem` has its tag only; its operations come with the tensor-memory tickets.
-/
import VeriTile.Triton.Semantics.State

namespace VeriTile.Dist

open VeriTile.Triton (RegionName)

/-! ## Ranks and CTAs -/

/-- A rank: one GPU of the launch. -/
abbrev Rank := Nat

/-- One program instance (one CUDA thread block) on one rank. `pid`
lists its program id on each axis of the launch grid, outermost axis first. -/
structure Cta where
  rank : Rank
  pid : List Nat
  deriving DecidableEq, Repr

/-! ## Spaces and lifetimes -/

/-- The memory that holds a region. `global r` is device memory of
rank `r`. `shared c`, `fragment c`, `barrier c` and `tmem c` belong to CTA `c`:
its shared memory, its fragment tiles, its barrier objects and its tensor
memory. `local c t` belongs to thread `t` of CTA `c`. -/
inductive Space where
  | global (r : Rank)
  | shared (c : Cta)
  | fragment (c : Cta)
  | «local» (c : Cta) (t : Nat)
  | barrier (c : Cta)
  | tmem (c : Cta)
  deriving DecidableEq, Repr

/-- A cell of an upstream region, tagged with the space that holds
it. Two addresses with different space tags are different cells. -/
structure Addr where
  space : Space
  region : RegionName
  offset : Nat
  deriving DecidableEq

/-- How long a space exists: for the whole kernel launch, for the
run of one CTA, or for the run of one thread. -/
inductive Lifetime where
  | kernel
  | cta
  | thread
  deriving DecidableEq, Repr

/-- The lifetime of each space. -/
def Space.lifetime : Space → Lifetime
  | .global _ => .kernel
  | .shared _ | .fragment _ | .barrier _ | .tmem _ => .cta
  | .local _ _ => .thread

/-- The CTA that a space belongs to; `none` for device memory. -/
def Space.cta? : Space → Option Cta
  | .global _ => none
  | .shared c | .fragment c | .barrier c | .tmem c | .local c _ => some c

/-- The space exists. `running c t` says that thread `t` of CTA `c`
has started and has not returned. Device memory always exists; the memory of a
CTA exists while one of its threads runs; the memory of a thread exists while
that thread runs. The execution semantics rejects an access to a space that is
not live. -/
def Space.live (running : Cta → Nat → Prop) : Space → Prop
  | .global _ => True
  | .shared c | .fragment c | .barrier c | .tmem c => ∃ t, running c t
  | .local c t => running c t

/-- A space with kernel lifetime is live whichever threads run. -/
theorem Space.live_of_lifetime_kernel {running : Cta → Nat → Prop} {s : Space}
    (h : s.lifetime = .kernel) : s.live running := by
  cases s <;> simp_all [Space.lifetime, Space.live]

/-- Once no thread of a CTA runs, none of its spaces is live: its shared
memory, fragments, barriers, tensor memory and the local memory of its threads. -/
theorem Space.not_live_of_cta_ended {running : Cta → Nat → Prop} {s : Space} {c : Cta}
    (hs : s.cta? = some c) (hend : ∀ t, ¬ running c t) : ¬ s.live running := by
  cases s <;> simp_all [Space.cta?, Space.live]

/-- The local memory of a thread that does not run is not live, even while
other threads of its CTA run. -/
theorem Space.not_live_local_of_thread_ended {running : Cta → Nat → Prop} {c : Cta} {t : Nat}
    (hend : ¬ running c t) : ¬ (Space.local c t).live running := by
  simpa [Space.live] using hend

/-- Negative example of S1.2: the shared memory of a CTA that has ended is not
live while another CTA still runs, so a read of it is rejected. -/
example : ¬ (Space.shared ⟨0, [0]⟩).live (fun c _ => c = ⟨0, [1]⟩) :=
  Space.not_live_of_cta_ended rfl (fun _ => by decide)

end VeriTile.Dist
