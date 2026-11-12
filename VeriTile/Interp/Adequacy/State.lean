/-
VeriTile.Interp.Adequacy.State

State and enumeration lemmas for the adequacy proof (ticket I3.1): what
`interp_eq_structural_exec` needs about `State.lift`, the lane enumeration and
the helper twins. The agreement of each executable definition with its
`FP.Structural` original is stated in the module that defines it, or here:

- `State.read` and `Structural.State.mem`: `State.lift_mem` (here).
- `State.getReg` and `Structural.State.regs`: `State.lift_regs` (here).
- `State.pids`, `numPids`, `undef` and the upstream fields of the same name:
  `State.lift_pids`, `lift_numPids`, `lift_undef` (here).
- `State.setReg` and `Structural.State.setReg`: `State.lift_setReg`
  (`Interp.State`).
- `State.write` and `Structural.State.write`: `State.lift_write`
  (`Interp.State`).
- `State.scatter` and the folds of `Structural.store` and
  `Structural.atomicAdd`: `State.lift_scatter` (`Interp.State`) for a fold over
  any list, `State.store_lift_eq_scatter` and `State.atomicAdd_lift_eq_scatter`
  (here) for the folds upstream runs.
- `collectValues?` and `Structural.collectValues?`: `collectValues?_eq`
  (`Interp.Enum`).
- `numeric`, `numericLt`, `numericLe`, `numericEq` and the upstream helpers of
  the same name: `numeric_lift`, `numericLt_lift`, `numericLe_lift`,
  `numericEq_lift` (`Interp.Numeric`).
- `CAlgebra.literal` and the `literal` of `CAlgebra.lift`:
  `CAlgebra.lift_literal` (`Interp.Algebra`) on the image of a rational,
  `CAlgebra.lift_literal_zero` (here) on the real zero.

No agreement lemma is stated for:

- `Reg` and the fields `cells`, `init` and `regs` of the executable `State`:
  representation without an upstream counterpart; `lift` reads them only
  through `read` and `getReg`.
- `Structural.bop`, `integralFloorDiv`, `integralMod`, `natLt`, `natLe`: the
  interpreter uses the upstream definitions themselves, so there is no twin.
- The other fields of `CAlgebra.lift`: each is the field of the same name of
  the `CAlgebra` by `rfl`.
- `Structural.boundaryZero` and `Structural.atomicAddition`: their twins are
  not defined in `State`, `Enum`, `Numeric` or `Algebra` (tickets I2.3 and
  I2.4). `CAlgebra.lift_literal_zero` is the one fact their agreement needs
  that is not `rfl`.
-/
import VeriTile.Interp.Algebra
import VeriTile.Interp.State

namespace VeriTile.Interp

open VeriTile.Triton
open VeriTile.Triton.FP

/-- `Structural.boundaryZero` asks the lifted algebra for the literal of the
real zero, which is not syntactically the image of a rational. -/
theorem CAlgebra.lift_literal_zero {α : Type} (M : CAlgebra α) (p : Option ComputeDType)
    (d : FloatDType) : M.lift.literal p d 0 = M.literal p d 0 := by
  simpa using M.lift_literal p d 0

namespace State

variable {α : Type}

/-! ## Reads

Every field of the lifted state is the executable accessor of the same role. -/

@[simp] theorem lift_mem (s : State α) : s.lift.mem = s.read := rfl
@[simp] theorem lift_regs (s : State α) : s.lift.regs = s.getReg := rfl
@[simp] theorem lift_pids (s : State α) : s.lift.pids = s.pids := rfl
@[simp] theorem lift_numPids (s : State α) : s.lift.numPids = s.numPids := rfl
@[simp] theorem lift_undef (s : State α) : s.lift.undef = s.undef := rfl

/-! ## Scatter

`lift_scatter` is stated against a fold over an arbitrary list. The two
theorems below check it against the folds that upstream actually runs: every
successful `Structural.store` or `Structural.atomicAdd` from a lifted state
ends in the lift of a `scatter` over `TileShape.allIndices`, for each of the
three address forms and the three mask forms. -/

variable [Inhabited α] {dtype : TileDType} {shape : TileShape}

theorem store_lift_eq_scatter (M : Structural.Algebra α) (mem : MemAccess dtype shape)
    (mask : MaskOpt dtype shape) (values : Structural.Values α dtype shape) (s : State α)
    {t : Structural.State α} (h : Structural.store M mem mask values s.lift = some t) :
    ∃ active addr, t = (s.scatter (TileShape.allIndices shape) active addr
      fun i _ => .mk dtype (values i)).lift := by
  simp only [lift_scatter]
  cases mem with
  | blockPtr ptr checkedAxes =>
    simp only [Structural.store, Option.bind_eq_bind, Option.bind_eq_some_iff,
      Option.pure_def, Option.some.injEq] at h
    obtain ⟨ptrs, -, active, -, rfl⟩ := h
    exact ⟨fun i => active i && (ptrs i).inBounds (TileShape.indexToList shape i) checkedAxes,
      fun i => ((ptrs i).region, (ptrs i).address (TileShape.indexToList shape i)), rfl⟩
  | _ =>
    simp only [Structural.store, Option.bind_eq_bind, Option.bind_eq_some_iff,
      Option.pure_def, Option.some.injEq] at h
    obtain ⟨addresses, -, active, -, rfl⟩ := h
    exact ⟨active, addresses, rfl⟩

theorem atomicAdd_lift_eq_scatter (M : Structural.Algebra α) (nd : NumericDType dtype)
    (mem : MemAccess dtype shape) (mask : MaskOpt dtype shape)
    (values : Structural.Values α dtype shape) (s : State α)
    {t : Structural.State α} (h : Structural.atomicAdd M nd mem mask values s.lift = some t) :
    ∃ (combine : Structural.Value α dtype → Structural.Value α dtype → Structural.Value α dtype)
      (active : _) (addr : _), t = (s.scatter (TileShape.allIndices shape) active addr
        fun i c => .mk dtype (combine (c.read dtype) (values i))).lift := by
  simp only [lift_scatter]
  cases mem with
  | blockPtr ptr checkedAxes =>
    simp only [Structural.atomicAdd, Option.bind_eq_bind, Option.bind_eq_some_iff,
      Option.pure_def, Option.some.injEq] at h
    obtain ⟨combine, -, ptrs, -, active, -, rfl⟩ := h
    exact ⟨combine,
      fun i => active i && (ptrs i).inBounds (TileShape.indexToList shape i) checkedAxes,
      fun i => ((ptrs i).region, (ptrs i).address (TileShape.indexToList shape i)), rfl⟩
  | _ =>
    simp only [Structural.atomicAdd, Option.bind_eq_bind, Option.bind_eq_some_iff,
      Option.pure_def, Option.some.injEq] at h
    obtain ⟨combine, -, addresses, -, active, -, rfl⟩ := h
    exact ⟨combine, active, addresses, rfl⟩

end State

/-! ## Negative examples -/

private def s0 : State Nat where
  init := fun _ _ => .mk .nat 0
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => Structural.defaultValue d

/-- Negative example of `State.atomicAdd_lift_eq_scatter`: a scatter is not the
fold in which every lane reads the initial memory. Two lanes that add 1 at one
address leave 2, not 1. -/
example : ¬ ∀ (s : State Nat) (l : List Nat) (addr : Nat → RegionName × Nat)
    (val : Nat → Structural.Cell Nat → Structural.Cell Nat),
    (s.scatter l (fun _ => true) addr val).lift = l.foldl (fun t i =>
      t.write (addr i).1 (addr i).2 (val i (s.lift.mem (addr i).1 (addr i).2))) s.lift := by
  intro h
  have h := congrArg (fun t => (t.mem "x" 0).read .nat)
    (h s0 [0, 1] (fun _ => ("x", 0)) fun _ c => .mk .nat (c.read .nat + 1))
  rw [State.lift_scatter] at h
  simp [Structural.State.write, s0, State.read] at h

/-- Negative example of `State.store_lift_eq_scatter`: a scatter is not the
fold in the reverse lane order. Two lanes that store 0 and 1 at one address
leave 1, the value of the last lane. -/
example : ¬ ∀ (s : State Nat) (l : List Nat) (addr : Nat → RegionName × Nat)
    (val : Nat → Structural.Cell Nat),
    (s.scatter l (fun _ => true) addr fun i _ => val i).lift = l.reverse.foldl (fun t i =>
      t.write (addr i).1 (addr i).2 (val i)) s.lift := by
  intro h
  have h := congrArg (fun t => (t.mem "x" 0).read .nat)
    (h s0 [0, 1] (fun _ => ("x", 0)) fun i => .mk .nat i)
  rw [State.lift_scatter] at h
  simp [Structural.State.write] at h

end VeriTile.Interp
