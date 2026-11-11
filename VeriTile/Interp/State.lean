/-
VeriTile.Interp.State

Executable state of the interpreter (ticket I2.1) and the function-valued
`FP.Structural.State` it represents.

Memory is a finite map of the cells written so far over a computable initial
memory; registers are a finite map from a name to one typed tile. `lift`
reads both through `read` and `getReg`, so it commutes with every operation
here by `funext` (`lift_setReg`, `lift_write`, `lift_scatter`).
-/
import Std.Data.HashMap
import VeriTile.Triton.Float.Structural

namespace VeriTile.Interp

open VeriTile.Triton
open VeriTile.Triton.FP

/-- The tile bound to a register name. A name holds one dtype and one shape at
a time, as after `Structural.State.setReg`. The tile stays a function: `lift`
then returns exactly the value that was assigned. -/
structure Reg (α : Type) where
  dtype : TileDType
  shape : TileShape
  values : Structural.Values α dtype shape

/-- Executable state: registers, finite-map memory of typed cells, program
ids, grid sizes and the undefined-value table. -/
structure State (α : Type) where
  /-- The cells written so far, by region and offset. -/
  cells : Std.HashMap (RegionName × Nat) (Structural.Cell α) := {}
  /-- Initial memory: the content of every cell absent from `cells`. -/
  init : RegionName → Nat → Structural.Cell α
  regs : Std.HashMap RegName (Reg α) := {}
  pids : Nat → Nat
  numPids : Nat → Nat
  undef : (dtype : TileDType) → RegionName → Nat → Structural.Value α dtype

namespace State

variable {α : Type}

/-- The cell at an address: `Structural.State.mem` of the lifted state. -/
def read (s : State α) (r : RegionName) (o : Nat) : Structural.Cell α :=
  s.cells.getD (r, o) (s.init r o)

/-- A register at a dtype and shape: `Structural.State.regs` of the lifted
state. A name bound at another dtype or shape reads as `none`. -/
def getReg (s : State α) (d : TileDType) (sh : TileShape) (n : RegName) :
    Option (Structural.Values α d sh) :=
  s.regs[n]?.bind fun b =>
    if hd : b.dtype = d then
      if hs : b.shape = sh then some (hd ▸ hs ▸ b.values) else none
    else none

def setReg (s : State α) (n : RegName) (d : TileDType) (sh : TileShape)
    (v : Structural.Values α d sh) : State α :=
  { s with regs := s.regs.insert n ⟨d, sh, v⟩ }

def write (s : State α) (r : RegionName) (o : Nat) (c : Structural.Cell α) : State α :=
  { s with cells := s.cells.insert (r, o) c }

/-- Masked scatter in list order, the fold of `Structural.store` and
`Structural.atomicAdd`: an active lane replaces the cell at its address by
`val` of the cell left there by the preceding lanes. -/
def scatter {ι : Type} (s : State α) (l : List ι) (active : ι → Bool)
    (addr : ι → RegionName × Nat) (val : ι → Structural.Cell α → Structural.Cell α) :
    State α :=
  l.foldl (fun t i =>
    if active i then t.write (addr i).1 (addr i).2 (val i (t.read (addr i).1 (addr i).2))
    else t) s

/-- The function-valued state that an executable state represents. -/
def lift (s : State α) : Structural.State α where
  mem := s.read
  regs := s.getReg
  pids := s.pids
  numPids := s.numPids
  undef := s.undef

theorem lift_setReg (s : State α) (n : RegName) (d : TileDType) (sh : TileShape)
    (v : Structural.Values α d sh) : (s.setReg n d sh v).lift = s.lift.setReg n d sh v := by
  unfold lift setReg Structural.State.setReg
  congr 1
  funext d' sh' n'
  by_cases h : n' = n
  · simp [getReg, h]
  · simp [getReg, Std.HashMap.getElem?_insert, h, Ne.symm h]

theorem lift_write (s : State α) (r : RegionName) (o : Nat) (c : Structural.Cell α) :
    (s.write r o c).lift = s.lift.write r o c := by
  unfold lift write Structural.State.write
  congr 1
  funext r' o'
  simp [read, Std.HashMap.getD_insert, eq_comm (a := r), eq_comm (a := o)]

theorem lift_scatter {ι : Type} (s : State α) (l : List ι) (active : ι → Bool)
    (addr : ι → RegionName × Nat) (val : ι → Structural.Cell α → Structural.Cell α) :
    (s.scatter l active addr val).lift = l.foldl (fun t i =>
      if active i then t.write (addr i).1 (addr i).2 (val i (t.mem (addr i).1 (addr i).2))
      else t) s.lift := by
  unfold scatter
  induction l generalizing s with
  | nil => rfl
  | cons i l ih => rw [List.foldl_cons, List.foldl_cons, ih, apply_ite lift, lift_write]; rfl

end State

/-! Round trips on the carrier `Nat`. -/

private def s0 : State Nat where
  init := fun _ _ => .mk .nat 0
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => Structural.defaultValue d

-- A written cell reads back; other cells keep the initial memory.
#guard ((s0.write "x" 3 (.mk .real 7)).read "x" 3).read .real = (7 : Nat)
#guard ((s0.write "x" 3 (.mk .real 7)).read "x" 4).read .nat = (0 : Nat)
#guard ((s0.write "x" 3 (.mk .real 7)).read "y" 3).read .nat = (0 : Nat)
-- The lifted state returns the same cell.
#guard ((s0.write "x" 3 (.mk .real 7)).lift.mem "x" 3).read .real = (7 : Nat)
-- A register reads back at its dtype and shape only, and the last binding wins.
#guard ((s0.setReg "a" .nat [] fun _ => 5).getReg .nat [] "a").map (· ()) = some (5 : Nat)
#guard ((s0.setReg "a" .nat [] fun _ => 5).getReg .int [] "a").isNone
#guard (((s0.setReg "a" .nat [] fun _ => 5).setReg "a" .int [] fun _ => -2).getReg .nat [] "a").isNone
-- Scatter skips inactive lanes and accumulates on a repeated address.
#guard ((s0.scatter [0, 1, 2, 3] (· != 1) (fun i => ("x", i % 2))
    (fun i c => .mk .nat (c.read .nat + 10 ^ i))).read "x" 0).read .nat = (101 : Nat)
#guard ((s0.scatter [0, 1, 2, 3] (· != 1) (fun i => ("x", i % 2))
    (fun i c => .mk .nat (c.read .nat + 10 ^ i))).read "x" 1).read .nat = (1000 : Nat)

end VeriTile.Interp
