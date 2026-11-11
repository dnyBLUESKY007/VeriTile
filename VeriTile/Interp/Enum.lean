/-
VeriTile.Interp.Enum

Computable replacement for `FP.Structural.collectValues?` on the lanes of a
tile (ticket I2.2). Upstream decides "every lane is defined" by classical
choice; here the lanes are enumerated with `TileShape.allIndices`.
-/
import VeriTile.Triton.Float.Structural

namespace VeriTile.Interp

open VeriTile.Triton
open VeriTile.Triton.FP

/-- The tile of the values when every lane is defined, `none` otherwise. Each
read of the result evaluates `values` at that lane again. -/
def collectValues? {shape : TileShape} {β : Type} (values : TileIndex shape → Option β) :
    Option (TileIndex shape → β) :=
  if h : (TileShape.allIndices shape).all fun i => (values i).isSome then
    some fun i => (values i).get (List.all_eq_true.mp h i (TileShape.mem_allIndices shape i))
  else none

theorem collectValues?_eq {shape : TileShape} {β : Type} (values : TileIndex shape → Option β) :
    collectValues? values = Structural.collectValues? values := by
  unfold collectValues? Structural.collectValues?
  by_cases h : ∀ i, (values i).isSome
  · rw [dif_pos (by simpa using h), dif_pos h]
  · rw [dif_neg (by simpa using h), dif_neg h]

/-! Unit checks. -/

-- Every lane defined: the values, in index order.
#guard ((collectValues? (shape := [2, 3]) fun (i, j, _) => some (i.val * 3 + j.val)).map
    (TileShape.allIndices [2, 3]).map) = some [0, 1, 2, 3, 4, 5]
-- One undefined lane fails the whole tile.
#guard (collectValues? (shape := [2, 3]) fun (i, j, _) =>
    if i.val = 1 ∧ j.val = 2 then none else some 0).isNone
-- A tile without lanes succeeds.
#guard (collectValues? (shape := [0]) fun _ => (none : Option Nat)).isSome

end VeriTile.Interp
