/-
VeriTile.Interp.Bridge.RealAlgebra

The two definitions under the bridge theorem (ticket I4.1): `realAlgebra`, the
instruction algebra that makes `FP.Structural.exec` compute over the reals as
the proof semantics does, and `erase`, the state of the proof semantics that a
real-carrier structural state stands for.

The proof semantics computes floating channels in `WithBot ℝ`; a structural
state over `ℝ` holds no `⊥`. `erase` therefore embeds each real as `some`, and
a field of `realAlgebra` that calls an upstream tile operation reads a `⊥`
result as `0`, the value a store writes for `⊥`. The cases in which the two
executions differ are listed in `documents/BridgeTheoremDifferences.md`.
-/
import VeriTile.Triton.Float.Structural
import VeriTile.Triton.Semantics.TileOps

namespace VeriTile.Interp

open VeriTile.Triton
open VeriTile.Triton.FP

/-- The reals held by a tile of the proof semantics; `⊥` reads as `0`, as at a store. -/
def reals {shape : TileShape} (t : Tile .real shape) : TileIndex shape → ℝ :=
  fun i => (t.data i).unbotD 0

/-- Literals as given, real arithmetic, casts and rounding as identity. Every
field ignores its precision and dtype tags. Each one is the real evaluator's
treatment of the construct that reaches it, restricted to finite values;
`floatToInt` and `floatToNat` stay unsupported because the real evaluator
fails on them. -/
noncomputable def realAlgebra : Structural.Algebra ℝ where
  literal _ _ c := c
  -- `⊥` has no real counterpart; `0` is what a store of `⊥` writes.
  negInf := 0
  binary _ _ op := match op with
    | .add => (· + ·) | .sub => (· - ·) | .mul => (· * ·) | .div => (· / ·)
    | .max => max | .pow => Real.rpow
  fma _ := some fun a b c => a * b + c
  fmaTyped _ := some fun a b c => a * b + c
  atomicAdd _ := some (· + ·)
  unary _ op x := match op with
    | .neg _ => -x
    | .abs _ => |x|
    | .exp | .libdeviceExp => Real.exp x
    | .log | .libdeviceLog => Real.log x
    | .libdeviceExpm1 => Real.exp x - 1
    | .libdeviceLog1p => Real.log (1 + x)
    | .exp2 => Real.exp (x * Real.log 2)
    | .log2 => Real.log x / Real.log 2
    | .sigmoid => Real.sigmoid x
    | .sqrt => Real.sqrt x
    | .rsqrt | .libdeviceRsqrt => 1 / Real.sqrt x
    | .tanh | .libdeviceTanh => Real.tanh x
    | .sin => Real.sin x
    | .cos => Real.cos x
    | .tan => Real.tan x
    | .atan => Real.arctan x
    | .cosh => Real.cosh x
    | .sinh => Real.sinh x
    | .erf => VeriTile.Math.realErf x
    | .minPosInf => x
  libdevicePowi _ := some fun x n => x ^ n
  isFinite _ := some fun _ => true
  compareLt _ _ := some fun a b => decide (a < b)
  compareLe _ _ := some fun a b => decide (a ≤ b)
  compareEq _ _ := some fun a b => decide (a = b)
  cast _ _ _ x := x
  fromNat _ n := n
  fromInt _ n := n
  fp32Bits b := (Float32Bits.decodeReal b).unbotD 0
  float8E5Bits := some fun _ b => ((b.decodeRat.getD 0 : ℚ) : ℝ)
  fp32Load x := x
  reduceMin _ {_} axis keepDims v := (Tile.reduceMin axis keepDims (Tile.ofReal v)).map reals
  -- Structural execution calls this on a non-empty axis only.
  reduceMax _ {_} axis keepDims v :=
    ((Tile.reduceMax axis keepDims (Tile.ofReal v)).map reals).getD fun _ => 0
  reduceSum _ {_} axis keepDims v := reals (Tile.reduceSum axis keepDims (Tile.ofReal v))
  reduceSumTyped _ d {_} axis keepDims v := some fun i => Structural.ofFloat d
    (reals (Tile.reduceSum axis keepDims (Tile.ofReal fun j => Structural.toFloat d (v j))) i)
  reduceMaxTyped _ d {_} axis keepDims v :=
    (Tile.reduceMax axis keepDims (Tile.ofReal fun j => Structural.toFloat d (v j))).map
      fun m i => Structural.ofFloat d (reals m i)
  indexedMax _ d {_} axis v :=
    let x := Tile.ofReal fun j => Structural.toFloat d (v j)
    (Tile.reduceMaxDrop axis x).map fun m =>
      (fun i => Structural.ofFloat d (reals m i), (Tile.argMaxDrop axis x).data)
  scan _ {_} op axis dir v := some (reals (Tile.scan op axis dir (Tile.ofReal v)))
  dot _ {batch} {_} {_} {_} a b :=
    some (reals (Tile.dot batch (Tile.ofReal a) (Tile.ofReal b)))
  dotWithOptions _ _ {batch} {_} {_} {_} a b :=
    some (reals (Tile.dot batch (Tile.ofReal a) (Tile.ofReal b)))
  dotTyped _ l r o _ {batch} {_} {_} {_} a b := some fun i => Structural.ofFloat o
    (reals (Tile.dot batch (Tile.ofReal fun j => Structural.toFloat l (a j))
      (Tile.ofReal fun j => Structural.toFloat r (b j))) i)
  dotAcc _ l r o _ {batch} {_} {_} {_} a b acc := some fun i => Structural.ofFloat o
    (reals (Tile.dot batch (Tile.ofReal fun j => Structural.toFloat l (a j))
      (Tile.ofReal fun j => Structural.toFloat r (b j))) i + Structural.toFloat o (acc i))
  integerDot _ {batch} {_} {_} {_} a b := some (Tile.dotInt batch ⟨a⟩ ⟨b⟩).data
  minimum _ _ := some min
  divRN := some (· / ·)

/-- A value of a real-carrier structural state as a value of the proof
semantics: a real is a finite value, a discrete value is itself. -/
def liftValue : (d : TileDType) → Structural.Value ℝ d → TileCarrier d
  | .real, x | .fp32, x | .fp16, x | .bf16, x | .f8e4, x | .f8e5, x => (some x : WithBot ℝ)
  | .nat, x | .int, x | .bool, x | .ptr, x | .blockPtr, x => x

/-- A typed cell as a memory cell of the proof semantics, under the same dtype tag. -/
def liftCell : Structural.Cell ℝ → MemCell
  | .mk d v => MemCell.of d (liftValue d v)

/-- The state of the proof semantics that a real-carrier structural state
stands for. Cells and registers keep their dtype tag, because the projection
`toAlgorithm?` keeps it in the kernel. The undefined-value table loses its
dtype index: the proof semantics has one real table, taken here from the
`.real` channel. -/
def erase (s : Structural.State ℝ) : BlockState where
  mem r o := liftCell (s.mem r o)
  regs d sh n := (s.regs d sh n).map fun v => ⟨fun i => liftValue d (v i)⟩
  pids := s.pids
  undef := s.undef .real
  numPids := s.numPids

@[simp] theorem erase_mem (s : Structural.State ℝ) (r : RegionName) (o : Nat) :
    (erase s).mem r o = liftCell (s.mem r o) := rfl

@[simp] theorem erase_regs (s : Structural.State ℝ) (d : TileDType) (sh : TileShape)
    (n : RegName) :
    (erase s).regs d sh n = (s.regs d sh n).map fun v => ⟨fun i => liftValue d (v i)⟩ := rfl

@[simp] theorem erase_pids (s : Structural.State ℝ) : (erase s).pids = s.pids := rfl

@[simp] theorem erase_numPids (s : Structural.State ℝ) : (erase s).numPids = s.numPids := rfl

@[simp] theorem erase_undef (s : Structural.State ℝ) : (erase s).undef = s.undef .real := rfl

end VeriTile.Interp
