/-
VeriTile.Interp.Bridge.RealAlgebra

The two definitions under the bridge theorem (ticket I4.1): `realAlgebra`, the
instruction algebra that makes `FP.Structural.exec` compute as the proof
semantics does, and `erase`, the state of the proof semantics that a
structural state stands for.

The carrier is `WithBot ℝ`, the type in which the proof semantics computes
every floating channel; `⊥` is `-inf`. Each field of `realAlgebra` is the
function the real evaluator (`Semantics/EvalOp.lean`) applies to the same
construct. `erase` changes a value in two places only, where the proof
semantics holds a real and not a `WithBot ℝ`: a memory cell and the
undefined-value table. The cases in which the two executions still differ are
listed in `documents/BridgeTheoremDifferences.md`.
-/
import VeriTile.Triton.Float.Structural
import VeriTile.Triton.Semantics.TileOps

namespace VeriTile.Interp

open VeriTile.Triton
open VeriTile.Triton.FP

/-- A structural value over `WithBot ℝ` as a value of the proof semantics.
The two carriers are the same type at every dtype. -/
def toCarrier : (d : TileDType) → Structural.Value (WithBot ℝ) d → TileCarrier d
  | .real, x | .fp32, x | .fp16, x | .bf16, x | .f8e4, x | .f8e5, x
  | .nat, x | .int, x | .bool, x | .ptr, x | .blockPtr, x => x

/-- The inverse of `toCarrier`. -/
def ofCarrier : (d : TileDType) → TileCarrier d → Structural.Value (WithBot ℝ) d
  | .real, x | .fp32, x | .fp16, x | .bf16, x | .f8e4, x | .f8e5, x
  | .nat, x | .int, x | .bool, x | .ptr, x | .blockPtr, x => x

/-- A structural tile as a tile of the proof semantics. -/
def toTile {d : TileDType} {shape : TileShape} (v : Structural.Values (WithBot ℝ) d shape) :
    Tile d shape :=
  ⟨fun i => toCarrier d (v i)⟩

/-- The inverse of `toTile`. -/
def ofTile {d : TileDType} {shape : TileShape} (t : Tile d shape) :
    Structural.Values (WithBot ℝ) d shape :=
  fun i => ofCarrier d (t.data i)

/-- The real evaluator's operations on `WithBot ℝ`. Every field ignores its
precision and dtype tags; casts and loads are the identity. `floatToInt`,
`floatToNat` and `isFinite` stay unsupported: the real evaluator fails on the
first two, and the projection `toAlgorithm?` rejects the third. -/
noncomputable def realAlgebra : Structural.Algebra (WithBot ℝ) where
  literal _ _ c := (c : ℝ)
  negInf := ⊥
  binary _ _ op := match op with
    | .add => WithBot.realAdd | .sub => WithBot.realSub | .mul => WithBot.realMul
    | .div => WithBot.realDiv | .max => max | .pow => WithBot.realPow
  fma _ := some fun a b c => WithBot.realAdd (WithBot.realMul a b) c
  fmaTyped _ := some fun a b c => WithBot.realAdd (WithBot.realMul a b) c
  atomicAdd _ := some WithBot.realAdd
  unary _ op := match op with
    | .neg _ => WithBot.map fun x => -x
    | .abs _ => WithBot.map fun x => |x|
    | .exp | .libdeviceExp => WithBot.realExp
    | .log | .libdeviceLog => WithBot.realLog
    | .libdeviceExpm1 => WithBot.realExpm1
    | .libdeviceLog1p => WithBot.realLog1p
    | .exp2 => WithBot.realExp2
    | .log2 => WithBot.realLog2
    | .sigmoid => WithBot.realSigmoid
    | .sqrt => WithBot.realSqrt
    | .rsqrt | .libdeviceRsqrt => WithBot.realRsqrt
    | .tanh | .libdeviceTanh => WithBot.realTanh
    | .sin => WithBot.realSin
    | .cos => WithBot.realCos
    | .tan => WithBot.realTan
    | .atan => WithBot.realAtan
    | .cosh => WithBot.realCosh
    | .sinh => WithBot.realSinh
    | .erf => WithBot.realErf
    | .minPosInf => id
  libdevicePowi _ := some WithBot.realPowi
  compareLt _ _ := some fun a b => decide (a < b)
  compareLe _ _ := some fun a b => decide (a ≤ b)
  compareEq _ _ := some fun a b => decide (a = b)
  cast _ _ _ := id
  fromNat _ n := (n : ℝ)
  fromInt _ n := (n : ℝ)
  fp32Bits := Float32Bits.decodeReal
  -- The real evaluator fails on a byte without a rational value; see D5.
  float8E5Bits := some fun _ b => ((b.decodeRat.getD 0 : ℚ) : ℝ)
  fp32Load := id
  reduceMin _ {_} axis keepDims v := (Tile.reduceMin axis keepDims (toTile v)).map ofTile
  -- Structural execution calls this on a non-empty axis only.
  reduceMax _ {_} axis keepDims v :=
    ((Tile.reduceMax axis keepDims (toTile v)).map ofTile).getD fun _ => ⊥
  reduceSum _ {_} axis keepDims v := ofTile (Tile.reduceSum axis keepDims (toTile v))
  reduceSumTyped _ d {_} axis keepDims v :=
    some (ofTile (Tile.reduceSumTyped d axis keepDims (toTile v)))
  reduceMaxTyped _ d {_} axis keepDims v :=
    (Tile.reduceMaxTyped d axis keepDims (toTile v)).map ofTile
  indexedMax _ d {_} axis v := (Tile.reduceMaxTyped d axis false (toTile v)).map fun m =>
    (ofTile m, (Tile.argMaxTyped d axis (toTile v)).data)
  scan _ {_} op axis dir v := some (ofTile (Tile.scan op axis dir (toTile v)))
  dot _ {batch} {_} {_} {_} a b := some (ofTile (Tile.dot batch (toTile a) (toTile b)))
  dotWithOptions _ _ {batch} {_} {_} {_} a b :=
    some (ofTile (Tile.dot batch (toTile a) (toTile b)))
  dotTyped _ l r o _ {batch} {_} {_} {_} a b := some fun i => Structural.ofFloat o
    ((Tile.dot batch ⟨fun j => Structural.toFloat l (a j)⟩
      ⟨fun j => Structural.toFloat r (b j)⟩).data i)
  dotAcc _ l r o _ {batch} {_} {_} {_} a b acc := some fun i => Structural.ofFloat o
    (WithBot.realAdd ((Tile.dot batch ⟨fun j => Structural.toFloat l (a j)⟩
      ⟨fun j => Structural.toFloat r (b j)⟩).data i) (Structural.toFloat o (acc i)))
  integerDot _ {batch} {_} {_} {_} a b := some (ofTile (Tile.dotInt batch (toTile a) (toTile b)))
  minimum _ _ := some min
  divRN := some WithBot.realDiv

/-- What the proof semantics keeps of a value in memory: its stores write `0`
for a floating `⊥` and its loads read `0` for one (`FloatDType.storeValue`). -/
def storedValue : (d : TileDType) → Structural.Value (WithBot ℝ) d → TileCarrier d
  | .real, x | .fp32, x | .fp16, x | .bf16, x | .f8e4, x | .f8e5, x =>
    (WithBot.unbotD 0 x : ℝ)
  | .nat, x | .int, x | .bool, x | .ptr, x | .blockPtr, x => x

/-- A typed cell as a memory cell of the proof semantics, under the same dtype tag. -/
def eraseCell : Structural.Cell (WithBot ℝ) → MemCell
  | .mk d v => MemCell.of d (storedValue d v)

/-- The state of the proof semantics that a structural state over `WithBot ℝ`
stands for. Registers are copied. Cells keep their dtype tag, because the
projection `toAlgorithm?` keeps it in the kernel, and a `⊥` in a floating cell
becomes `0`. The proof semantics has one undefined-value table, of reals: it is
the `.real` table, with `0` for `⊥`. -/
def erase (s : Structural.State (WithBot ℝ)) : BlockState where
  mem r o := eraseCell (s.mem r o)
  regs d sh n := (s.regs d sh n).map toTile
  pids := s.pids
  undef r o := WithBot.unbotD 0 (s.undef .real r o)
  numPids := s.numPids

@[simp] theorem erase_mem (s : Structural.State (WithBot ℝ)) (r : RegionName) (o : Nat) :
    (erase s).mem r o = eraseCell (s.mem r o) := rfl

@[simp] theorem erase_regs (s : Structural.State (WithBot ℝ)) (d : TileDType)
    (sh : TileShape) (n : RegName) :
    (erase s).regs d sh n = (s.regs d sh n).map toTile := rfl

@[simp] theorem erase_pids (s : Structural.State (WithBot ℝ)) : (erase s).pids = s.pids := rfl

@[simp] theorem erase_numPids (s : Structural.State (WithBot ℝ)) :
    (erase s).numPids = s.numPids := rfl

@[simp] theorem erase_undef (s : Structural.State (WithBot ℝ)) (r : RegionName) (o : Nat) :
    (erase s).undef r o = WithBot.unbotD 0 (s.undef .real r o) := rfl

end VeriTile.Interp
