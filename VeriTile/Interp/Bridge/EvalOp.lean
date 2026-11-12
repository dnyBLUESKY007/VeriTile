/-
VeriTile.Interp.Bridge.EvalOp

The bridge theorem at the level of expressions (ticket I4.2): structural
evaluation over `realAlgebra` and the proof semantics agree under erasure.
Both succeed with the same value, or both fail.

* `evalOp_bridge`: every `Op`, in every precision context.
* `evalComputeOp_bridge`, `evalExpr_bridge`: a `ComputeOp` or a `ComputeExpr`
  against its projection `toAlgorithm?`.

Each difference between the two sides is a named precondition; the numbers are
those of `documents/BridgeTheoremDifferences.md`. `BridgeSite s` is what one
node needs in the state `s`, and `AllNodes (BridgeSite s) e` asks for it at
every node of `e`.

| Precondition | Difference | Constrains |
| --- | --- | --- |
| `StructuralNode` | D3, D4 | the node |
| `NonEmptyArgMax` | D6 | the node |
| `ReshapeSameSize` | D7 | the node |
| `DecodesFinite` | D5 | the bytes a node decodes in `s` |
| `LoadOk`: `CellFinite`, `CellTagged`, `UndefAgrees` | D1, D13, D8 | what a load reads in `s` |
| `SupportedCompute`, `SupportedExpr` | D9, D10 | the compute operation |

Every constructor is in exactly one list.

`Op`, 113 constructors.

Proved under the preconditions named after the constructor, 110:
`const`, `constFloat`, `constNat`, `constInt`, `constBool`, `negInf`,
`programId`, `numPrograms`, `ref`, `arange`, `rand`, `randTensor`,
`philox4x32`, `uniform32`, `broadcast`, `full`, `withPrecision`, `castFloat`,
`negFloat`, `absFloat`, `floatToInt`, `floatToNat`, `castNatToInt`,
`castIntToNat`, `decodeFloat8E5` (D5), `add`, `sub`, `mul`, `div`, `divRN`,
`floorDiv`, `mod`, `bitAnd`, `bitOr`, `bitXor`, `shiftLeft`, `shiftRight`,
`exp`, `libdeviceExp`, `libdeviceLog`, `libdeviceExpm1`, `libdeviceLog1p`,
`libdeviceTanh`, `exp2`, `log`, `log2`, `sigmoid`, `sqrt`, `rsqrt`,
`libdeviceRsqrt`, `tanh`, `sin`, `cos`, `tan`, `atan`, `cosh`, `sinh`, `erf`,
`minPosInf`, `lt`, `le`, `eq`, `gt`, `ge`, `ne`, `boolAnd`, `boolOr`,
`boolNot`, `max2`, `maximumFloat`, `pow`, `libdevicePowi`, `where`, `ite`,
`reduceMin`, `reduceMax`, `reduceMaxNat`, `reduceMaxInt`, `reduceSum`,
`reduceSumBool`, `reduceSumTyped`, `reduceMaxTyped`, `scan`, `argMax` (D6),
`argMaxInt`, `indexedMaxValue`, `argMaxTyped` (D6), `dot`, `dotTyped`,
`dotAcc`, `dotInt`, `transpose`, `reshape` (D7), `remap`, `join`, `split`,
`expandDim`, `ptrBase`, `ptrAdd`, `ptrSub`, `makeBlockPtr`, `makeBlockPtrDyn`,
`makeBlockPtrDynOffsets`, `makeBlockPtrDynMetadata`, `advanceBlockPtr`,
`advanceBlockPtrDyn`, `load` (D1, D8, D13; the three address forms, the three
mask forms and both paddings), `natToReal`, `intToReal`, `minimumFloat`.

Excluded by a precondition, 3: `castRealToInt8` (D3), `argMin`, `sort` (D4).

Not proved: none.

`ComputeOp`, 5 constructors. Proved, 5: `alg`, `const` (D9 excludes `fp64`),
`full`, `load` (D9 excludes `fp64`), `bitcast` (D10 keeps `int32` and `uint32`
to `fp32`). Excluded: none. Not proved: none.

`ComputeExpr`, 7 constructors. Proved, 4: `alg`, `compute`, `fma`, `fmaTyped`.
Excluded, 3: `opaque`, `llrint`, `isFinite`; their projection fails, so the
hypothesis `toAlgorithm? = .ok op` is false. Not proved: none.

D11 and D12 are about statements and belong to the statement-level bridge.
-/
import VeriTile.Interp.Bridge.RealAlgebra
import VeriTile.Triton.Semantics.EvalOp
import VeriTile.Triton.Float.DecoderCompatibility

namespace VeriTile.Interp

open VeriTile.Triton
open VeriTile.Triton.FP

local notation "SE" => Structural.evalOp realAlgebra

/-! ## Preconditions -/

/-- D3, D4. The node is one structural execution evaluates. It fails on
`castRealToInt8`, `argMin` and `sort`, which the proof semantics evaluates. -/
def StructuralNode : {d : TileDType} → {sh : TileShape} → Op d sh → Prop
  | _, _, .castRealToInt8 _ | _, _, .argMin _ _ | _, _, .sort _ _ => False
  | _, _, _ => True

/-- D6. An `argMax` or `argMaxTyped` reduces a non-empty axis. On an empty one
structural execution fails and the proof semantics returns index `0`. -/
def NonEmptyArgMax : {d : TileDType} → {sh : TileShape} → Op d sh → Prop
  | _, _, .argMax (shape := shape) axis _ | _, _, .argMaxTyped (shape := shape) _ axis _ _ =>
      shape.axisDim axis ≠ 0
  | _, _, _ => True

/-- D7. A `reshape` keeps the number of elements. Otherwise structural
execution fails and the proof semantics pads with the dtype default. -/
def ReshapeSameSize : {d : TileDType} → {sh : TileShape} → Op d sh → Prop
  | _, _, .reshape (shape := input) output _ => input.product = output.product
  | _, _, _ => True

/-- D5. Every byte that `decodeFloat8E5` decodes has a finite encoding. On an
infinity or a NaN structural execution returns `0` for that byte and the proof
semantics fails for the whole tile. -/
def DecodesFinite (s : Structural.State (WithBot ℝ)) :
    {d : TileDType} → {sh : TileShape} → Op d sh → Prop
  | _, sh, .decodeFloat8E5 _ a => ∀ p v, SE p a s = some v → ∀ i : TileIndex sh,
      Float8E5Bits.isInt8 (v i) = true → (Float8E5Bits.ofInt (v i)).decodeRat.isSome = true
  | _, _, _ => True

/-- D1. A cell read at its own dtype `d` holds a value the proof semantics can
keep in memory: not a floating `⊥`, for which it stores and reads `0`. -/
def CellFinite (d : TileDType) (c : Structural.Cell (WithBot ℝ)) : Prop :=
  ∀ v, c = .mk d v → storedValue d v = toCarrier d v

/-- D13. A floating read at dtype `d` finds a cell tagged `d`. On another tag
structural execution reads the carrier default `⊥` and the proof semantics `0`. -/
def CellTagged (d : TileDType) (c : Structural.Cell (WithBot ℝ)) : Prop :=
  (∃ f : FloatDType, f.toTileDType = d) → ∃ v, c = .mk d v

/-- D8. The undefined value structural execution takes from its table of dtype
`d` is the one the proof semantics derives from its single real table: the
dtype default in a masked-off discrete lane (`masked`), and
`BlockState.unspecifiedValue` in every other case. -/
def UndefAgrees (s : Structural.State (WithBot ℝ)) (d : TileDType) (masked : Bool)
    (r : RegionName) (o : Nat) : Prop :=
  toCarrier d (s.undef d r o) =
    open Classical in
    if masked = true ∧ ∀ f : FloatDType, f.toTileDType ≠ d then BlockState.defaultCarrier d
    else (erase s).unspecifiedValue d r o

/-- The address of each lane of a load, and whether it is in bounds: always,
except through a block pointer. -/
noncomputable def laneAddrs (p : Option ComputeDType) (s : Structural.State (WithBot ℝ))
    {d : TileDType} {sh : TileShape} :
    MemAccess d sh → Option (TileIndex sh → RegionName × Nat × Bool)
  | .region r off => (SE p off s).map fun v i => (Region.cast r, v i, true)
  | .ptr e => (SE p e s).map fun v i => ((v i).1, (v i).2, true)
  | .blockPtr e check => (SE p e s).map fun v i =>
      ((v i).region, (v i).address (TileShape.indexToList sh i),
        (v i).inBounds (TileShape.indexToList sh i) check)

/-- The lanes a load reads. -/
noncomputable def laneActive (p : Option ComputeDType) (s : Structural.State (WithBot ℝ))
    {d : TileDType} {sh : TileShape} : MaskOpt d sh → Option (TileIndex sh → Bool)
  | .none => some fun _ => true
  | .mask m | .maskOther m _ => SE p m s

/-- D1, D8, D13 at a `load`, lane by lane. An active in-bounds lane reads a
cell that is `CellFinite` and `CellTagged`. A lane that takes an undefined
value satisfies `UndefAgrees`: an active out-of-bounds lane under unspecified
padding, and a masked-off lane of a load without `other` unless it is out of
bounds under zero padding. -/
def LoadOk (s : Structural.State (WithBot ℝ)) :
    {d : TileDType} → {sh : TileShape} → Op d sh → Prop
  | _, sh, .load d mem mask padding =>
      ∀ p addr active, laneAddrs p s mem = some addr → laneActive p s mask = some active →
        ∀ i : TileIndex sh,
          ((active i = true → (addr i).2.2 = true →
            CellFinite d (s.mem (addr i).1 (addr i).2.1) ∧
              CellTagged d (s.mem (addr i).1 (addr i).2.1)) ∧
          (active i = true → (addr i).2.2 = false → padding = .unspecified →
            UndefAgrees s d false (addr i).1 (addr i).2.1) ∧
          (active i = false → (∃ m, mask = .mask m) →
            ((addr i).2.2 = true ∨ padding = .unspecified) →
            UndefAgrees s d true (addr i).1 (addr i).2.1))
  | _, _, _ => True

/-- What the bridge needs of one node of an expression evaluated in `s`. -/
def BridgeSite (s : Structural.State (WithBot ℝ)) {d : TileDType} {sh : TileShape}
    (e : Op d sh) : Prop :=
  StructuralNode e ∧ NonEmptyArgMax e ∧ ReshapeSameSize e ∧ DecodesFinite s e ∧ LoadOk s e

set_option maxHeartbeats 1600000 in
/-- `P` holds at every node of the expression: the node itself, the addresses,
masks and `other` values of its loads, and its dynamic offset lists. -/
def AllNodes (P : ∀ {d : TileDType} {sh : TileShape}, Op d sh → Prop) :
    {d : TileDType} → {sh : TileShape} → Op d sh → Prop
  | _, _, e@(.const _) | _, _, e@(.constFloat _ _) | _, _, e@(.constNat _) | _, _, e@(.constInt _)
  | _, _, e@(.constBool _) | _, _, e@(.negInf) | _, _, e@(.programId _) | _, _, e@(.numPrograms _)
  | _, _, e@(.ref _ _ _) | _, _, e@(.arange _) | _, _, e@(.ptrBase _)
  | _, _, e@(.makeBlockPtr _ _ _ _ _ _) => P e
  | _, _, e@(.uniform32 a) | _, _, e@(.broadcast a _) | _, _, e@(.full _ a)
  | _, _, e@(.withPrecision _ _ a) | _, _, e@(.castFloat _ _ a) | _, _, e@(.negFloat _ a)
  | _, _, e@(.absFloat _ a) | _, _, e@(.floatToInt _ _ _ a) | _, _, e@(.floatToNat _ _ _ a)
  | _, _, e@(.castNatToInt a) | _, _, e@(.castIntToNat a) | _, _, e@(.castRealToInt8 a)
  | _, _, e@(.decodeFloat8E5 _ a) | _, _, e@(.exp a) | _, _, e@(.libdeviceExp a)
  | _, _, e@(.libdeviceLog a) | _, _, e@(.libdeviceExpm1 a) | _, _, e@(.libdeviceLog1p a)
  | _, _, e@(.libdeviceTanh a) | _, _, e@(.exp2 a) | _, _, e@(.log a) | _, _, e@(.log2 a)
  | _, _, e@(.sigmoid a) | _, _, e@(.sqrt a) | _, _, e@(.rsqrt a) | _, _, e@(.libdeviceRsqrt a)
  | _, _, e@(.tanh a) | _, _, e@(.sin a) | _, _, e@(.cos a) | _, _, e@(.tan a) | _, _, e@(.atan a)
  | _, _, e@(.cosh a) | _, _, e@(.sinh a) | _, _, e@(.erf a) | _, _, e@(.minPosInf a)
  | _, _, e@(.boolNot a) | _, _, e@(.reduceMin _ _ a) | _, _, e@(.reduceMax _ _ a)
  | _, _, e@(.reduceMaxNat _ _ a) | _, _, e@(.reduceMaxInt _ _ a) | _, _, e@(.reduceSum _ _ a)
  | _, _, e@(.reduceSumBool _ _ a) | _, _, e@(.reduceSumTyped _ _ _ a)
  | _, _, e@(.reduceMaxTyped _ _ _ a) | _, _, e@(.scan _ _ _ a) | _, _, e@(.argMax _ a)
  | _, _, e@(.argMaxInt _ a) | _, _, e@(.indexedMaxValue _ _ a) | _, _, e@(.argMaxTyped _ _ a _)
  | _, _, e@(.argMin _ a) | _, _, e@(.sort _ a) | _, _, e@(.transpose a) | _, _, e@(.reshape _ a)
  | _, _, e@(.remap _ _ a) | _, _, e@(.split _ a) | _, _, e@(.expandDim _ a)
  | _, _, e@(.makeBlockPtrDyn _ a _ _ _ _) | _, _, e@(.advanceBlockPtr a _)
  | _, _, e@(.natToReal a) | _, _, e@(.intToReal a) => P e ∧ AllNodes P a
  | _, _, e@(.rand _ a b) | _, _, e@(.randTensor _ a b) | _, _, e@(.add _ _ a b)
  | _, _, e@(.sub _ _ a b) | _, _, e@(.mul _ _ a b) | _, _, e@(.div _ _ a b)
  | _, _, e@(.divRN _ a b) | _, _, e@(.floorDiv _ _ a b) | _, _, e@(.mod _ _ a b)
  | _, _, e@(.bitAnd _ a b) | _, _, e@(.bitOr _ a b) | _, _, e@(.bitXor _ a b)
  | _, _, e@(.shiftLeft _ a b) | _, _, e@(.shiftRight _ a b) | _, _, e@(.lt _ _ a b)
  | _, _, e@(.le _ _ a b) | _, _, e@(.eq _ _ a b) | _, _, e@(.gt _ _ a b) | _, _, e@(.ge _ _ a b)
  | _, _, e@(.ne _ _ a b) | _, _, e@(.boolAnd _ a b) | _, _, e@(.boolOr _ a b)
  | _, _, e@(.max2 _ a b) | _, _, e@(.maximumFloat _ _ a b) | _, _, e@(.minimumFloat _ _ a b)
  | _, _, e@(.pow _ a b) | _, _, e@(.libdevicePowi _ a b) | _, _, e@(.dot a b _)
  | _, _, e@(.dotTyped _ _ _ a b _) | _, _, e@(.dotInt a b _) | _, _, e@(.join a b)
  | _, _, e@(.ptrAdd _ a b) | _, _, e@(.ptrSub _ a b) => P e ∧ AllNodes P a ∧ AllNodes P b
  | _, _, e@(.where a b c) | _, _, e@(.ite a b c) | _, _, e@(.dotAcc _ _ _ a b c _) =>
      P e ∧ AllNodes P a ∧ AllNodes P b ∧ AllNodes P c
  | _, _, e@(.philox4x32 _ _ _ seed c0 c1 c2 c3) =>
      P e ∧ AllNodes P seed ∧ AllNodes P c0 ∧ AllNodes P c1 ∧ AllNodes P c2 ∧ AllNodes P c3
  | _, _, e@(.makeBlockPtrDynOffsets _ a _ _ _ offsets) =>
      P e ∧ AllNodes P a ∧ ∀ x ∈ offsets, AllNodes P x
  | _, _, e@(.makeBlockPtrDynMetadata _ a _ _ strides offsets) =>
      P e ∧ AllNodes P a ∧ (∀ x ∈ strides, AllNodes P x) ∧ ∀ x ∈ offsets, AllNodes P x
  | _, _, e@(.advanceBlockPtrDyn a deltas) => P e ∧ AllNodes P a ∧ ∀ x ∈ deltas, AllNodes P x
  | _, _, e@(.load _ mem mask _) =>
      P e ∧
      (match mem with
        | .region _ off => AllNodes P off
        | .ptr ptr => AllNodes P ptr
        | .blockPtr ptr _ => AllNodes P ptr) ∧
      (match mask with
        | .none => True
        | .mask m => AllNodes P m
        | .maskOther m other => AllNodes P m ∧ AllNodes P other)

/-! ## Values -/

@[simp] theorem toTile_ofTile {d : TileDType} {sh : TileShape} (t : Tile d sh) :
    toTile (ofTile t) = t := by
  cases d <;> rfl

@[simp] theorem map_toTile_ofTile {d : TileDType} {sh : TileShape} (x : Option (Tile d sh)) :
    x.map (toTile ∘ ofTile) = x := by
  cases x <;> simp

private theorem reduceMaxTyped_isSome (d : FloatDType) {sh : TileShape} (axis : Fin sh.length)
    (x : Tile d.toTileDType sh) (h : sh.axisDim axis ≠ 0) :
    ∃ m, Tile.reduceMaxTyped d axis false x = some m := by
  have : 0 < sh.axisDim axis := by omega
  simp [Tile.reduceMaxTyped, Tile.reduceMax, Tile.reduceMaxDrop, this]

private theorem toTile_transpose {d : TileDType} : ∀ (batch : TileShape) {M N : Nat}
    (v : Structural.Values (WithBot ℝ) d (batch ++ [M, N])),
    toTile (Structural.transposeValues batch v) = Tile.transpose batch (toTile v)
  | [], _, _, _ => rfl
  | _ :: rest, _, _, v => by
    have ih := fun i => toTile_transpose rest (fun k => v (i, k))
    simp only [toTile, Structural.transposeValues, Tile.transpose] at ih ⊢
    congr 1; funext ⟨i, tail⟩
    exact congrArg (fun t => t.data tail) (ih i)

/-- `fp32Bits` on bits with a rational value is that value. -/
private theorem fp32Bits_of_toRat (x : FP.Value .fp32) (q : ℚ) (h : x.toRat? = some q) :
    realAlgebra.fp32Bits ⟨x.bits⟩ = ((q : ℝ) : WithBot ℝ) := by
  have h' : (⟨x.bits⟩ : FP.Value .fp32).toRat? = some q := h
  simp [realAlgebra, Float32Bits.decodeReal, DecoderCompatibility.equals_software_decoder, h']
  rfl

/-! ## Memory lanes -/

private theorem read_agree (s : Structural.State (WithBot ℝ)) (d : TileDType) (r : RegionName)
    (o : Nat) (hf : CellFinite d (s.mem r o)) (ht : CellTagged d (s.mem r o)) :
    toCarrier d ((s.mem r o).read d) = (erase s).readMemValue d r o := by
  cases hc : s.mem r o with
  | mk d' v =>
    have hm : (erase s).mem r o = MemCell.of d' (storedValue d' v) := by rw [erase_mem, hc]; rfl
    by_cases hd : d' = d
    · subst hd
      rw [Structural.Cell.read_same, ← hf v hc]
      cases d' <;>
        simp only [BlockState.readMemValue, BlockState.readMemAs, BlockState.readMemTyped, hm,
          MemCell.readAs_of_same] <;> rfl
    · have hnf : ¬ ∃ f : FloatDType, f.toTileDType = d := fun hfl => by
        obtain ⟨v', hv'⟩ := ht hfl
        rw [hc] at hv'
        cases hv'
        exact hd rfl
      cases d <;>
        first
        | exact absurd ⟨.real, rfl⟩ hnf | exact absurd ⟨.fp32, rfl⟩ hnf
        | exact absurd ⟨.fp16, rfl⟩ hnf | exact absurd ⟨.bf16, rfl⟩ hnf
        | exact absurd ⟨.f8e4, rfl⟩ hnf | exact absurd ⟨.f8e5, rfl⟩ hnf
        | simp [Structural.Cell.read, hd, Structural.defaultValue, BlockState.readMemValue,
            BlockState.readMemTyped, hc, eraseCell, MemCell.of, MemCell.readAs,
            BlockState.defaultCarrier, toCarrier]

private theorem zero_agree (p : Option ComputeDType) (d : TileDType) :
    toCarrier d (Structural.boundaryZero realAlgebra p d) = BlockState.defaultCarrier d := by
  cases d <;> rfl

/-- An active lane of a load. -/
private theorem active_agree (s : Structural.State (WithBot ℝ)) (d : TileDType)
    (padding : PaddingOption) (r : RegionName) (o : Nat) (ok : Bool) (p : Option ComputeDType)
    (h1 : ok = true → CellFinite d (s.mem r o) ∧ CellTagged d (s.mem r o))
    (h2 : ok = false → padding = .unspecified → UndefAgrees s d false r o) :
    toCarrier d (if ok = true then (s.mem r o).read d else
        match (generalizing := false) padding with
        | .zero => Structural.boundaryZero realAlgebra p d
        | .unspecified => s.undef d r o) =
      if ok = true then (erase s).readMemValue d r o else (erase s).paddingValue d padding r o := by
  cases ok with
  | true => simpa using read_agree s d r o (h1 rfl).1 (h1 rfl).2
  | false =>
    cases padding with
    | zero => simpa using zero_agree p d
    | unspecified => simpa [UndefAgrees, BlockState.paddingValue] using h2 rfl rfl

/-- A masked-off lane of a load without `other`. -/
private theorem masked_agree (s : Structural.State (WithBot ℝ)) (d : TileDType)
    (padding : PaddingOption) (r : RegionName) (o : Nat) (ok : Bool) (p : Option ComputeDType)
    (h : ok = true ∨ padding = .unspecified → UndefAgrees s d true r o) :
    toCarrier d (if ok = true then s.undef d r o else
        match (generalizing := false) padding with
        | .zero => Structural.boundaryZero realAlgebra p d
        | .unspecified => s.undef d r o) =
      (match (generalizing := false) d with
      | .real => if ok = true then some ((erase s).undef r o)
          else (erase s).paddingValue .real padding r o
      | .fp32 => if ok = true then FloatDType.fp32.ofReal ((erase s).undef r o)
          else (erase s).paddingValue .fp32 padding r o
      | .fp16 => if ok = true then FloatDType.fp16.ofReal ((erase s).undef r o)
          else (erase s).paddingValue .fp16 padding r o
      | .bf16 => if ok = true then FloatDType.bf16.ofReal ((erase s).undef r o)
          else (erase s).paddingValue .bf16 padding r o
      | .f8e4 => if ok = true then FloatDType.f8e4.ofReal ((erase s).undef r o)
          else (erase s).paddingValue .f8e4 padding r o
      | .f8e5 => if ok = true then FloatDType.f8e5.ofReal ((erase s).undef r o)
          else (erase s).paddingValue .f8e5 padding r o
      | dtype => BlockState.defaultCarrier dtype) := by
  have hdisc : ∀ d : TileDType, (∀ f : FloatDType, f.toTileDType ≠ d) ↔
      d = .int ∨ d = .nat ∨ d = .bool ∨ d = .ptr ∨ d = .blockPtr := by
    intro d; constructor
    · intro h
      cases d <;> (first | exact absurd rfl (h .real) | exact absurd rfl (h .fp32) |
        exact absurd rfl (h .fp16) | exact absurd rfl (h .bf16) | exact absurd rfl (h .f8e4) |
        exact absurd rfl (h .f8e5) | simp)
    · rintro (h | h | h | h | h) f <;> subst h <;> cases f <;> simp
  cases d <;> cases ok <;> cases padding <;>
    simp [UndefAgrees, hdisc, BlockState.paddingValue, BlockState.unspecifiedValue,
      FloatDType.ofReal, FloatDType.ofWithBot, toCarrier, Structural.boundaryZero, realAlgebra,
      BlockState.defaultCarrier, Structural.defaultValue] at h ⊢ <;>
    (first | exact h | rfl)


/-- The `load` step: the three address forms and the three mask forms, given
the bridge for the address, mask and `other` expressions. -/
private theorem load_bridge (s : Structural.State (WithBot ℝ)) {d : TileDType} {sh : TileShape}
    (mem : MemAccess d sh) (mask : MaskOpt d sh) (padding : PaddingOption)
    (p : Option ComputeDType)
    (hmem : match mem with
      | .region _ e | .ptr e | .blockPtr e _ => ∀ p, (SE p e s).map toTile = evalOp e (erase s))
    (hmask : match mask with
      | .none => True
      | .mask m => ∀ p, (SE p m s).map toTile = evalOp m (erase s)
      | .maskOther m other => (∀ p, (SE p m s).map toTile = evalOp m (erase s)) ∧
          ∀ p, (SE p other s).map toTile = evalOp other (erase s))
    (hok : LoadOk s (.load d mem mask padding)) :
    (SE p (.load d mem mask padding) s).map toTile =
      evalOp (.load d mem mask padding) (erase s) := by
  simp only [LoadOk] at hok
  cases mask with
  | none =>
    cases mem with
    | region r e | ptr e | blockPtr e check =>
      rw [Structural.evalOp_unfold]; simp only [evalOp]; rw [← hmem p]
      cases ha : SE p e s with
      | none => rfl
      | some v =>
        have hl := hok p _ _ (by simp only [laneAddrs, ha]; rfl) rfl
        refine congrArg some (Tile.ext fun i => ?_)
        have h := active_agree s d padding _ _ _ p (fun hk => (hl i).1 rfl hk)
          (fun hk => (hl i).2.1 rfl hk)
        simpa using h
  | mask m =>
    cases mem with
    | region r e | ptr e | blockPtr e check =>
      rw [Structural.evalOp_unfold]; simp only [evalOp]; rw [← hmem p, ← hmask p]
      cases ha : SE p e s with
      | none => rfl
      | some v =>
        cases hm : SE p m s with
        | none => rfl
        | some act =>
          have hl := hok p _ _ (by simp only [laneAddrs, ha]; rfl)
            (by simp only [laneActive]; exact hm)
          refine congrArg some (Tile.ext fun i => ?_)
          cases hact : act i with
          | true =>
            have h := active_agree s d padding _ _ _ p (fun hk => (hl i).1 hact hk)
              (fun hk => (hl i).2.1 hact hk)
            simpa [toTile, toCarrier, hact] using h
          | false =>
            have h := masked_agree s d padding _ _ _ p ((hl i).2.2 hact ⟨m, rfl⟩)
            cases d <;> simpa [toTile, toCarrier, hact] using h
  | maskOther m other =>
    cases mem with
    | region r e | ptr e | blockPtr e check =>
      rw [Structural.evalOp_unfold]; simp only [evalOp]; rw [← hmem p, ← hmask.1 p, ← hmask.2 p]
      cases ha : SE p e s with
      | none => rfl
      | some v =>
        cases hm : SE p m s with
        | none => rfl
        | some act =>
          cases ho : SE p other s with
          | none => rfl
          | some fb =>
            have hl := hok p _ _ (by simp only [laneAddrs, ha]; rfl)
              (by simp only [laneActive]; exact hm)
            refine congrArg some (Tile.ext fun i => ?_)
            cases hact : act i with
            | true =>
              have h := active_agree s d padding _ _ _ p (fun hk => (hl i).1 hact hk)
                (fun hk => (hl i).2.1 hact hk)
              simpa [toTile, toCarrier, hact] using h
            | false => simp [toTile, toCarrier, hact]

/-! ## Dynamic lists, reshape, byte decoding -/

private theorem evalOffsets_bridge (s : Structural.State (WithBot ℝ)) (p : Option ComputeDType) :
    ∀ l : List (Op .nat []), (∀ e ∈ l, ∀ p, (SE p e s).map toTile = evalOp e (erase s)) →
      Structural.evalOffsets realAlgebra p l s =
        l.mapM (fun off => do
          let v ← evalOp off (erase s)
          some (v.data PUnit.unit))
  | [], _ => by simp [Structural.evalOffsets]
  | e :: rest, h => by
      rw [Structural.evalOffsets, List.mapM_cons, ← h e (List.mem_cons_self ..) p,
        evalOffsets_bridge s p rest fun x hx => h x (List.mem_cons_of_mem _ hx)]
      cases SE p e s <;> rfl

private theorem evalDeltas_bridge (s : Structural.State (WithBot ℝ)) (p : Option ComputeDType) :
    ∀ l : List (Op .int []), (∀ e ∈ l, ∀ p, (SE p e s).map toTile = evalOp e (erase s)) →
      Structural.evalDeltas realAlgebra p l s =
        l.mapM (fun off => do
          let v ← evalOp off (erase s)
          some (v.data PUnit.unit))
  | [], _ => by simp [Structural.evalDeltas]
  | e :: rest, h => by
      rw [Structural.evalDeltas, List.mapM_cons, ← h e (List.mem_cons_self ..) p,
        evalDeltas_bridge s p rest fun x hx => h x (List.mem_cons_of_mem _ hx)]
      cases SE p e s <;> rfl

private theorem toTile_reshape {d : TileDType} {input output : TileShape}
    (h : input.product = output.product) (v : Structural.Values (WithBot ℝ) d input) :
    toTile (fun i => v (TileShape.reshapeIndex h i)) = Tile.reshape (toTile v) := by
  refine Tile.ext fun i => ?_
  have hlt : TileShape.linearIndex output i < (TileShape.allIndices input).length := by
    rw [TileShape.length_allIndices, h]; exact TileShape.linearIndex_lt output i
  simp [toTile, Tile.reshape, TileShape.reshapeIndex, List.getElem?_eq_getElem hlt]

private theorem decode_bridge (dst : FloatDType) {sh : TileShape}
    (v : Structural.Values (WithBot ℝ) .int sh)
    (hfin : ∀ i, Float8E5Bits.isInt8 (v i) = true →
      (Float8E5Bits.ofInt (v i)).decodeRat.isSome = true) :
    Option.map toTile
      (if (TileShape.allIndices sh).all (fun i => Float8E5Bits.isInt8 (v i)) = true then
        some (fun i => Structural.ofFloat dst
          ((((Float8E5Bits.ofInt (v i)).decodeRat.getD 0 : ℚ) : ℝ) : WithBot ℝ))
      else none) = Float8E5Bits.decodeTile dst (toTile v) := by
  have hd : (toTile v).data = v := rfl
  by_cases hall : (TileShape.allIndices sh).all (fun i => Float8E5Bits.isInt8 (v i)) = true
  · have hall' : (TileShape.allIndices sh).all (fun i => Float8E5Bits.isInt8 (v i) &&
        (Float8E5Bits.ofInt (v i)).decodeRat.isSome) = true := by
      rw [List.all_eq_true] at hall ⊢
      intro i hi
      simp [hall i hi, hfin i (hall i hi)]
    rw [if_pos hall, Float8E5Bits.decodeTile, hd, if_pos hall']
    cases dst <;> rfl
  · have hall' : ¬ (TileShape.allIndices sh).all (fun i => Float8E5Bits.isInt8 (v i) &&
        (Float8E5Bits.ofInt (v i)).decodeRat.isSome) = true := by
      intro h'
      apply hall
      rw [List.all_eq_true] at h' ⊢
      intro i hi
      have := h' i hi
      simp only [Bool.and_eq_true] at this
      exact this.1
    rw [if_neg hall, Float8E5Bits.decodeTile, hd, if_neg hall']
    rfl

/-! ## Expressions -/

set_option linter.unusedSimpArgs false in
set_option maxHeartbeats 3200000 in
/-- The bridge for expressions. Under `BridgeSite s` at every node, structural
evaluation of `e` over `realAlgebra` and the proof semantics on the erased
state succeed with the same value or both fail, in every precision context. -/
theorem evalOp_bridge (s : Structural.State (WithBot ℝ)) :
    ∀ {d : TileDType} {sh : TileShape} (e : Op d sh), AllNodes (BridgeSite s) e →
      ∀ p, (SE p e s).map toTile = evalOp e (erase s)
  | _, _, .const _, _, _ | _, _, .constNat _, _, _ | _, _, .constInt _, _, _
  | _, _, .constBool _, _, _ | _, _, .negInf, _, _ | _, _, .programId _, _, _
  | _, _, .numPrograms _, _, _ | _, _, .arange _, _, _ | _, _, .ref _ _ _, _, _
  | _, _, .ptrBase _, _, _ | _, _, .makeBlockPtr _ _ _ _ _ _, _, _ => by
      rw [Structural.evalOp_unfold]; simp only [evalOp]; rfl
  | _, _, .constFloat f _, _, _ => by
      rw [Structural.evalOp_unfold]; simp only [evalOp]; cases f <;> rfl
  | _, _, .exp a, h, p | _, _, .libdeviceExp a, h, p | _, _, .libdeviceLog a, h, p
  | _, _, .libdeviceExpm1 a, h, p | _, _, .libdeviceLog1p a, h, p
  | _, _, .libdeviceTanh a, h, p | _, _, .exp2 a, h, p | _, _, .log a, h, p
  | _, _, .log2 a, h, p | _, _, .sigmoid a, h, p | _, _, .sqrt a, h, p | _, _, .rsqrt a, h, p
  | _, _, .libdeviceRsqrt a, h, p | _, _, .tanh a, h, p | _, _, .sin a, h, p
  | _, _, .cos a, h, p | _, _, .tan a, h, p | _, _, .atan a, h, p | _, _, .cosh a, h, p
  | _, _, .sinh a, h, p | _, _, .erf a, h, p | _, _, .minPosInf a, h, p
  | _, _, .natToReal a, h, p | _, _, .intToReal a, h, p | _, _, .castNatToInt a, h, p
  | _, _, .castIntToNat a, h, p | _, _, .boolNot a, h, p | _, _, .broadcast a _, h, p
  | _, _, .full _ a, h, p | _, _, .reduceSum _ _ a, h, p | _, _, .reduceSumBool _ _ a, h, p
  | _, _, .reduceMaxInt _ _ a, h, p | _, _, .scan _ _ _ a, h, p | _, _, .argMaxInt _ a, h, p
  | _, _, .remap _ _ a, h, p | _, _, .split _ a, h, p | _, _, .expandDim _ a, h, p
  | _, _, .makeBlockPtrDyn _ a _ _ _ _, h, p | _, _, .advanceBlockPtr a _, h, p => by
      simp only [AllNodes] at h
      rw [Structural.evalOp_unfold]; simp only [evalOp]; rw [← evalOp_bridge s a h.2 p]
      cases SE p a s <;> rfl
  | _, _, .castFloat w w' a, h, p => by
      simp only [AllNodes] at h
      rw [Structural.evalOp_unfold]; simp only [evalOp]; rw [← evalOp_bridge s a h.2 p]
      cases w <;> cases w' <;> (cases SE p a s <;> rfl)
  | _, _, .negFloat w a, h, p | _, _, .absFloat w a, h, p => by
      simp only [AllNodes] at h
      rw [Structural.evalOp_unfold]; simp only [evalOp]; rw [← evalOp_bridge s a h.2 p]
      cases w <;> (cases SE p a s <;> rfl)
  | _, _, .withPrecision _ _ a, h, _ => by
      simp only [AllNodes] at h
      rw [Structural.evalOp_unfold]; simp only [evalOp]
      exact evalOp_bridge s a h.2 _
  | _, _, .floatToInt .., _, _ | _, _, .floatToNat .., _, _ => by
      rw [Structural.evalOp_unfold]; simp only [evalOp]; rfl
  | _, _, .castRealToInt8 _, h, _ | _, _, .argMin _ _, h, _ | _, _, .sort _ _, h, _ => by
      simp only [AllNodes, BridgeSite, StructuralNode] at h
      exact h.1.1.elim
  | _, _, .divRN _ a b, h, p | _, _, .max2 _ a b, h, p | _, _, .pow _ a b, h, p
  | _, _, .bitAnd _ a b, h, p | _, _, .bitOr _ a b, h, p | _, _, .bitXor _ a b, h, p
  | _, _, .shiftLeft _ a b, h, p | _, _, .shiftRight _ a b, h, p | _, _, .boolAnd _ a b, h, p
  | _, _, .boolOr _ a b, h, p | _, _, .ptrAdd _ a b, h, p | _, _, .ptrSub _ a b, h, p
  | _, _, .dotInt a b _, h, p => by
      simp only [AllNodes] at h
      rw [Structural.evalOp_unfold]; simp only [evalOp]
      rw [← evalOp_bridge s a h.2.1 p, ← evalOp_bridge s b h.2.2 p]
      cases SE p a s <;> cases SE p b s <;> rfl
  | _, _, .add w _ a b, h, p | _, _, .sub w _ a b, h, p | _, _, .mul w _ a b, h, p
  | _, _, .div w _ a b, h, p | _, _, .floorDiv w _ a b, h, p | _, _, .mod w _ a b, h, p
  | _, _, .maximumFloat w _ a b, h, p | _, _, .minimumFloat w _ a b, h, p => by
      simp only [AllNodes] at h
      rw [Structural.evalOp_unfold]; simp only [evalOp]
      rw [← evalOp_bridge s a h.2.1 p, ← evalOp_bridge s b h.2.2 p]
      cases w <;> (cases SE p a s <;> cases SE p b s <;> rfl)
  | _, _, .lt w _ a b, h, p | _, _, .le w _ a b, h, p | _, _, .eq w _ a b, h, p
  | _, _, .gt w _ a b, h, p | _, _, .ge w _ a b, h, p | _, _, .ne w _ a b, h, p => by
      simp only [AllNodes] at h
      rw [Structural.evalOp_unfold]; simp only [evalOp]
      rw [← evalOp_bridge s a h.2.1 p, ← evalOp_bridge s b h.2.2 p]
      cases w <;> (cases SE p a s <;> cases SE p b s <;> first
        | rfl
        | (simp [Structural.numericLt, Structural.numericLe, Structural.numericEq,
            Structural.natLt, Structural.natLe, realAlgebra, toTile, toCarrier, Tile.cop,
            Structural.bop, ComparableDType.lt, ComparableDType.le, ComparableDType.eq,
            ComparableDType.gt, ComparableDType.ge, ComparableDType.ne]))
  | _, _, .libdevicePowi _ a b, h, p => by
      simp only [AllNodes] at h
      rw [Structural.evalOp_unfold]; simp only [evalOp]
      rw [← evalOp_bridge s a h.2.1 p, ← evalOp_bridge s b h.2.2 p]
      cases SE p a s <;> cases hy : SE p b s <;> try rfl
      rename_i va vb
      by_cases hv : Int32Bits.valuesValid vb = true
      · have : Int32Bits.valuesValid (toTile vb).data = true := hv
        simp [hv, this, realAlgebra]; rfl
      · have : ¬ Int32Bits.valuesValid (toTile vb).data = true := hv
        simp [hv, this, realAlgebra]
  | _, _, .where c a b, h, p => by
      simp only [AllNodes] at h
      rw [Structural.evalOp_unfold]; simp only [evalOp]
      rw [← evalOp_bridge s c h.2.1 p, ← evalOp_bridge s a h.2.2.1 p,
        ← evalOp_bridge s b h.2.2.2 p]
      cases SE p c s <;> cases SE p a s <;> cases SE p b s <;> try rfl
      rename_i vc vx vy
      refine congrArg some (Tile.ext fun i => ?_)
      show toCarrier _ (if vc i = true then vx i else vy i) =
        if vc i = true then toCarrier _ (vx i) else toCarrier _ (vy i)
      split <;> rfl
  | _, _, .ite c a b, h, p => by
      simp only [AllNodes] at h
      rw [Structural.evalOp_unfold]; simp only [evalOp]
      rw [← evalOp_bridge s c h.2.1 p, ← evalOp_bridge s a h.2.2.1 p,
        ← evalOp_bridge s b h.2.2.2 p]
      cases SE p c s <;> try rfl
      rename_i vc
      show Option.map toTile (if vc PUnit.unit = true then _ else _) =
        if vc PUnit.unit = true then _ else _
      split <;> rfl
  | _, _, .reduceMin (shape := shape) axis keepDims a, h, p => by
      simp only [AllNodes] at h
      rw [Structural.evalOp_unfold]; simp only [evalOp]; rw [← evalOp_bridge s a h.2 p]
      cases SE p a s <;> try rfl
      by_cases h0 : 0 < shape.axisDim axis
      · simp [h0, realAlgebra, map_toTile_ofTile]
      · cases keepDims <;> simp [h0, Tile.reduceMin, Tile.reduceMinDrop, Tile.reduceMinKeep]
  | _, _, .reduceMax (shape := shape) axis keepDims a, h, p => by
      simp only [AllNodes] at h
      rw [Structural.evalOp_unfold]; simp only [evalOp]; rw [← evalOp_bridge s a h.2 p]
      cases SE p a s <;> try rfl
      by_cases h0 : 0 < shape.axisDim axis
      · cases keepDims <;>
          simp [h0, realAlgebra, Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceMaxKeep]
      · cases keepDims <;> simp [h0, Tile.reduceMax, Tile.reduceMaxDrop, Tile.reduceMaxKeep]
  | _, _, .reduceMaxNat _ keepDims a, h, p => by
      simp only [AllNodes] at h
      rw [Structural.evalOp_unfold]; simp only [evalOp]; rw [← evalOp_bridge s a h.2 p]
      cases SE p a s <;> try rfl
      cases keepDims <;>
        simp only [Option.bind_eq_bind, Option.bind_some, Option.map_some,
          TileShape.discreteMaximum, Tile.reduceMaxNat, Tile.reduceMaxNatDrop,
          Tile.reduceMaxNatKeep] <;>
        split <;> rfl
  | _, _, .reduceSumTyped _ _ _ a, h, p => by
      simp only [AllNodes] at h
      rw [Structural.evalOp_unfold]; simp only [evalOp]; rw [← evalOp_bridge s a h.2 p]
      cases SE p a s <;> try rfl
      simp [realAlgebra]
  | _, _, .reduceMaxTyped (shape := shape) _ axis _ a, h, p => by
      simp only [AllNodes] at h
      rw [Structural.evalOp_unfold]; simp only [evalOp]; rw [← evalOp_bridge s a h.2 p]
      cases SE p a s <;> try rfl
      by_cases h0 : 0 < shape.axisDim axis
      · simp [h0, realAlgebra, map_toTile_ofTile]
      · have h0' : shape.axisDim axis = 0 := by omega
        simp [h0, Tile.reduceMaxTyped_eq_none _ _ _ _ h0']
  | _, _, .indexedMaxValue (shape := shape) f axis a, h, p => by
      simp only [AllNodes] at h
      rw [Structural.evalOp_unfold]; simp only [evalOp]; rw [← evalOp_bridge s a h.2 p]
      cases SE p a s <;> try rfl
      rename_i v
      by_cases h0 : shape.axisDim axis = 0
      · simp [h0, Tile.reduceMaxTyped_eq_none _ _ _ _ h0]
      · obtain ⟨m, hm⟩ := reduceMaxTyped_isSome f axis (toTile v) h0
        simp [h0, realAlgebra, hm]
  | _, _, .argMax axis a, h, p => by
      simp only [AllNodes, BridgeSite, NonEmptyArgMax] at h
      have hne := h.1.2.1
      rw [Structural.evalOp_unfold]; simp only [evalOp]; rw [← evalOp_bridge s a h.2 p]
      cases SE p a s <;> try rfl
      rename_i v
      obtain ⟨m, hm⟩ := reduceMaxTyped_isSome .real axis (toTile v) hne
      simp [hne, realAlgebra, hm]
      exact ⟨⟨m, hm⟩, rfl⟩
  | _, _, .argMaxTyped f axis a precision, h, p => by
      simp only [AllNodes, BridgeSite, NonEmptyArgMax] at h
      have hne := h.1.2.1
      have ih := evalOp_bridge s a h.2
      rw [Structural.evalOp_unfold]; simp only [evalOp]
      rcases precision with _ | _ | _ <;>
        [rw [← ih p]; rw [← ih (some .fp32)]; rw [← ih (some .fp64)]] <;>
        (generalize Structural.evalOp realAlgebra _ a s = x
         cases x with
         | none => rfl
         | some v =>
           obtain ⟨m, hm⟩ := reduceMaxTyped_isSome f axis (toTile v) hne
           simp [hne, realAlgebra, hm]; rfl)
  | _, _, .dot a b options, h, p => by
      simp only [AllNodes] at h
      rw [Structural.evalOp_unfold]; simp only [evalOp]
      rw [← evalOp_bridge s a h.2.1 p, ← evalOp_bridge s b h.2.2 p]
      cases SE p a s <;> cases SE p b s <;> try rfl
      by_cases ho : options = {} <;> simp [ho, realAlgebra]
  | _, _, .dotTyped l r o a b _, h, p => by
      simp only [AllNodes] at h
      rw [Structural.evalOp_unfold]; simp only [evalOp]
      rw [← evalOp_bridge s a h.2.1 p, ← evalOp_bridge s b h.2.2 p]
      cases SE p a s <;> cases SE p b s <;> try rfl
      cases l <;> cases r <;> cases o <;> rfl
  | _, _, .dotAcc l r o a b c _, h, p => by
      simp only [AllNodes] at h
      rw [Structural.evalOp_unfold]; simp only [evalOp]
      rw [← evalOp_bridge s a h.2.1 p, ← evalOp_bridge s b h.2.2.1 p,
        ← evalOp_bridge s c h.2.2.2 p]
      cases SE p a s <;> cases SE p b s <;> cases SE p c s <;> try rfl
      cases l <;> cases r <;> cases o <;> rfl
  | _, _, .join a b, h, p => by
      simp only [AllNodes] at h
      rw [Structural.evalOp_unfold]; simp only [evalOp]
      rw [← evalOp_bridge s a h.2.1 p, ← evalOp_bridge s b h.2.2 p]
      cases SE p a s <;> cases SE p b s <;> try rfl
      refine congrArg some (Tile.ext fun i => ?_)
      simp only [toTile, Tile.join]
      split <;> rfl
  | _, _, .transpose (batch := batch) a, h, p => by
      simp only [AllNodes] at h
      rw [Structural.evalOp_unfold]; simp only [evalOp]; rw [← evalOp_bridge s a h.2 p]
      cases SE p a s <;> try rfl
      exact congrArg some (toTile_transpose batch _)
  | _, _, .reshape _ a, h, p => by
      simp only [AllNodes, BridgeSite, ReshapeSameSize] at h
      have hsize := h.1.2.2.1
      rw [Structural.evalOp_unfold]; simp only [evalOp]; rw [← evalOp_bridge s a h.2 p]
      cases SE p a s <;> try rfl
      simp only [hsize, dite_true, Option.bind_some, Option.map_some]
      exact congrArg some (toTile_reshape hsize _)
  | _, _, .decodeFloat8E5 dst a, h, p => by
      simp only [AllNodes, BridgeSite, DecodesFinite] at h
      have hfin := h.1.2.2.2.1
      rw [Structural.evalOp_unfold]; simp only [evalOp]; rw [← evalOp_bridge s a h.2 p]
      cases hv : SE p a s with
      | none => rfl
      | some v => exact decode_bridge dst v (hfin p v hv)
  | _, _, .uniform32 a, h, p => by
      simp only [AllNodes] at h
      rw [Structural.evalOp_unfold]; simp only [evalOp]; rw [← evalOp_bridge s a h.2 p]
      cases SE p a s <;> try rfl
      rename_i v
      have hd : (toTile v).data = v := rfl
      by_cases hv : Random.Direct.uniformInputsValid v = true
      · simp [hv, hd, fp32Bits_of_toRat _ _ (Random.uniform_toRat_uniformRat _), Random.uniformReal]
        rfl
      · simp [hv, hd]
  | _, _, .rand cfg a b, h, p => by
      simp only [AllNodes] at h
      rw [Structural.evalOp_unfold]; simp only [evalOp]
      rw [← evalOp_bridge s a h.2.1 p, ← evalOp_bridge s b h.2.2 p]
      cases SE p a s <;> cases SE p b s <;> try rfl
      rename_i vs vo
      have hds : (toTile vs).data = vs := rfl
      have hdo : (toTile vo).data = vo := rfl
      by_cases hv : cfg.inputsValid (vs PUnit.unit) vo = true
      · simp [hv, hds, hdo, fp32Bits_of_toRat _ _ (Random.rand_toRat_randRat _ _ _ _),
          Random.randReal]
        rfl
      · simp [hv, hds, hdo]
  | _, _, .randTensor cfg a b, h, p => by
      simp only [AllNodes] at h
      rw [Structural.evalOp_unfold]; simp only [evalOp]
      rw [← evalOp_bridge s a h.2.1 p, ← evalOp_bridge s b h.2.2 p]
      cases SE p a s <;> cases SE p b s <;> try rfl
      rename_i vs vo
      have hds : (toTile vs).data = vs := rfl
      have hdo : (toTile vo).data = vo := rfl
      by_cases hv : cfg.tensorInputsValid vs vo = true
      · simp [hv, hds, hdo, fp32Bits_of_toRat _ _ (Random.rand_toRat_randRat _ _ _ _),
          Random.randReal]
        rfl
      · simp [hv, hds, hdo]
  | _, _, .philox4x32 (outs := outs) cfg plan field seed c0 c1 c2 c3, h, p => by
      simp only [AllNodes] at h
      rw [Structural.evalOp_unfold]; simp only [evalOp]
      rw [← evalOp_bridge s seed h.2.1 p, ← evalOp_bridge s c0 h.2.2.1 p,
        ← evalOp_bridge s c1 h.2.2.2.1 p, ← evalOp_bridge s c2 h.2.2.2.2.1 p,
        ← evalOp_bridge s c3 h.2.2.2.2.2 p]
      cases SE p seed s <;> cases SE p c0 s <;> cases SE p c1 s <;> cases SE p c2 s <;>
        cases SE p c3 s <;> try rfl
      rename_i vs v0 v1 v2 v3
      have key : ∀ x : Option (TileIndex (outs.at field) → Nat),
          Option.map toTile (x.bind fun r => some r) =
            x.bind fun r => some (⟨r⟩ : Tile .nat (outs.at field)) := by
        intro x; cases x <;> rfl
      exact key _
  | _, _, .makeBlockPtrDynOffsets _ a _ _ _ offsets, h, p => by
      simp only [AllNodes] at h
      rw [Structural.evalOp_unfold]; simp only [evalOp]
      rw [← evalOp_bridge s a h.2.1 p,
        ← evalOffsets_bridge s p offsets fun x hx => evalOp_bridge s x (h.2.2 x hx)]
      cases SE p a s <;> try rfl
      cases Structural.evalOffsets realAlgebra p offsets s <;> rfl
  | _, _, .makeBlockPtrDynMetadata _ a _ _ strides offsets, h, p => by
      simp only [AllNodes] at h
      rw [Structural.evalOp_unfold]; simp only [evalOp]
      rw [← evalOp_bridge s a h.2.1 p,
        ← evalOffsets_bridge s p strides fun x hx => evalOp_bridge s x (h.2.2.1 x hx),
        ← evalOffsets_bridge s p offsets fun x hx => evalOp_bridge s x (h.2.2.2 x hx)]
      cases SE p a s <;> try rfl
      cases Structural.evalOffsets realAlgebra p strides s <;> try rfl
      cases Structural.evalOffsets realAlgebra p offsets s <;> rfl
  | _, _, .advanceBlockPtrDyn a deltas, h, p => by
      simp only [AllNodes] at h
      rw [Structural.evalOp_unfold]; simp only [evalOp]
      rw [← evalOp_bridge s a h.2.1 p,
        ← evalDeltas_bridge s p deltas fun x hx => evalOp_bridge s x (h.2.2 x hx)]
      cases SE p a s <;> try rfl
      cases Structural.evalDeltas realAlgebra p deltas s <;> rfl
  | _, _, .load _ (.region _ a) .none padding, h, p
  | _, _, .load _ (.ptr a) .none padding, h, p
  | _, _, .load _ (.blockPtr a _) .none padding, h, p => by
      simp only [AllNodes] at h
      exact load_bridge s _ _ padding p (evalOp_bridge s a h.2.1) trivial h.1.2.2.2.2
  | _, _, .load _ (.region _ a) (.mask m) padding, h, p
  | _, _, .load _ (.ptr a) (.mask m) padding, h, p
  | _, _, .load _ (.blockPtr a _) (.mask m) padding, h, p => by
      simp only [AllNodes] at h
      exact load_bridge s _ _ padding p (evalOp_bridge s a h.2.1) (evalOp_bridge s m h.2.2)
        h.1.2.2.2.2
  | _, _, .load _ (.region _ a) (.maskOther m other) padding, h, p
  | _, _, .load _ (.ptr a) (.maskOther m other) padding, h, p
  | _, _, .load _ (.blockPtr a _) (.maskOther m other) padding, h, p => by
      simp only [AllNodes] at h
      exact load_bridge s _ _ padding p (evalOp_bridge s a h.2.1)
        ⟨evalOp_bridge s m h.2.2.1, evalOp_bridge s other h.2.2.2⟩ h.1.2.2.2.2
termination_by _ _ e _ _ => sizeOf e
decreasing_by
  all_goals
    simp_wf
    try omega
    try
      have hs := List.sizeOf_lt_of_mem (by assumption)
      omega

/-! ## Compute operations and expressions -/

/-- D9, D10. Structural execution evaluates this compute operation: it has no
`fp64` constant and no `fp64` load, and every bitcast goes from `int32` or
`uint32` to `fp32`. On the others it fails, and the projection of an `fp64`
constant, of an `fp64` load or of another bitcast of a constant can succeed. -/
def SupportedCompute : {cd : ComputeDType} → {sh : TileShape} → ComputeOp cd sh → Prop
  | _, _, .alg _ _ => True
  | .fp64, _, .const _ => False
  | _, _, .const _ => True
  | _, _, .full _ e => SupportedCompute e
  | _, _, .load .fp64 _ _ _ => False
  | _, _, .load _ _ _ _ => True
  | _, _, .bitcast src dst _ e => (src = .int32 ∨ src = .uint32) ∧ dst = .fp32 ∧ SupportedCompute e

/-- `SupportedCompute` at every compute operation of an expression. -/
def SupportedExpr : {d : TileDType} → {sh : TileShape} → ComputeExpr d sh → Prop
  | _, _, .compute c => SupportedCompute c
  | _, _, .fma _ _ _ a b c | _, _, .fmaTyped _ _ _ a b c =>
      SupportedExpr a ∧ SupportedExpr b ∧ SupportedExpr c
  | _, _, _ => True

private theorem ofInt_toInt (c : Int32Bits) : BitVec.ofInt 32 c.toInt = c.bits := by
  have h : c.toInt = c.bits.toInt := by
    simp only [Int32Bits.toInt, BitVec.toInt, Int.ofNat_eq_natCast]
    split <;> split <;> omega
  rw [h, BitVec.ofInt_toInt]

/-- The bridge for a compute operation whose projection succeeds. -/
theorem evalComputeOp_bridge (s : Structural.State (WithBot ℝ)) :
    ∀ {cd : ComputeDType} {sh : TileShape} (c : ComputeOp cd sh) {op : Op cd.eraseDType sh},
      c.toAlgorithm? = .ok op → SupportedCompute c → AllNodes (BridgeSite s) op →
      (Structural.evalComputeOp realAlgebra c s).map toTile = evalOp op (erase s)
  | _, _, .alg d e, _, h, _, hn => by
      cases h
      exact evalOp_bridge s e hn (some d)
  | .fp64, _, .const _, _, _, hs, _ => by simp only [SupportedCompute] at hs
  | .fp32, _, .const c, _, h, _, _ => by
      simp only [ComputeOp.toAlgorithm?, ComputeOp.constToAlgorithm?] at h
      split at h <;> cases h
      show _ = evalOp (dtype := .real) (Op.const _) (erase s)
      simp [Structural.evalComputeOp, realAlgebra, Float32Bits.decodeReal, *]; rfl
  | .int32, _, .const c, _, h, _, _ => by
      cases h
      show _ = evalOp (dtype := .int) (Op.constInt _) (erase s)
      simp only [Structural.evalComputeOp, evalOp]; rfl
  | .uint32, _, .const c, _, h, _, _ => by
      cases h
      show _ = evalOp (dtype := .nat) (Op.constNat _) (erase s)
      simp only [Structural.evalComputeOp, evalOp]; rfl
  | _, _, .full sh e, _, h, hs, hn => by
      simp only [ComputeOp.toAlgorithm?] at h
      split at h <;> cases h
      rename_i e' he
      simp only [SupportedCompute] at hs
      simp only [AllNodes] at hn
      simp only [Structural.evalComputeOp, evalOp]
      rw [← evalComputeOp_bridge s e he hs hn.2]
      cases Structural.evalComputeOp realAlgebra e s <;> rfl
  | _, _, .load .fp64 _ _ _, _, _, hs, _ => by simp only [SupportedCompute] at hs
  | _, _, .load .fp32 mem mask padding, _, h, _, hn => by
      cases h
      refine Eq.trans ?_ (evalOp_bridge s _ hn (some .fp32))
      show Option.map toTile ((SE (some .fp32) (Op.load .real mem mask padding) s).bind
        fun v => some fun i => realAlgebra.fp32Load (v i)) =
        Option.map toTile (SE (some .fp32) (Op.load .real mem mask padding) s)
      cases SE (some ComputeDType.fp32) (Op.load .real mem mask padding) s <;> rfl
  | _, _, .load .int32 mem mask padding, _, h, _, hn => by
      cases h
      exact evalOp_bridge s _ hn (some .int32)
  | _, _, .load .uint32 mem mask padding, _, h, _, hn => by
      cases h
      exact evalOp_bridge s _ hn (some .uint32)
  | _, sh, .bitcast src dst _ e, _, h, hs, _ => by
      simp only [SupportedCompute] at hs
      obtain ⟨hsrc, rfl, he⟩ := hs
      rcases hsrc with rfl | rfl <;> cases e with
        | const c =>
          simp only [ComputeOp.toAlgorithm?, ComputeOp.constOpToAlgorithm?,
            ComputeOp.constPayload?, ComputeOp.bitcastPayload, ComputeOp.constToAlgorithm?] at h
          split at h <;> cases h
          show _ = evalOp (dtype := .real) (Op.const _) (erase s)
          simp [Structural.evalComputeOp, realAlgebra, Float32Bits.decodeReal, ofInt_toInt,
            UInt32Bits.toNat, *]
          rfl
        | bitcast _ _ _ _ => simp [SupportedCompute] at he
        | _ =>
          cases sh <;> simp [ComputeOp.toAlgorithm?, ComputeOp.constOpToAlgorithm?,
            ComputeOp.constPayload?] at h

/-- The bridge for a compute expression whose projection succeeds. -/
theorem evalExpr_bridge (s : Structural.State (WithBot ℝ)) :
    ∀ {d : TileDType} {sh : TileShape} (e : ComputeExpr d sh) {op : Op d sh},
      e.toAlgorithm? = .ok op → SupportedExpr e → AllNodes (BridgeSite s) op →
      (Structural.evalExpr realAlgebra e s).map toTile = evalOp op (erase s)
  | _, _, .alg e, _, h, _, hn => by
      cases h
      exact evalOp_bridge s e hn none
  | _, _, .compute c, _, h, hs, hn => evalComputeOp_bridge s c h hs hn
  | _, _, .opaque _, _, h, _, _ | _, _, .llrint _, _, h, _, _
  | _, _, .isFinite _ _, _, h, _, _ => by
      cases h
  | _, _, .fma _ bcAB bcC a b c, _, h, hs, hn | _, _, .fmaTyped _ bcAB bcC a b c, _, h, hs, hn => by
      simp only [ComputeExpr.toAlgorithm?] at h
      cases ha : a.toAlgorithm? with
      | error _ => simp [ha, bind, Except.bind] at h
      | ok a' =>
        cases hb : b.toAlgorithm? with
        | error _ => simp [ha, hb, bind, Except.bind] at h
        | ok b' =>
          cases hc : c.toAlgorithm? with
          | error _ => simp [ha, hb, hc, bind, Except.bind] at h
          | ok c' =>
            simp only [ha, hb, hc, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h
            subst h
            simp only [SupportedExpr] at hs
            simp only [AllNodes] at hn
            simp only [Structural.evalExpr, evalOp]
            rw [← evalExpr_bridge s a ha hs.1 hn.2.1.2.1,
              ← evalExpr_bridge s b hb hs.2.1 hn.2.1.2.2, ← evalExpr_bridge s c hc hs.2.2 hn.2.2]
            cases Structural.evalExpr realAlgebra a s <;>
              cases Structural.evalExpr realAlgebra b s <;>
              cases Structural.evalExpr realAlgebra c s <;>
              first | rfl | (rename FloatDType => f; cases f <;> rfl)

/-! ## Examples -/

/-- A state whose every cell is `c`, with no register and default undefined values. -/
private def cellState (c : Structural.Cell (WithBot ℝ)) : Structural.State (WithBot ℝ) where
  mem _ _ := c
  regs _ _ _ := none
  pids _ := 0
  numPids _ := 1
  undef d _ _ := ofCarrier d (BlockState.defaultCarrier d)

/-- An unmasked `.real` load of cell `0` of region `x`. -/
private def loadX : Op .real [] := .load .real (.region ⟨"x"⟩ (.constNat 0)) .none

private theorem loadX_structural (c : Structural.Cell (WithBot ℝ)) :
    SE none loadX (cellState c) = some fun _ => c.read .real := by
  rw [loadX, Structural.evalOp_unfold]
  simp only []
  rw [Structural.evalOp_unfold]
  rfl

private theorem loadX_proof (c : Structural.Cell (WithBot ℝ)) :
    evalOp loadX (erase (cellState c)) =
      some ⟨fun _ => (erase (cellState c)).readMemValue .real "x" 0⟩ := by
  simp [loadX]

/-- Negative example of `evalOp_bridge` (D1). A `.real` cell that holds `⊥` is
not `CellFinite`: the load reads `⊥` on the structural side and `0` on the
proof side. -/
example : ¬ CellFinite .real (.mk .real (⊥ : WithBot ℝ)) ∧
    (SE none loadX (cellState (.mk .real (⊥ : WithBot ℝ)))).map toTile ≠
      evalOp loadX (erase (cellState (.mk .real (⊥ : WithBot ℝ)))) := by
  constructor
  · intro h
    exact absurd (h ⊥ rfl) (by simp [storedValue, toCarrier])
  · rw [loadX_structural, loadX_proof]
    intro h
    have := congrArg (fun t => t.map fun x => x.data PUnit.unit) h
    simp [toTile, toCarrier, BlockState.readMemValue, BlockState.readMemAs, cellState, erase,
      eraseCell, storedValue, MemCell.of, MemCell.readAs, FloatDType.ofReal, FloatDType.ofWithBot,
      FloatDType.storeValue, FloatDType.toWithBot] at this

/-- Negative example of `evalOp_bridge` (D13). A cell tagged `.fp32` is not
`CellTagged .real`: the `.real` load reads the carrier default `⊥` on the
structural side and `0` on the proof side. -/
example : ¬ CellTagged .real (.mk .fp32 ((1 : ℝ) : WithBot ℝ)) ∧
    (SE none loadX (cellState (.mk .fp32 ((1 : ℝ) : WithBot ℝ)))).map toTile ≠
      evalOp loadX (erase (cellState (.mk .fp32 ((1 : ℝ) : WithBot ℝ)))) := by
  constructor
  · intro h
    obtain ⟨v, hv⟩ := h ⟨.real, rfl⟩
    cases hv
  · rw [loadX_structural, loadX_proof]
    intro h
    have := congrArg (fun t => t.map fun x => x.data PUnit.unit) h
    simp [toTile, toCarrier, BlockState.readMemValue, BlockState.readMemAs, cellState, erase,
      eraseCell, storedValue, MemCell.of, MemCell.readAs, FloatDType.ofReal, FloatDType.ofWithBot,
      Structural.Cell.read, Structural.defaultValue] at this

/-- The preconditions hold for the load of a finite `.real` cell. -/
example : AllNodes (BridgeSite (cellState (.mk .real ((1 : ℝ) : WithBot ℝ)))) loadX := by
  simp only [loadX, AllNodes, BridgeSite, StructuralNode, NonEmptyArgMax, ReshapeSameSize,
    DecodesFinite, LoadOk, and_true, true_and]
  intro p addr active ha _ i
  simp only [laneAddrs] at ha
  rw [Structural.evalOp_unfold] at ha
  simp only [Option.map_some, Option.some.injEq] at ha
  subst ha
  refine ⟨fun _ _ => ⟨fun v hv => ?_, fun _ => ⟨_, rfl⟩⟩, fun _ hk => ?_, fun _ hm => ?_⟩
  · cases hv; rfl
  · cases hk
  · obtain ⟨m, hm⟩ := hm
    cases hm

/-- The same state satisfies D8 at every dtype and address, masked or not: its
undefined values are `0` on the floating dtypes and the default on the others. -/
example (c : Structural.Cell (WithBot ℝ)) (d : TileDType) (masked : Bool) (r : RegionName)
    (o : Nat) : UndefAgrees (cellState c) d masked r o := by
  cases d <;> cases masked <;>
    simp [UndefAgrees, cellState, erase, BlockState.unspecifiedValue, BlockState.defaultCarrier,
      toCarrier, ofCarrier, FloatDType.ofReal, FloatDType.ofWithBot]

/-- The `fp64` encoding of `1`. -/
private def one64 : ComputeOp .fp64 [] := .const (⟨0x3FF0000000000000#64⟩ : Float64Bits)

/-- Negative example of `evalComputeOp_bridge` (D9). An `fp64` constant is not
`SupportedCompute`: its projection succeeds, structural execution fails and
the proof side returns the constant. -/
example (s : Structural.State (WithBot ℝ)) : ¬ SupportedCompute one64 ∧
    ∃ op, one64.toAlgorithm? = .ok op ∧
      (Structural.evalComputeOp realAlgebra one64 s).map toTile ≠ evalOp op (erase s) := by
  refine ⟨fun h => by simp [one64, SupportedCompute] at h, ?_⟩
  obtain ⟨q, hq⟩ : ∃ q, Float64Bits.decodeRat ⟨0x3FF0000000000000#64⟩ = some q := ⟨_, rfl⟩
  refine ⟨Op.const (q : ℝ), ?_, ?_⟩
  · simp [one64, ComputeOp.toAlgorithm?, ComputeOp.constToAlgorithm?, hq]
  · show _ ≠ evalOp (dtype := .real) (Op.const (q : ℝ)) (erase s)
    simp [one64, Structural.evalComputeOp]

/-- Negative example of `evalExpr_bridge` (D9). An `fp64` load is not
`SupportedExpr`: it projects to a `.real` load, on which the proof side
succeeds, and structural execution fails. -/
example (s : Structural.State (WithBot ℝ)) :
    let e : ComputeExpr .real [] :=
      .compute (.load .fp64 (.region ⟨"x"⟩ (.constNat 0)) .none)
    ¬ SupportedExpr e ∧ e.toAlgorithm? = .ok loadX ∧
      (Structural.evalExpr realAlgebra e s).map toTile ≠ evalOp loadX (erase s) := by
  refine ⟨fun h => by simp [SupportedExpr, SupportedCompute] at h, rfl, ?_⟩
  simp [Structural.evalExpr, Structural.evalComputeOp, loadX]

end VeriTile.Interp
