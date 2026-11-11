/-
VeriTile.Interp.Mirror

Mirror of the upstream kernel AST (`VeriTile/Triton/Core/Ast.lean`) whose
floating literals are rationals, and the `toReal` maps back into the upstream
types. Upstream carries a real number in exactly two places, `Op.const` and
`Op.constFloat`; the mirror carries a rational there and is otherwise the
upstream declaration with the type names replaced:

| Upstream | Mirror |
| --- | --- |
| `Op`, `MemAccess`, `MaskOpt` | `QOp`, `QMemAccess`, `QMaskOpt` |
| `Stmt` | `QAlgStmt` |
| `ComputeOp` | `QComputeOp` |
| `ComputeExpr` | `QExpr` |
| `ComputeStmt` | `QStmt` |
| `ComputeKernel` | `QKernel` |

`AlgDType` is written as the `TileDType` it abbreviates. The command at the end
of the file fails the build when a mirror stops matching its upstream type.
-/
import VeriTile.Triton.Core.Ast

namespace VeriTile.Interp

open VeriTile.Triton

/-! ## Mirror types -/

mutual

/-- Mirror of `Op`: the same constructors, except that `const` and `constFloat` carry a
rational instead of a real. -/
inductive QOp : TileDType → TileShape → Type where
  | const     : ℚ → QOp .real []
  | constFloat : (dtype : FloatDType) → ℚ → QOp dtype.toTileDType []
  | constNat  : Nat → QOp .nat []
  | constInt  : Int → QOp .int []
  | constBool : Bool → QOp .bool []
  | negInf    : QOp .real []
  | programId : (axis : Nat) → QOp .nat []
  | numPrograms : (axis : Nat) → QOp .nat []
  | ref       : (dtype : TileDType) → (shape : TileShape) → RegName → QOp dtype shape
  | arange    : (n : Nat) → QOp .nat [n]
  | rand : Random.SourceProfile → QOp .int [] → QOp .int shape → QOp .real shape
  | randTensor : Random.SourceProfile → QOp .int shape → QOp .int shape → QOp .real shape
  | philox4x32 {ins outs : Random.Direct.Shapes} {rounds : Nat} :
      Random.Direct.Profile → Random.Direct.Plan ins rounds outs →
      (field : Random.Direct.Field) → QOp .int [] →
      QOp .int ins.c0 → QOp .int ins.c1 → QOp .int ins.c2 → QOp .int ins.c3 →
      QOp .nat (outs.at field)
  | uniform32 : QOp .nat shape → QOp .real shape
  | broadcast : QOp dtype [] → (shape : TileShape) → QOp dtype shape
  | full      : (shape : TileShape) → QOp dtype [] → QOp dtype shape
  | withPrecision : (dtype : FloatDType) → IndexComputePrecision →
      QOp dtype.toTileDType shape → QOp dtype.toTileDType shape
  | castFloat : (src dst : FloatDType) → QOp src.toTileDType shape → QOp dst.toTileDType shape
  | negFloat : (dtype : FloatDType) → QOp dtype.toTileDType shape → QOp dtype.toTileDType shape
  | absFloat : (dtype : FloatDType) → QOp dtype.toTileDType shape → QOp dtype.toTileDType shape
  | floatToInt : Option IndexComputePrecision → (src : FloatDType) → Nat →
      QOp src.toTileDType shape → QOp .int shape
  | floatToNat : Option IndexComputePrecision → (src : FloatDType) → Nat →
      QOp src.toTileDType shape → QOp .nat shape
  | castNatToInt : QOp .nat shape → QOp .int shape
  | castIntToNat : QOp .int shape → QOp .nat shape
  | castRealToInt8 : QOp .real shape → QOp .int shape
  | decodeFloat8E5 : (dst : FloatDType) → QOp .int shape → QOp dst.toTileDType shape
  | add       : NumericDType dtype → Broadcast a b out →
      QOp dtype a → QOp dtype b → QOp dtype out
  | sub       : NumericDType dtype → Broadcast a b out →
      QOp dtype a → QOp dtype b → QOp dtype out
  | mul       : NumericDType dtype → Broadcast a b out →
      QOp dtype a → QOp dtype b → QOp dtype out
  | div       : NumericDType dtype → Broadcast a b out →
      QOp dtype a → QOp dtype b → QOp dtype out
  | divRN : Broadcast a b out → QOp .real a → QOp .real b → QOp .real out
  | floorDiv  : IntegralDType dtype → Broadcast a b out →
      QOp dtype a → QOp dtype b → QOp dtype out
  | mod       : IntegralDType dtype → Broadcast a b out →
      QOp dtype a → QOp dtype b → QOp dtype out
  | bitAnd    : Broadcast a b out → QOp .nat a → QOp .nat b → QOp .nat out
  | bitOr     : Broadcast a b out → QOp .nat a → QOp .nat b → QOp .nat out
  | bitXor    : Broadcast a b out → QOp .nat a → QOp .nat b → QOp .nat out
  | shiftLeft : Broadcast a b out → QOp .nat a → QOp .nat b → QOp .nat out
  | shiftRight : Broadcast a b out → QOp .nat a → QOp .nat b → QOp .nat out
  | exp       : QOp .real shape → QOp .real shape
  | libdeviceExp : QOp .real shape → QOp .real shape
  | libdeviceLog : QOp .real shape → QOp .real shape
  | libdeviceExpm1 : QOp .real shape → QOp .real shape
  | libdeviceLog1p : QOp .real shape → QOp .real shape
  | libdeviceTanh : QOp .real shape → QOp .real shape
  | exp2      : QOp .real shape → QOp .real shape
  | log       : QOp .real shape → QOp .real shape
  | log2      : QOp .real shape → QOp .real shape
  | sigmoid   : QOp .real shape → QOp .real shape
  | sqrt      : QOp .real shape → QOp .real shape
  | rsqrt     : QOp .real shape → QOp .real shape
  | libdeviceRsqrt     : QOp .real shape → QOp .real shape
  | tanh      : QOp .real shape → QOp .real shape
  | sin       : QOp .real shape → QOp .real shape
  | cos       : QOp .real shape → QOp .real shape
  | tan       : QOp .real shape → QOp .real shape
  | atan      : QOp .real shape → QOp .real shape
  | cosh      : QOp .real shape → QOp .real shape
  | sinh      : QOp .real shape → QOp .real shape
  | erf       : QOp .real shape → QOp .real shape
  | minPosInf : QOp .real shape → QOp .real shape
  | lt        : ComparableDType dtype → Broadcast a b out →
      QOp dtype a → QOp dtype b → QOp .bool out
  | le        : ComparableDType dtype → Broadcast a b out →
      QOp dtype a → QOp dtype b → QOp .bool out
  | eq        : ComparableDType dtype → Broadcast a b out →
      QOp dtype a → QOp dtype b → QOp .bool out
  | gt        : ComparableDType dtype → Broadcast a b out →
      QOp dtype a → QOp dtype b → QOp .bool out
  | ge        : ComparableDType dtype → Broadcast a b out →
      QOp dtype a → QOp dtype b → QOp .bool out
  | ne        : ComparableDType dtype → Broadcast a b out →
      QOp dtype a → QOp dtype b → QOp .bool out
  | boolAnd   : Broadcast a b out → QOp .bool a → QOp .bool b → QOp .bool out
  | boolOr    : Broadcast a b out → QOp .bool a → QOp .bool b → QOp .bool out
  | boolNot   : QOp .bool shape → QOp .bool shape
  | max2      : Broadcast a b out → QOp .real a → QOp .real b → QOp .real out
  | maximumFloat : (dtype : FloatDType) → Broadcast a b out →
      QOp dtype.toTileDType a → QOp dtype.toTileDType b → QOp dtype.toTileDType out
  | pow       : Broadcast a b out → QOp .real a → QOp .real b → QOp .real out
  | libdevicePowi : Broadcast a b out → QOp .real a → QOp .int b → QOp .real out
  | where     : {dtype : TileDType} → {shape : TileShape} →
                QOp .bool shape → QOp dtype shape → QOp dtype shape →
                QOp dtype shape
  | ite       : {dtype : TileDType} → {shape : TileShape} →
                QOp .bool [] → QOp dtype shape → QOp dtype shape →
                QOp dtype shape
  | reduceMin : (axis : Fin shape.length) → (keepDims : Bool) →
                QOp .real shape →
                QOp .real (TileShape.reduceShape shape axis keepDims)
  | reduceMax : (axis : Fin shape.length) → (keepDims : Bool) →
                QOp .real shape →
                QOp .real (TileShape.reduceShape shape axis keepDims)
  | reduceMaxNat : (axis : Fin shape.length) → (keepDims : Bool) →
                QOp .nat shape →
                QOp .nat (TileShape.reduceShape shape axis keepDims)
  | reduceMaxInt : (axis : Fin shape.length) → (keepDims : Bool) →
                QOp .int shape →
                QOp .int (TileShape.reduceShape shape axis keepDims)
  | reduceSum : (axis : Fin shape.length) → (keepDims : Bool) →
                QOp .real shape →
                QOp .real (TileShape.reduceShape shape axis keepDims)
  | reduceSumBool : (axis : Fin shape.length) → (keepDims : Bool) →
                QOp .bool shape →
                QOp .nat (TileShape.reduceShape shape axis keepDims)
  | reduceSumTyped : (dtype : FloatDType) → (axis : Fin shape.length) → (keepDims : Bool) →
                QOp dtype.toTileDType shape →
                QOp dtype.toTileDType (TileShape.reduceShape shape axis keepDims)
  | reduceMaxTyped : (dtype : FloatDType) → (axis : Fin shape.length) → (keepDims : Bool) →
                QOp dtype.toTileDType shape →
                QOp dtype.toTileDType (TileShape.reduceShape shape axis keepDims)
  | scan      : (op : ScanOp) → (axis : Fin shape.length) → (dir : ScanDirection) →
                QOp .real shape → QOp .real shape
  | argMax    : (axis : Fin shape.length) → QOp .real shape →
                QOp .nat (TileShape.eraseAxis shape axis)
  | argMaxInt : (axis : Fin shape.length) → QOp .int shape →
                QOp .nat (TileShape.eraseAxis shape axis)
  | indexedMaxValue : (dtype : FloatDType) → (axis : Fin shape.length) →
                QOp dtype.toTileDType shape →
                QOp dtype.toTileDType (TileShape.eraseAxis shape axis)
  | argMaxTyped : (dtype : FloatDType) → (axis : Fin shape.length) →
                QOp dtype.toTileDType shape → (precision : Option IndexComputePrecision := none) →
                QOp .nat (TileShape.eraseAxis shape axis)
  | argMin    : (axis : Fin shape.length) → QOp .real shape →
                QOp .nat (TileShape.eraseAxis shape axis)
  | sort      : (axis : Fin shape.length) → QOp .real shape → QOp .real shape
  | dot       : {batch : TileShape} → {M K N : Nat} →
                QOp .real (batch ++ [M, K]) → QOp .real (batch ++ [K, N]) →
                (options : optParam DotOptions ({} : DotOptions)) → QOp .real (batch ++ [M, N])
  | dotTyped  : (left right out : FloatDType) →
                {batch : TileShape} → {M K N : Nat} →
                QOp left.toTileDType (batch ++ [M, K]) →
                QOp right.toTileDType (batch ++ [K, N]) →
                (options : optParam DotOptions ({} : DotOptions)) →
                QOp out.toTileDType (batch ++ [M, N])
  | dotAcc : (left right out : FloatDType) →
                {batch : TileShape} → {M K N : Nat} →
                QOp left.toTileDType (batch ++ [M, K]) →
                QOp right.toTileDType (batch ++ [K, N]) →
                QOp out.toTileDType (batch ++ [M, N]) →
                (options : optParam DotOptions ({} : DotOptions)) →
                QOp out.toTileDType (batch ++ [M, N])
  | dotInt    : {batch : TileShape} → {M K N : Nat} →
                QOp .int (batch ++ [M, K]) → QOp .int (batch ++ [K, N]) →
                (options : optParam DotOptions ({} : DotOptions)) → QOp .int (batch ++ [M, N])
  | transpose : {dtype : TileDType} → {batch : TileShape} → {M N : Nat} →
                QOp dtype (batch ++ [M, N]) →
                QOp dtype (batch ++ [N, M])
  | reshape  : {dtype : TileDType} → (outShape : TileShape) →
                QOp dtype shape → QOp dtype outShape
  | remap    : {dtype : TileDType} → (outShape : TileShape) →
                (TileIndex outShape → TileIndex shape) →
                QOp dtype shape → QOp dtype outShape
  | join     : {dtype : TileDType} → {shape : TileShape} →
                QOp dtype shape → QOp dtype shape → QOp dtype (shape ++ [2])
  | split    : {dtype : TileDType} → {shape : TileShape} →
                (side : Fin 2) → QOp dtype (shape ++ [2]) → QOp dtype shape
  | expandDim : {dtype : TileDType} → {shape : TileShape} →
                (axis : Fin (shape.length + 1)) →
                QOp dtype shape →
                QOp dtype (TileShape.insertAxis shape axis 1)
  | ptrBase   : {d : TileDType} → (region : Region d) → QOp .ptr []
  | ptrAdd    : Broadcast a b out → QOp .ptr a → QOp .nat b → QOp .ptr out
  | ptrSub    : Broadcast a b out → QOp .ptr a → QOp .nat b → QOp .ptr out
  | makeBlockPtr : {d : TileDType} → (region : Region d) → (baseOffset : Nat) →
                (parentShape : List Nat) → (blockShape : TileShape) →
                (strides offsets : List Nat) →
                QOp .blockPtr blockShape
  | makeBlockPtrDyn : (region : RegionName) → (baseOffset : QOp .nat []) →
                (parentShape : List Nat) → (blockShape : TileShape) →
                (strides offsets : List Nat) →
                QOp .blockPtr blockShape
  | makeBlockPtrDynOffsets : (region : RegionName) → (baseOffset : QOp .nat []) →
                (parentShape : List Nat) → (blockShape : TileShape) →
                (strides : List Nat) → (offsets : List (QOp .nat [])) →
                QOp .blockPtr blockShape
  | makeBlockPtrDynMetadata : (region : RegionName) → (baseOffset : QOp .nat []) →
                (parentShape : List Nat) → (blockShape : TileShape) →
                (strides offsets : List (QOp .nat [])) → QOp .blockPtr blockShape
  | advanceBlockPtr : QOp .blockPtr shape → (offsetDeltas : List Int) → QOp .blockPtr shape
  | advanceBlockPtrDyn : QOp .blockPtr shape → List (QOp .int []) → QOp .blockPtr shape
  | load      : (dtype : TileDType) → QMemAccess dtype shape →
                QMaskOpt dtype shape → (padding : PaddingOption := .zero) → QOp dtype shape
  | natToReal : QOp .nat shape → QOp .real shape
  | intToReal : QOp .int shape → QOp .real shape
  | minimumFloat : (dtype : FloatDType) → Broadcast a b out →
      QOp dtype.toTileDType a → QOp dtype.toTileDType b → QOp dtype.toTileDType out

/-- Mirror of `MemAccess` over `QOp`. -/
inductive QMemAccess : (d : TileDType) → TileShape → Type where
  | region : (region : Region d) → (offset : QOp .nat shape) → QMemAccess d shape
  | ptr : {d : TileDType} → QOp .ptr shape → QMemAccess d shape
  | blockPtr : {d : TileDType} → QOp .blockPtr shape → (boundaryCheck : List Nat) →
      QMemAccess d shape

/-- Mirror of `MaskOpt` over `QOp`. -/
inductive QMaskOpt : TileDType → TileShape → Type where
  | none : QMaskOpt dtype shape
  | mask : QOp .bool shape → QMaskOpt dtype shape
  | maskOther : QOp .bool shape → QOp dtype shape → QMaskOpt dtype shape

end

/-- Mirror of `Stmt` over `QOp`; `ComputeStmt.alg` embeds `Stmt`, so `QStmt.alg` needs it. -/
inductive QAlgStmt : Type where
  | assign  : (dtype : TileDType) → (shape : TileShape) → RegName → QOp dtype shape → QAlgStmt
  | store   : (dtype : TileDType) → (shape : TileShape) →
              QMemAccess dtype shape → (value : QOp dtype shape) →
              (mask : QMaskOpt dtype shape) → QAlgStmt
  | atomicAdd : NumericDType dtype → (shape : TileShape) →
              QMemAccess dtype shape → (value : QOp dtype shape) →
              (mask : QMaskOpt dtype shape) → QAlgStmt
  | atomicRMW : RMWOp → (dtype : TileDType) → (shape : TileShape) →
              QMemAccess dtype shape → (input : QOp dtype shape) →
              (extraInput : Option (QOp dtype shape)) →
              (mask : QMaskOpt dtype shape) → (dest : Option RegName) → QAlgStmt
  | forLoop : (idx : RegName) → (n : Nat) → (body : List QAlgStmt) → QAlgStmt
  | forRange : (idx : RegName) → (start stop step : Nat) → (body : List QAlgStmt) → QAlgStmt
  | forRangeDyn : (idx : RegName) → (start stop step : QOp .nat []) →
      (body : List QAlgStmt) → QAlgStmt
  | forRangeInt : (idx : RegName) → (start stop : QOp .int []) →
      (step : QOp .nat []) → (body : List QAlgStmt) → QAlgStmt
  | ifThen  : (cond : QOp .bool []) → (body : List QAlgStmt) → QAlgStmt
  | ifThenElse : (cond : QOp .bool []) → (thenBody elseBody : List QAlgStmt) → QAlgStmt

/-- Mirror of `ComputeOp` over `QOp`; its own literals are bit patterns and stay as they are. -/
inductive QComputeOp : ComputeDType → TileShape → Type where
  | alg : (dtype : ComputeDType) →
      QOp dtype.eraseDType shape →
      QComputeOp dtype shape
  | const : ComputeCarrier dtype → QComputeOp dtype []
  | full : (shape : TileShape) → QComputeOp dtype [] → QComputeOp dtype shape
  | load :
      (dtype : ComputeDType) →
      QMemAccess dtype.eraseDType shape →
      QMaskOpt dtype.eraseDType shape →
      (padding : PaddingOption := .zero) →
      QComputeOp dtype shape
  | bitcast :
      (src dst : ComputeDType) →
      src.width = dst.width →
      QComputeOp src shape →
      QComputeOp dst shape

/-- Mirror of `ComputeExpr` over `QOp` and `QComputeOp`. -/
inductive QExpr : TileDType → TileShape → Type where
  | alg : QOp dtype shape → QExpr dtype shape
  | compute : QComputeOp cdtype shape → QExpr cdtype.eraseDType shape
  | opaque : (op : String) → QExpr dtype shape
  | llrint : QExpr .real shape → QExpr .int shape
  | isFinite : FinitePrecision → QExpr .real shape → QExpr .bool shape
  | fma : IndexComputePrecision → Broadcast a b ab → Broadcast ab c out →
      QExpr .real a → QExpr .real b → QExpr .real c →
      QExpr .real out
  | fmaTyped : (dtype : FloatDType) → Broadcast a b ab → Broadcast ab c out →
      QExpr dtype.toTileDType a → QExpr dtype.toTileDType b →
      QExpr dtype.toTileDType c → QExpr dtype.toTileDType out

/-- Mirror of `ComputeStmt` over `QOp`, `QExpr` and `QAlgStmt`. -/
inductive QStmt : Type where
  | alg : QAlgStmt → QStmt
  | assign : (dtype : TileDType) → (shape : TileShape) → RegName →
      QExpr dtype shape → QStmt
  | store : (dtype : TileDType) → (shape : TileShape) →
      QMemAccess dtype shape → QExpr dtype shape → QMaskOpt dtype shape → QStmt
  | atomicAdd : NumericDType dtype → (shape : TileShape) →
      QMemAccess dtype shape → QExpr dtype shape → QMaskOpt dtype shape → QStmt
  | atomicRMW : RMWOp → (dtype : TileDType) → (shape : TileShape) →
      QMemAccess dtype shape → QExpr dtype shape →
      Option (QExpr dtype shape) → QMaskOpt dtype shape →
      Option RegName → QStmt
  | effectMarker : (op : String) → QStmt
  | forLoop : (idx : RegName) → (n : Nat) → (body : List QStmt) → QStmt
  | forRange : (idx : RegName) → (start stop step : Nat) → (body : List QStmt) → QStmt
  | forRangeDyn : (idx : RegName) → (start stop step : QOp .nat []) →
      (body : List QStmt) → QStmt
  | forRangeInt : (idx : RegName) → (start stop : QOp .int []) →
      (step : QOp .nat []) → (body : List QStmt) → QStmt
  | ifThen : (cond : QOp .bool []) → (body : List QStmt) → QStmt
  | ifThenElse : (cond : QOp .bool []) →
      (thenBody elseBody : List QStmt) → QStmt

/-- Mirror of `ComputeKernel`: inputs, outputs and a body of `QStmt`. -/
inductive QKernel where
  | mk : (inputs outputs : List RegionName) → (body : List QStmt) → QKernel

/-! ## `toReal`

Every case rebuilds the constructor it matched from the same fields, mapping
the sub-terms. The only fields that change are the two rational literals. -/

mutual

/-- The upstream expression with every rational literal `q` replaced by `(q : ℝ)`. -/
def QOp.toReal : QOp dtype shape → Op dtype shape
  | .const q => .const (q : ℝ)
  | .constFloat dtype q => .constFloat dtype (q : ℝ)
  | .constNat n => .constNat n
  | .constInt i => .constInt i
  | .constBool v => .constBool v
  | .negInf => .negInf
  | .programId axis => .programId axis
  | .numPrograms axis => .numPrograms axis
  | .ref dtype shape name => .ref dtype shape name
  | .arange n => .arange n
  | .rand cfg e₁ e₂ => .rand cfg e₁.toReal e₂.toReal
  | .randTensor cfg e₁ e₂ => .randTensor cfg e₁.toReal e₂.toReal
  | .philox4x32 cfg plan field e₁ e₂ e₃ e₄ e₅ =>
      .philox4x32 cfg plan field e₁.toReal e₂.toReal e₃.toReal e₄.toReal e₅.toReal
  | .uniform32 e => .uniform32 e.toReal
  | .broadcast e shape => .broadcast e.toReal shape
  | .full shape e => .full shape e.toReal
  | .withPrecision dtype prec e => .withPrecision dtype prec e.toReal
  | .castFloat src dst e => .castFloat src dst e.toReal
  | .negFloat dtype e => .negFloat dtype e.toReal
  | .absFloat dtype e => .absFloat dtype e.toReal
  | .floatToInt prec src n e => .floatToInt prec src n e.toReal
  | .floatToNat prec src n e => .floatToNat prec src n e.toReal
  | .castNatToInt e => .castNatToInt e.toReal
  | .castIntToNat e => .castIntToNat e.toReal
  | .castRealToInt8 e => .castRealToInt8 e.toReal
  | .decodeFloat8E5 dst e => .decodeFloat8E5 dst e.toReal
  | .add h bc e₁ e₂ => .add h bc e₁.toReal e₂.toReal
  | .sub h bc e₁ e₂ => .sub h bc e₁.toReal e₂.toReal
  | .mul h bc e₁ e₂ => .mul h bc e₁.toReal e₂.toReal
  | .div h bc e₁ e₂ => .div h bc e₁.toReal e₂.toReal
  | .divRN bc e₁ e₂ => .divRN bc e₁.toReal e₂.toReal
  | .floorDiv h bc e₁ e₂ => .floorDiv h bc e₁.toReal e₂.toReal
  | .mod h bc e₁ e₂ => .mod h bc e₁.toReal e₂.toReal
  | .bitAnd bc e₁ e₂ => .bitAnd bc e₁.toReal e₂.toReal
  | .bitOr bc e₁ e₂ => .bitOr bc e₁.toReal e₂.toReal
  | .bitXor bc e₁ e₂ => .bitXor bc e₁.toReal e₂.toReal
  | .shiftLeft bc e₁ e₂ => .shiftLeft bc e₁.toReal e₂.toReal
  | .shiftRight bc e₁ e₂ => .shiftRight bc e₁.toReal e₂.toReal
  | .exp e => .exp e.toReal
  | .libdeviceExp e => .libdeviceExp e.toReal
  | .libdeviceLog e => .libdeviceLog e.toReal
  | .libdeviceExpm1 e => .libdeviceExpm1 e.toReal
  | .libdeviceLog1p e => .libdeviceLog1p e.toReal
  | .libdeviceTanh e => .libdeviceTanh e.toReal
  | .exp2 e => .exp2 e.toReal
  | .log e => .log e.toReal
  | .log2 e => .log2 e.toReal
  | .sigmoid e => .sigmoid e.toReal
  | .sqrt e => .sqrt e.toReal
  | .rsqrt e => .rsqrt e.toReal
  | .libdeviceRsqrt e => .libdeviceRsqrt e.toReal
  | .tanh e => .tanh e.toReal
  | .sin e => .sin e.toReal
  | .cos e => .cos e.toReal
  | .tan e => .tan e.toReal
  | .atan e => .atan e.toReal
  | .cosh e => .cosh e.toReal
  | .sinh e => .sinh e.toReal
  | .erf e => .erf e.toReal
  | .minPosInf e => .minPosInf e.toReal
  | .lt h bc e₁ e₂ => .lt h bc e₁.toReal e₂.toReal
  | .le h bc e₁ e₂ => .le h bc e₁.toReal e₂.toReal
  | .eq h bc e₁ e₂ => .eq h bc e₁.toReal e₂.toReal
  | .gt h bc e₁ e₂ => .gt h bc e₁.toReal e₂.toReal
  | .ge h bc e₁ e₂ => .ge h bc e₁.toReal e₂.toReal
  | .ne h bc e₁ e₂ => .ne h bc e₁.toReal e₂.toReal
  | .boolAnd bc e₁ e₂ => .boolAnd bc e₁.toReal e₂.toReal
  | .boolOr bc e₁ e₂ => .boolOr bc e₁.toReal e₂.toReal
  | .boolNot e => .boolNot e.toReal
  | .max2 bc e₁ e₂ => .max2 bc e₁.toReal e₂.toReal
  | .maximumFloat dtype bc e₁ e₂ => .maximumFloat dtype bc e₁.toReal e₂.toReal
  | .pow bc e₁ e₂ => .pow bc e₁.toReal e₂.toReal
  | .libdevicePowi bc e₁ e₂ => .libdevicePowi bc e₁.toReal e₂.toReal
  | .where e₁ e₂ e₃ => .where e₁.toReal e₂.toReal e₃.toReal
  | .ite e₁ e₂ e₃ => .ite e₁.toReal e₂.toReal e₃.toReal
  | .reduceMin axis keepDims e => .reduceMin axis keepDims e.toReal
  | .reduceMax axis keepDims e => .reduceMax axis keepDims e.toReal
  | .reduceMaxNat axis keepDims e => .reduceMaxNat axis keepDims e.toReal
  | .reduceMaxInt axis keepDims e => .reduceMaxInt axis keepDims e.toReal
  | .reduceSum axis keepDims e => .reduceSum axis keepDims e.toReal
  | .reduceSumBool axis keepDims e => .reduceSumBool axis keepDims e.toReal
  | .reduceSumTyped dtype axis keepDims e => .reduceSumTyped dtype axis keepDims e.toReal
  | .reduceMaxTyped dtype axis keepDims e => .reduceMaxTyped dtype axis keepDims e.toReal
  | .scan op axis dir e => .scan op axis dir e.toReal
  | .argMax axis e => .argMax axis e.toReal
  | .argMaxInt axis e => .argMaxInt axis e.toReal
  | .indexedMaxValue dtype axis e => .indexedMaxValue dtype axis e.toReal
  | .argMaxTyped dtype axis e precision => .argMaxTyped dtype axis e.toReal precision
  | .argMin axis e => .argMin axis e.toReal
  | .sort axis e => .sort axis e.toReal
  | .dot e₁ e₂ options => .dot e₁.toReal e₂.toReal options
  | .dotTyped left right out e₁ e₂ options => .dotTyped left right out e₁.toReal e₂.toReal options
  | .dotAcc left right out e₁ e₂ e₃ options =>
      .dotAcc left right out e₁.toReal e₂.toReal e₃.toReal options
  | .dotInt e₁ e₂ options => .dotInt e₁.toReal e₂.toReal options
  | .transpose e => .transpose e.toReal
  | .reshape outShape e => .reshape outShape e.toReal
  | .remap outShape f e => .remap outShape f e.toReal
  | .join e₁ e₂ => .join e₁.toReal e₂.toReal
  | .split side e => .split side e.toReal
  | .expandDim axis e => .expandDim axis e.toReal
  | .ptrBase r => .ptrBase r
  | .ptrAdd bc e₁ e₂ => .ptrAdd bc e₁.toReal e₂.toReal
  | .ptrSub bc e₁ e₂ => .ptrSub bc e₁.toReal e₂.toReal
  | .makeBlockPtr r baseOffset parentShape blockShape strides offsets =>
      .makeBlockPtr r baseOffset parentShape blockShape strides offsets
  | .makeBlockPtrDyn r baseOffset parentShape blockShape strides offsets =>
      .makeBlockPtrDyn r baseOffset.toReal parentShape blockShape strides offsets
  | .makeBlockPtrDynOffsets r baseOffset parentShape blockShape strides offsets =>
      .makeBlockPtrDynOffsets r baseOffset.toReal parentShape blockShape strides
        (QOp.toRealNats offsets)
  | .makeBlockPtrDynMetadata r baseOffset parentShape blockShape strides offsets =>
      .makeBlockPtrDynMetadata r baseOffset.toReal parentShape blockShape (QOp.toRealNats strides)
        (QOp.toRealNats offsets)
  | .advanceBlockPtr e offsetDeltas => .advanceBlockPtr e.toReal offsetDeltas
  | .advanceBlockPtrDyn e deltas => .advanceBlockPtrDyn e.toReal (QOp.toRealInts deltas)
  | .load dtype mem msk padding => .load dtype mem.toReal msk.toReal padding
  | .natToReal e => .natToReal e.toReal
  | .intToReal e => .intToReal e.toReal
  | .minimumFloat dtype bc e₁ e₂ => .minimumFloat dtype bc e₁.toReal e₂.toReal
termination_by structural x => x

/-- `QOp.toReal` on each element; the structural form of `List.map`. -/
def QOp.toRealNats : List (QOp .nat []) → List (Op .nat [])
  | [] => []
  | e :: es => e.toReal :: QOp.toRealNats es
termination_by structural x => x

/-- `QOp.toReal` on each element; the structural form of `List.map`. -/
def QOp.toRealInts : List (QOp .int []) → List (Op .int [])
  | [] => []
  | e :: es => e.toReal :: QOp.toRealInts es
termination_by structural x => x

def QMemAccess.toReal : QMemAccess dtype shape → MemAccess dtype shape
  | .region r offset => .region r offset.toReal
  | .ptr e => .ptr e.toReal
  | .blockPtr e boundaryCheck => .blockPtr e.toReal boundaryCheck
termination_by structural x => x

def QMaskOpt.toReal : QMaskOpt dtype shape → MaskOpt dtype shape
  | .none => .none
  | .mask e => .mask e.toReal
  | .maskOther e₁ e₂ => .maskOther e₁.toReal e₂.toReal
termination_by structural x => x

end

mutual

def QAlgStmt.toReal : QAlgStmt → Stmt
  | .assign dtype shape name e => .assign dtype shape name e.toReal
  | .store dtype shape mem value msk => .store dtype shape mem.toReal value.toReal msk.toReal
  | .atomicAdd h shape mem value msk => .atomicAdd h shape mem.toReal value.toReal msk.toReal
  | .atomicRMW rmw dtype shape mem input extraInput msk dest =>
      .atomicRMW rmw dtype shape mem.toReal input.toReal (extraInput.map (·.toReal))
        msk.toReal dest
  | .forLoop idx n body => .forLoop idx n (QAlgStmt.toRealList body)
  | .forRange idx start stop step body => .forRange idx start stop step (QAlgStmt.toRealList body)
  | .forRangeDyn idx start stop step body =>
      .forRangeDyn idx start.toReal stop.toReal step.toReal (QAlgStmt.toRealList body)
  | .forRangeInt idx start stop step body =>
      .forRangeInt idx start.toReal stop.toReal step.toReal (QAlgStmt.toRealList body)
  | .ifThen cond body => .ifThen cond.toReal (QAlgStmt.toRealList body)
  | .ifThenElse cond thenBody elseBody =>
      .ifThenElse cond.toReal (QAlgStmt.toRealList thenBody) (QAlgStmt.toRealList elseBody)
termination_by structural x => x

/-- `QAlgStmt.toReal` on each element; the structural form of `List.map`. -/
def QAlgStmt.toRealList : List QAlgStmt → List Stmt
  | [] => []
  | st :: rest => st.toReal :: QAlgStmt.toRealList rest
termination_by structural x => x

end

def QComputeOp.toReal : QComputeOp dtype shape → ComputeOp dtype shape
  | .alg dtype e => .alg dtype e.toReal
  | .const bits => .const bits
  | .full shape e => .full shape e.toReal
  | .load dtype mem msk padding => .load dtype mem.toReal msk.toReal padding
  | .bitcast src dst h e => .bitcast src dst h e.toReal
termination_by structural x => x

def QExpr.toReal : QExpr dtype shape → ComputeExpr dtype shape
  | .alg e => .alg e.toReal
  | .compute e => .compute e.toReal
  | .opaque op => .opaque op
  | .llrint e => .llrint e.toReal
  | .isFinite prec e => .isFinite prec e.toReal
  | .fma prec bc₁ bc₂ e₁ e₂ e₃ => .fma prec bc₁ bc₂ e₁.toReal e₂.toReal e₃.toReal
  | .fmaTyped dtype bc₁ bc₂ e₁ e₂ e₃ => .fmaTyped dtype bc₁ bc₂ e₁.toReal e₂.toReal e₃.toReal
termination_by structural x => x

mutual

def QStmt.toReal : QStmt → ComputeStmt
  | .alg st => .alg st.toReal
  | .assign dtype shape name e => .assign dtype shape name e.toReal
  | .store dtype shape mem e msk => .store dtype shape mem.toReal e.toReal msk.toReal
  | .atomicAdd h shape mem e msk => .atomicAdd h shape mem.toReal e.toReal msk.toReal
  | .atomicRMW rmw dtype shape mem e extraInput msk dest =>
      .atomicRMW rmw dtype shape mem.toReal e.toReal (extraInput.map (·.toReal)) msk.toReal dest
  | .effectMarker op => .effectMarker op
  | .forLoop idx n body => .forLoop idx n (QStmt.toRealList body)
  | .forRange idx start stop step body => .forRange idx start stop step (QStmt.toRealList body)
  | .forRangeDyn idx start stop step body =>
      .forRangeDyn idx start.toReal stop.toReal step.toReal (QStmt.toRealList body)
  | .forRangeInt idx start stop step body =>
      .forRangeInt idx start.toReal stop.toReal step.toReal (QStmt.toRealList body)
  | .ifThen cond body => .ifThen cond.toReal (QStmt.toRealList body)
  | .ifThenElse cond thenBody elseBody =>
      .ifThenElse cond.toReal (QStmt.toRealList thenBody) (QStmt.toRealList elseBody)
termination_by structural x => x

/-- `QStmt.toReal` on each element; the structural form of `List.map`. -/
def QStmt.toRealList : List QStmt → List ComputeStmt
  | [] => []
  | st :: rest => st.toReal :: QStmt.toRealList rest
termination_by structural x => x

end

/-- Replaces every rational literal `q` by the real `(q : ℝ)` and changes nothing else. -/
def QKernel.toReal : QKernel → ComputeKernel
  | .mk inputs outputs body => .mk inputs outputs (QStmt.toRealList body)

/-! ## Exhaustiveness check -/

open Lean Elab Command in
/-- `#assert_mirror U ↦ Q, …` fails unless every mirror `Q` has the constructors of its
upstream `U`, in the same order, and the type of each (and of the family) is the upstream
type with `ℝ` replaced by `ℚ` and every listed `U` by its `Q`. -/
elab "#assert_mirror " pairs:(ident " ↦ " ident),+ : command => do
  let pairs ← pairs.getElems.mapM fun p => liftCoreM do
    return (← realizeGlobalConstNoOverloadWithInfo p.raw[0],
      ← realizeGlobalConstNoOverloadWithInfo p.raw[2])
  let env ← getEnv
  let names := #[(``Real, ``Rat), (``AlgDType, ``TileDType)] ++ pairs
  let mirror (e : Expr) : Expr := e.replace fun
    | .const n us => (names.find? (·.1 == n)).map (.const ·.2 us)
    | _ => none
  for (u, q) in pairs do
    let some (.inductInfo ui) := env.find? u | throwError "{u} is not an inductive type"
    let some (.inductInfo qi) := env.find? q | throwError "{q} is not an inductive type"
    let expected := ui.ctors.map (·.replacePrefix u q)
    unless expected == qi.ctors do
      throwError "{q} does not have the constructors of {u} in the same order:\n  \
        missing {expected.filter (!qi.ctors.contains ·)}\n  \
        extra {qi.ctors.filter (!expected.contains ·)}"
    for (uc, qc) in (u :: ui.ctors).zip (q :: qi.ctors) do
      let (some ud, some qd) := (env.find? uc, env.find? qc) | unreachable!
      unless mirror ud.type == qd.type do
        throwError "{qc} : {qd.type}\nis not the mirror of\n{uc} : {ud.type}"

#assert_mirror Op ↦ QOp, MemAccess ↦ QMemAccess, MaskOpt ↦ QMaskOpt, Stmt ↦ QAlgStmt,
  ComputeOp ↦ QComputeOp, ComputeExpr ↦ QExpr, ComputeStmt ↦ QStmt, ComputeKernel ↦ QKernel

end VeriTile.Interp
