/-
Sequential tensor descriptors, lowered to the existing block-pointer syntax.

Triton 3.5.1 documents a descriptor's shape, strides and block shape;
loads zero-fill outside the tensor and stores ignore those lanes:
https://github.com/triton-lang/triton/blob/v3.5.1/python/triton/language/core.py#L1293-L1309
https://github.com/triton-lang/triton/blob/v3.5.1/python/triton/language/core.py#L2053-L2103
Accessed 2026-11-13. Device lowering is compiler trust base; documented alignment and stride
requirements remain source input preconditions.
-/
import VeriTile.Triton.DSL.PythonFullConstructors

namespace VeriTile.Dist.Frontend.TritonExt.TensorDescriptor
open Lean Elab Term
open VeriTile.Triton.DSL

declare_syntax_cat descriptorBlockDim
syntax tritonExpr : descriptorBlockDim
syntax tritonExpr " if " "not " ident " else " tritonExpr : descriptorBlockDim

syntax (name := makeDescriptor) "tl.make_tensor_descriptor(" tritonExpr ","
  ident "=" "[" tritonExpr,* "]" "," ident "=" "[" tritonExpr,* "]" ","
  ident "=" "[" descriptorBlockDim,* "]" ")" : tritonExpr
syntax (name := loadDescriptor) "tl.load_tensor_descriptor(" ident "," "[" tritonExpr,* "]" ")" : tritonExpr
syntax (name := storeDescriptor) "tl.store_tensor_descriptor(" ident "," "[" tritonExpr,* "]" "," tritonExpr ")" : tritonStmt
syntax (name := methodLoad) ident noWs "(" "[" tritonExpr,* "]" ")" : tritonExpr
syntax (name := methodStore) ident noWs "(" "[" tritonExpr,* "]" "," tritonExpr ")" : tritonStmt

private partial def ranks (raw : Syntax) : MacroM (List (Name × Nat)) := do
  let mut found := []
  match (⟨raw⟩ : TSyntax `tritonStmt) with
  | `(tritonStmt| $name:ident = tl.make_tensor_descriptor($_base:tritonExpr,
      $_shapeKw:ident=[$shape:tritonExpr,*], $_stridesKw:ident=[$_strides:tritonExpr,*], $_blockKw:ident=[$_block:descriptorBlockDim,*])) =>
    let rank := shape.getElems.size
    found := [(name.getId.eraseMacroScopes, rank)]
  | _ => pure ()
  for child in raw.getArgs do found := found ++ (← ranks child)
  return found

private partial def checkBindings (known : List (Name × Nat)) (raw : Syntax) : MacroM Unit := do
  let isDescriptor (name : Ident) := known.any (·.1 == name.getId.eraseMacroScopes)
  match (⟨raw⟩ : TSyntax `tritonStmt) with
  | `(tritonStmt| $name:ident = $value:tritonExpr)
  | `(tritonStmt| $name:ident := $value:tritonExpr) =>
    if isDescriptor name && !value.raw.isOfKind ``makeDescriptor then
      Macro.throwErrorAt name "tensor descriptor: descriptor names cannot be rebound to other values"
  | _ => pure ()
  let args := raw.getArgs
  if args.size > 1 && args[0]!.isAtom &&
      ["for", "tl.for"].contains args[0]!.getAtomVal.trimAscii.toString && args[1]!.isIdent then
    if isDescriptor ⟨args[1]!⟩ then
      Macro.throwErrorAt raw "tensor descriptor: a loop counter cannot shadow a descriptor"
  for child in args do checkBindings known child

private def axes (known : List (Name × Nat)) (name : Ident) (offsets : TSyntaxArray `tritonExpr) :
    MacroM (TSyntax `term) := do
  let some (_, rank) := known.find? (·.1 == name.getId.eraseMacroScopes)
    | Macro.throwErrorAt name "tensor descriptor: receiver must have a make_tensor_descriptor binding"
  unless offsets.size == rank do
    Macro.throwErrorAt name "tensor descriptor: one offset is required per dimension"
  let values := (List.range rank).toArray.map fun n => (⟨Syntax.mkNumLit (toString n)⟩ : TSyntax `term)
  `([$values,*])

private def blockDim (raw : TSyntax `descriptorBlockDim) : MacroM (TSyntax `tritonExpr) := do
  match raw with
  | `(descriptorBlockDim| $a:tritonExpr if not $flag:ident else $b:tritonExpr) =>
    let a ← natDimTerm "tensor descriptor block_shape" a
    let b ← natDimTerm "tensor descriptor block_shape" b
    let dim ← `(if $flag:ident then $b else $a)
    `(tritonExpr| $($dim))
  | `(descriptorBlockDim| $d:tritonExpr) => pure d
  | _ => Macro.throwUnsupported

private partial def rewrite (known : List (Name × Nat)) (raw : Syntax) : MacroM Syntax := do
  let raw := raw.setArgs (← raw.getArgs.mapM (rewrite known))
  match (⟨raw⟩ : TSyntax `tritonExpr) with
  | `(tritonExpr| tl.make_tensor_descriptor($base:tritonExpr,
      $shapeKw:ident=[$shape:tritonExpr,*], $stridesKw:ident=[$strides:tritonExpr,*], $blockKw:ident=[$block:descriptorBlockDim,*])) =>
    unless shapeKw.getId.toString == "shape" && stridesKw.getId.toString == "strides" &&
        blockKw.getId.toString == "block_shape" do
      Macro.throwErrorAt raw "tensor descriptor: expected shape, strides, block_shape in that order"
    let rank := shape.getElems.size
    unless 2 ≤ rank && rank ≤ 5 && strides.getElems.size == rank && block.getElems.size == rank do
      Macro.throwErrorAt raw "tensor descriptor: shape, strides and block_shape must have the same rank (2–5)"
    let block ← block.getElems.mapM blockDim
    let zeros ← shape.getElems.mapM fun _ => `(tritonExpr| 0)
    let baseKw := mkIdent `base
    let offsetsKw := mkIdent `offsets
    return (← `(tritonExpr| tl.make_block_ptr($baseKw:ident=$base, $shapeKw:ident=($shape,*),
      $stridesKw:ident=($strides,*), $offsetsKw:ident=($zeros,*), $blockKw:ident=($block,*)))).raw
  | `(tritonExpr| $method:ident([$offsets:tritonExpr,*])) =>
    let .str receiver "load" := method.getId.eraseMacroScopes
      | Macro.throwErrorAt method "tensor descriptor: expected a .load method"
    let name := mkIdentFrom method receiver
    let checks ← axes known name offsets.getElems
    return (← `(tritonExpr| tl.load(tl.advance($name:ident, ($offsets,*)),
      boundary_check=$checks:term, padding_option="zero"))).raw
  | `(tritonExpr| tl.load_tensor_descriptor($name:ident, [$offsets:tritonExpr,*])) =>
    let checks ← axes known name offsets.getElems
    return (← `(tritonExpr| tl.load(tl.advance($name:ident, ($offsets,*)),
      boundary_check=$checks:term, padding_option="zero"))).raw
  | _ => pure ()
  match (⟨raw⟩ : TSyntax `tritonStmt) with
  | `(tritonStmt| $method:ident([$offsets:tritonExpr,*], $value:tritonExpr)) =>
    let .str receiver "store" := method.getId.eraseMacroScopes
      | Macro.throwErrorAt method "tensor descriptor: expected a .store method"
    let name := mkIdentFrom method receiver
    let checks ← axes known name offsets.getElems
    return (← `(tritonStmt| tl.store(tl.advance($name:ident, ($offsets,*)), $value,
      boundary_check=$checks:term))).raw
  | `(tritonStmt| tl.store_tensor_descriptor($name:ident, [$offsets:tritonExpr,*], $value:tritonExpr)) =>
    let checks ← axes known name offsets.getElems
    return (← `(tritonStmt| tl.store(tl.advance($name:ident, ($offsets,*)), $value,
      boundary_check=$checks:term))).raw
  | _ => return raw

private def lower (raw : Syntax) : MacroM Syntax := do
  let known ← ranks raw
  -- shortcut: direct descriptor bindings retain one rank; add lexical alias/rank tracking when a target requires it.
  for (name, rank) in known do
    unless known.all (fun entry => entry.1 != name || entry.2 == rank) do
      Macro.throwErrorAt raw "tensor descriptor: a name cannot change rank"
  checkBindings known raw
  rewrite known raw

@[term_elab VeriTile.Triton.DSL.tritonBlock]
def elaborate : TermElab := fun stx expected => do
  let rewritten ← liftMacroM (lower stx)
  if rewritten == stx then throwUnsupportedSyntax
  elabTerm rewritten expected

-- Signature checks must see descriptor reads/writes before validating pointer
-- ports, inferring storage types and inserting the source's store conversion.
@[term_elab VeriTile.Triton.DSL.CallSignature.withSignature,
  term_elab VeriTile.Triton.DSL.CallSignature.withCall]
def elaborateSource : TermElab := fun stx expected => do
  let raw := stx.getArgs.back!
  let .ok source := RawSourceText.normalize raw.getAtomVal | throwUnsupportedSyntax
  let source := PythonCallSyntax.normalize source
  let .ok parsed := Parser.runParserCategory (← getEnv) `term ("triton {\n" ++ source ++ "\n}")
    | throwUnsupportedSyntax
  let rewritten ← liftMacroM (lower parsed)
  if rewritten == parsed then throwUnsupportedSyntax
  let `(triton { $body:tritonStmt* }) := rewritten | throwUnsupportedSyntax
  let source := String.intercalate "\n" (body.toList.map (PythonFullConstructors.render ·.raw))
  let replacement := Syntax.atom raw.getHeadInfo ("{\n" ++ source ++ "\n}")
  elabTerm (stx.setArgs (stx.getArgs.set! (stx.getArgs.size - 1) replacement)) expected

end VeriTile.Dist.Frontend.TritonExt.TensorDescriptor
