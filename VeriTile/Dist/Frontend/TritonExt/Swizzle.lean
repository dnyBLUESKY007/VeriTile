import VeriTile.Triton.DSL.SourceProcedures
import VeriTile.Triton.DSL.PythonFullConstructors

/-!
Source adapter for the two `tl.swizzle2d` tuple assignments in ticket #453.
All arguments are captured once before either destination is assigned. Fresh
register names avoid identifiers in the complete source and its bindings, even
after rendering and erasing Lean macro scopes.

The coordinate correspondence has domain M > 0, N > 0, G > 0,
0 <= i < M and 0 <= j < N. Machine-integer source correspondence additionally
requires representable intermediate products. Runtime dimensions are retained.
Known nonpositive literal dimensions are rejected; runtime validity remains a
kernel precondition. The final group uses min(M - firstRow, G) rows.

Sources (accessed 2026-11-16):
* Triton v3.5.1, python/triton/language/standard.py:77-114 (docstring and
  implementation correspondence):
  https://github.com/triton-lang/triton/blob/v3.5.1/python/triton/language/standard.py#L77-L114
* Front-end documentation defines row-major to column-major order per group:
  https://triton-lang.org/main/python-api/generated/triton.language.swizzle2d.html
* Triton-distributed at 7908e4ea9010bb2238f9f05b28c904fd4b71c60a,
  python/triton_dist/kernels/nvidia/ep_all2all_fused.py:651,793:
  https://github.com/ByteDance-Seed/Triton-distributed/blob/7908e4ea9010bb2238f9f05b28c904fd4b71c60a/python/triton_dist/kernels/nvidia/ep_all2all_fused.py#L651
  Direct targets start at 599 and 750. Targets at 839, 1020 and 1288 reach
  them through calls at 981; 1095/1183; 1386/1468/1509/1585, respectively.
  This adapter covers the two call patterns, not the other constructs of the
  five complete kernels.
-/
namespace VeriTile.Dist.Frontend.TritonExt.Swizzle
open Lean Elab Term
open VeriTile.Triton.DSL

syntax (name := swizzleAssign) ident ", " ident " = " "tl.swizzle2d("
  tritonExpr ", " tritonExpr ", " tritonExpr ", " tritonExpr ", " tritonExpr ")" : tritonStmt

private partial def identifiers (raw : Syntax) : List Name :=
  (if raw.isIdent then [raw.getId.eraseMacroScopes] else []) ++
    raw.getArgs.toList.flatMap identifiers

private structure Fresh where
  used : List Name
  next : Nat := 0

private def freshName : StateT Fresh MacroM (TSyntax `ident) := do
  let state : Fresh ← get
  let mut index := state.next
  -- Among used.length + 1 distinct candidates at least one is unused.
  for _ in [:state.used.length + 1] do
    let name := Name.mkSimple ("veritile_swizzle_" ++ toString index)
    index := index + 1
    unless state.used.contains name do
      set ({ used := name :: state.used, next := index } : Fresh)
      return ⟨mkIdent name⟩
  liftM (Macro.throwError "could not allocate a fresh swizzle register")

private def checkDimension (value : TSyntax `tritonExpr) : MacroM Unit := do
  match value with
  | `(tritonExpr| $n:num) =>
    if n.getNat == 0 then Macro.throwErrorAt value "swizzle2d dimensions and group size must be positive"
  | `(tritonExpr| -$_:num) =>
    Macro.throwErrorAt value "swizzle2d dimensions and group size must be positive"
  | _ => pure ()

-- Standalone unannotated DSL literals default to Real. Preserve the integer
-- interpretation of scalar coordinate literals when creating a register.
private def integerLiteral (value : TSyntax `tritonExpr) : MacroM (TSyntax `tritonExpr) := do
  match value with
  | `(tritonExpr| $n:num) => `(tritonExpr| tl.cast($n:num, tl.int64))
  | `(tritonExpr| -$n:num) => `(tritonExpr| tl.cast(-$n:num, tl.int64))
  | _ => pure value

private partial def lower (raw : Syntax) : StateT Fresh MacroM (Array Syntax) := do
  if let `(tritonStmt| $a:ident, $b:ident = tl.swizzle2d($i:tritonExpr, $j:tritonExpr,
      $m:tritonExpr, $n:tritonExpr, $g:tritonExpr)) := raw then
    for dimension in [m,n,g] do liftM (checkDimension dimension)
    let i ← liftM (integerLiteral i)
    let j ← liftM (integerLiteral j)
    let m ← liftM (integerLiteral m)
    let n ← liftM (integerLiteral n)
    let g ← liftM (integerLiteral g)
    let ci ← freshName; let cj ← freshName
    let cm ← freshName; let cn ← freshName; let cg ← freshName
    let linear ← freshName; let width ← freshName; let first ← freshName
    let height ← freshName; let offset ← freshName
    let ri ← freshName; let rj ← freshName
    let block ← liftM ((`(triton {
      $ci:ident = $i
      $cj:ident = $j
      $cm:ident = $m
      $cn:ident = $n
      $cg:ident = $g
      $linear:ident = $ci:ident * $cn:ident + $cj:ident
      $width:ident = $cg:ident * $cn:ident
      $first:ident = ($linear:ident // $width:ident) * $cg:ident
      $height:ident = tl.minimum($cm:ident - $first:ident, $cg:ident)
      $offset:ident = $linear:ident % $width:ident
      $ri:ident = $first:ident + $offset:ident % $height:ident
      $rj:ident = $offset:ident // $height:ident
      $a:ident = $ri:ident
      $b:ident = $rj:ident
    })) : MacroM (TSyntax `term))
    let `(triton { $body:tritonStmt* }) := block
      | liftM (Macro.throwError "expected a swizzle statement block")
    return body.map (·.raw)
  let args ← raw.getArgs.foldlM (fun result arg => do
    return result ++ (← lower arg)) #[]
  return #[raw.setArgs args]

private def rewrite (raw bindings : Syntax) : MacroM Syntax := do
  let (result, _) ← (lower raw).run { used := identifiers raw ++ identifiers bindings }
  return result[0]!

@[term_elab VeriTile.Triton.DSL.tritonBlock]
def elaborate : TermElab := fun stx expected => do
  let rewritten ← liftMacroM (rewrite stx stx)
  if rewritten == stx then throwUnsupportedSyntax
  elabTerm rewritten expected

@[term_elab VeriTile.Triton.DSL.CallSignature.withSignature,
  term_elab VeriTile.Triton.DSL.CallSignature.withCall,
  term_elab VeriTile.Triton.DSL.SourceProcedures.sourceParams,
  term_elab VeriTile.Triton.DSL.SourceProcedures.sourceProcedure]
def elaborateSource : TermElab := fun stx expected => do
  let raw := stx.getArgs.back!
  let .ok source := RawSourceText.normalize raw.getAtomVal | throwUnsupportedSyntax
  let .ok parsed := Parser.runParserCategory (← getEnv) `term ("triton {\n" ++ source ++ "\n}")
    | throwUnsupportedSyntax
  let rewritten ← liftMacroM (rewrite parsed stx)
  if rewritten == parsed then throwUnsupportedSyntax
  let `(triton { $body:tritonStmt* }) := rewritten | throwUnsupportedSyntax
  let source := String.intercalate "\n" (body.toList.map (PythonFullConstructors.render ·.raw))
  let replacement := Syntax.atom raw.getHeadInfo ("{\n" ++ source ++ "\n}")
  elabTerm (stx.setArgs (stx.getArgs.set! (stx.getArgs.size - 1) replacement)) expected

end VeriTile.Dist.Frontend.TritonExt.Swizzle
