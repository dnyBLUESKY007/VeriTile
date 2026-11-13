import VeriTile.Triton.DSL.SourceProcedures

/-! Preserve the existing unary source-helper parser when procedure syntax is
also imported. Only choose an existing parse alternative; the upstream helper
prepass still checks bindings, scope and types and supplies all semantics. -/
namespace VeriTile.Dist.Frontend.TritonExt.SourceHelperSyntax
open Lean Elab Term
open VeriTile.Triton.DSL

private partial def helpers (raw : Syntax) : List Name :=
  let here := if raw.isOfKind ``tritonSourceUniformHelper ||
      raw.isOfKind ``tritonSourceFiniteHelper then [raw[1].getId.eraseMacroScopes] else []
  here ++ raw.getArgs.toList.flatMap helpers

private partial def select (bound : List Name) (raw : Syntax) : Syntax :=
  if raw.isOfKind choiceKind then
    match raw.getArgs.find? (fun candidate => candidate.isOfKind ``tritonSourceHelperCall &&
        bound.contains candidate[0].getId.eraseMacroScopes) with
    | some existing => select bound existing
    | none => raw.setArgs (raw.getArgs.map (select bound))
  else raw.setArgs (raw.getArgs.map (select bound))

@[term_elab VeriTile.Triton.DSL.tritonBlock]
def elaborate : TermElab := fun stx expected => do
  let rewritten := select (helpers stx) stx
  if rewritten == stx then throwUnsupportedSyntax
  elabTerm rewritten expected

end VeriTile.Dist.Frontend.TritonExt.SourceHelperSyntax
