/-
Sequential spelling extensions for the target census (#77).

`tl.range` uses Python range bounds; `loop_unroll_factor` is only a compiler
hint (the documentation includes values below two as no unrolling). The target
census uses that hint with a literal value, and uses `step`.
Other range keywords remain parse failures. `pass` emits no statement and
`tensor.sum(axis=...)` is the existing free-function sum.

Sources (accessed 2026-11-13):
- Triton v3.5.1 `range` documentation:
  https://github.com/triton-lang/triton/blob/v3.5.1/python/triton/language/core.py#L3212-L3253
- Triton sum/member-function documentation:
  https://triton-lang.org/main/python-api/generated/triton.language.sum.html
- Python pass statement:
  https://docs.python.org/3/reference/simple_stmts.html#the-pass-statement
- Target uses at Triton-distributed 7908e4ea9010bb2238f9f05b28c904fd4b71c60a:
  `python/triton_dist/kernels/nvidia/moe_reduce_rs.py:582,663`,
  `python/triton_dist/kernels/nvidia/moe_reduce_ar.py:589` (range keywords);
  `python/triton_dist/kernels/nvidia/allgather_group_gemm.py:119` (named step);
  `python/triton_dist/kernels/nvidia/low_latency_all_to_all_v2.py:486` (sum).

The upstream expander is unchanged. These term elaborators rewrite the new
spellings before delegating to it; call-source rewriting precedes its checked
constexpr specialization. This module does not implement the other semantic
and constexpr gaps assigned to #77 by the census. Its source hooks cover
`with_triton_signature` and `with_triton_call`; helper-specific source entrypoints
need the separately tracked source-binding work.
-/
import VeriTile.Triton.DSL.PythonFullConstructors

namespace VeriTile.Dist.Frontend.TritonExt
open Lean Elab Term
open VeriTile.Triton.DSL

declare_syntax_cat tritonRangeArg
syntax tritonExpr : tritonRangeArg
syntax "step=" tritonExpr : tritonRangeArg
syntax "loop_unroll_factor=" num : tritonRangeArg
syntax (name := tritonRange) "for " ident " in " "tl.range(" tritonRangeArg,* ")"
  " { " tritonStmt* " }" : tritonStmt
syntax (name := pythonRangeStep) "for " ident " in " "range(" tritonExpr ", " tritonExpr
  ", " "step=" tritonExpr ")" " { " tritonStmt* " }" : tritonStmt
-- The existing grammar tokenizes this literal prefix as one token.
syntax (name := pythonRangeZeroStep) "for " ident " in " "range(0, " tritonExpr
  ", " "step=" tritonExpr ")" " { " tritonStmt* " }" : tritonStmt
syntax (name := tritonPass) "pass" : tritonStmt
syntax:100 (name := sumMethod) tritonExpr:100 ".sum(" "axis" "=" num ")" : tritonExpr
syntax:max (name := sumDottedMethod) ident noWs "(" "axis" "=" num ")" : tritonExpr

private partial def lowerRange (raw : Syntax) : MacroM Syntax := do
  let raw := raw.setArgs (← (raw.getArgs.filter (! ·.isOfKind ``tritonPass)).mapM lowerRange)
  if raw.isOfKind ``sumDottedMethod then
    let .str receiver "sum" := raw[0].getId.eraseMacroScopes
      | Macro.throwErrorAt raw "expected tensor.sum(axis=...)"
    let receiver := mkIdent receiver
    let dim : TSyntax `num := ⟨raw[4]⟩
    return (← `(tritonExpr| tl.sum($receiver:ident, axis=$dim:num))).raw
  if raw.isOfKind ``sumMethod then
    let receiver : TSyntax `tritonExpr := ⟨raw[0]⟩
    let dim : TSyntax `num := ⟨raw[4]⟩
    return (← `(tritonExpr| tl.sum($receiver:tritonExpr, axis=$dim:num))).raw
  match raw with
  | `(tritonStmt| for $i:ident in tl.range($args:tritonRangeArg,*) { $body:tritonStmt* }) =>
    let mut bounds : List (TSyntax `tritonExpr) := []
    let mut step : Option (TSyntax `tritonExpr) := none
    let mut unroll := Bool.false
    let mut keyword := Bool.false
    for arg in args.getElems do
      match arg with
      | `(tritonRangeArg| step=$value:tritonExpr) =>
        if step.isSome then Macro.throwErrorAt arg "duplicate tl.range step"
        step := some value
        keyword := Bool.true
      | `(tritonRangeArg| loop_unroll_factor=$_:num) =>
        if unroll then Macro.throwErrorAt arg "duplicate tl.range loop_unroll_factor"
        unroll := Bool.true
        keyword := Bool.true
      | `(tritonRangeArg| $value:tritonExpr) =>
        if keyword then Macro.throwErrorAt arg "positional tl.range bound after keyword"
        bounds := bounds ++ [value]
      | _ => Macro.throwUnsupported
    match bounds, step with
    | [stop], none => return (← `(tritonStmt| for $i:ident in range($stop) { $body* })).raw
    | [stop], some stride =>
      let start ← `(tritonExpr| 0)
      return (← `(tritonStmt| for $i:ident in range($start, $stop, $stride) { $body* })).raw
    | [start, stop], none =>
      return (← `(tritonStmt| for $i:ident in range($start, $stop) { $body* })).raw
    | [start, stop, stride], none | [start, stop], some stride =>
      return (← `(tritonStmt| for $i:ident in range($start, $stop, $stride) { $body* })).raw
    | _, _ => Macro.throwErrorAt raw "tl.range requires one to three bounds and at most one step"
  | `(tritonStmt| for $i:ident in range($start:tritonExpr, $stop:tritonExpr, step=$stride:tritonExpr) { $body:tritonStmt* }) =>
    return (← `(tritonStmt| for $i:ident in range($start, $stop, $stride) { $body* })).raw
  | `(tritonStmt| for $i:ident in range(0, $stop:tritonExpr, step=$stride:tritonExpr) { $body:tritonStmt* }) =>
    let start ← `(tritonExpr| 0)
    return (← `(tritonStmt| for $i:ident in range($start, $stop, $stride) { $body* })).raw
  | _ => return raw

@[term_elab VeriTile.Triton.DSL.tritonBlock]
def elaborateRange : TermElab := fun stx expected => do
  let lowered ← liftMacroM (lowerRange stx)
  if lowered == stx then throwUnsupportedSyntax
  elabTerm lowered expected

-- Rewrite before call-signature specialization so constexpr bounds take the
-- same path as the corresponding Python range.
@[term_elab VeriTile.Triton.DSL.CallSignature.withSignature,
  term_elab VeriTile.Triton.DSL.CallSignature.withCall]
def elaborateRangeSource : TermElab := fun stx expected => do
  let raw := stx.getArgs.back!
  let .ok source := RawSourceText.normalize raw.getAtomVal | throwUnsupportedSyntax
  let .ok parsed := Parser.runParserCategory (← getEnv) `term ("triton {\n" ++ source ++ "\n}")
    | throwUnsupportedSyntax
  let lowered ← liftMacroM (lowerRange parsed)
  if lowered == parsed then throwUnsupportedSyntax
  let `(triton { $body:tritonStmt* }) := lowered | throwUnsupportedSyntax
  let source := String.intercalate "\n" (body.toList.map (PythonFullConstructors.render ·.raw))
  let replacement := Syntax.atom raw.getHeadInfo ("{\n" ++ source ++ "\n}")
  elabTerm (stx.setArgs (stx.getArgs.set! (stx.getArgs.size - 1) replacement)) expected

end VeriTile.Dist.Frontend.TritonExt
