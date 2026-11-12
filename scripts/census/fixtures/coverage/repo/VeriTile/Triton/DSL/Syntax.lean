/- Fixture for test_triton_coverage.py: declarations in the style of the macro.
   /- A nested comment. -/ The next line is still a comment.
syntax "tl.void" : tritonDType
-/
namespace Fixture

declare_syntax_cat tritonExpr
syntax "tl.float32" : tritonDType  -- syntax "tl.void" : tritonDType
syntax ident " = " tritonExpr : tritonMemKwarg
syntax "axis" "=" num : tritonReduceKwarg
syntax "tl.load(" tritonExpr ("," tritonMemKwarg)* ")" : tritonExpr
syntax:max (name := methodCast) (priority := high) tritonExpr:max noWs "." "to" "(" tritonDType ")" : tritonExpr
syntax "tl.cast(" tritonExpr ", " tritonDType ")" : tritonExpr
syntax "tl.sum(" tritonExpr ("," tritonReduceKwarg)* ")" : tritonExpr
syntax "tl.math.exp2(" tritonExpr ")" : tritonExpr
syntax "libdevice.exp(" tritonExpr ")" : tritonExpr
syntax "tl.tanh(" tritonExpr ")" : tritonExpr
syntax "tl.inline_asm_elementwise(" str ")" : tritonExpr
syntax "tl.atomic_max(" tritonExpr ", " tritonExpr ("," tritonMemKwarg)* ")" : tritonStmt
syntax "tl.store(" tritonExpr ", " tritonExpr
  ("," tritonMemKwarg)* ")" : tritonStmt
syntax ident " = " tritonExpr : tritonStmt
syntax ident " += " tritonExpr : tritonStmt
syntax ident " : " "tl.constexpr" " = " tritonExpr : tritonStmt
syntax "for " ident " in " "range(" tritonExpr ")" " { " tritonStmt* " }" : tritonStmt
scoped syntax "pass" : controlSourceStmt

end Fixture
