import VeriTile.Dist.Frontend.TritonExt.SourceHelperSyntax

namespace VeriTile.Tests.Dist.SourceHelperSyntax
open VeriTile.Triton

def uniformAlias (word : Nat) : ComputeKernel := triton {
  tl.source_helper helper = tl.uint_to_uniform_float
  word_arg = $((word : Nat))
  tl.source_integer word_arg = tl.uint32
  uniform = helper(word_arg)
}
example (word : Nat) : (uniformAlias word).surfaceBody.getLast? = some
    (.assign .real [] "uniform" (.compute (.alg .fp32
      (.uniform32 (.ref .nat [] "word_arg"))))) := rfl

-- Selecting a parse does not grant a binding outside its original scope.
/-- error: random source helper requires an explicit tl.source_helper binding -/
#guard_msgs in
example (choose : Bool) : ComputeKernel := triton {
  if $((choose : Bool)) {
    tl.source_helper helper = tl.uint_to_uniform_float
  }
  uniform = helper((0).to(tl.uint32))
}

-- A procedure call retains the other parser and the existing FP16 conversion.
def copyProcedure : SourceProcedure :=
  with_triton_params (x,y) constexpr_args [] in triton {
    v = tl.load(x + 0)
    tl.store(y + 0,v)
  }
private def signature : CallSignature where
  pointers := [("x",.fp32),("y",.fp16)]
def caller (x y : RegionName) : ComputeKernel :=
  with_triton_call signature call_args [] source_procedures [helper := copyProcedure] in triton {
    helper(x=x,y=y)
  }
def procedureInstance (x y : RegionName) : ComputeKernel :=
  instantiate_source_procedure copyProcedure signature signature procedure_args [x := x,y := y]
example (x y : RegionName) : (caller x y).surfaceBody = (procedureInstance x y).surfaceBody := rfl

end VeriTile.Tests.Dist.SourceHelperSyntax
