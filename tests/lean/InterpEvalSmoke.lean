/-
tests.lean.InterpEvalSmoke

Run reflected bench kernels on a rational arithmetic test carrier. Expected
outputs are checked by #guard as well as displayed by #eval. This carrier is
not an IEEE model; the unrelated mandatory operations below are test stubs.
-/
import VeriTile.Interp.Exec
import VeriTile.Interp.Reflect
import bench.examples.VectorAdd.Kernels
import bench.examples.RowWiseSum.Kernels

namespace VeriTile.Tests.InterpEvalSmoke

open VeriTile Triton VeriTile.Interp
open VeriTile.Triton.FP
open VeriTile.Bench.Examples

private def sumAxis {shape : TileShape} (ax : Fin shape.length) (keepDims : Bool)
    (v : TileIndex shape → ℚ) : TileIndex (TileShape.reduceShape shape ax keepDims) → ℚ :=
  match keepDims with
  | .false => fun out =>
      ((List.finRange _).map fun k => v (TileShape.insertAxisIndex shape ax out k)).sum
  | .true => fun out =>
      ((List.finRange _).map fun k => v (TileShape.replaceAxisIndex shape ax out k)).sum

private def Q : CAlgebra ℚ where
  literal := fun _ _ q => q
  negInf := 0
  binary := fun _ _ op a b => match op with
    | .add => a + b | .sub => a - b | .mul => a * b | .div => a / b | .max => max a b | .pow => 0
  unary := fun _ _ a => a
  cast := fun _ _ _ a => a
  fromNat := fun _ n => n
  fromInt := fun _ i => i
  fp32Bits := fun _ => 0
  fp32Load := id
  reduceMax := fun _ {_} _ _ _ _ => 0
  reduceSum := fun _ => sumAxis

-- Compute fp32 loads/stores use the erased `.real` cell tag.
private def initial (input : RegionName → Nat → ℚ) (pid : Nat := 0) : State ℚ where
  init := fun r o => .mk .real (input r o)
  pids := fun _ => pid
  numPids := fun _ => 2
  undef := fun d _ _ => Structural.defaultValue d

private def output (s : Option (State ℚ)) (region : RegionName) (n : Nat) : Option (List ℚ) :=
  s.map fun t => (List.range n).map fun i => (t.read region i).read .real

reflect_kernel VectorAdd.Kernels.originalKernel
reflect_kernel RowWiseSum.Kernels.rowWiseSumKernel

private def vectorState : State ℚ := initial fun r o =>
  if r == "x" then [1/2, -2, 3, 4].getD o 0
  else if r == "y" then [1/2, 7, 2, -1].getD o 0
  else -99

private def vectorResult : Option (State ℚ) :=
  interp Q (VectorAdd.Kernels.originalKernel.mirror 4) vectorState

-- x + y = [1, 5, 5, 3], with the first unwritten cell retained.
#eval output vectorResult "out" 5
#guard output vectorResult "out" 5 = some [1, 5, 5, 3, -99]
#guard output vectorResult "x" 4 = some [1/2, -2, 3, 4]
#guard output vectorResult "y" 4 = some [1/2, 7, 2, -1]
-- Program id is supplied by the caller: program 1 writes only offsets 2 and 3.
#guard output (interp Q (VectorAdd.Kernels.originalKernel.mirror 2)
    { vectorState with pids := fun _ => 1 }) "out" 5 = some [-99, -99, 5, 3, -99]

private def rowKernel := RowWiseSum.Kernels.rowWiseSumKernel.mirror "x" "y" 3 3
private def rowResult : Option (State ℚ) := do
  let s ← interp Q rowKernel (initial fun r o => if r == "x" then o + 1 else -99)
  interp Q rowKernel { s with pids := fun _ => 1 }

-- [[1, 2, 3], [4, 5, 6]] sums to [6, 15]; no third row is written.
#eval output rowResult "y" 3
#guard output rowResult "y" 3 = some [6, 15, -99]
#guard output rowResult "x" 6 = some [1, 2, 3, 4, 5, 6]

-- Both kernel entrypoints execute the same code, including an empty kernel.
#guard output (exec Q (VectorAdd.Kernels.originalKernel.mirror 4) vectorState) "out" 4
    = some [1, 5, 5, 3]
#guard output (interp Q (.mk [] [] []) vectorState) "out" 2 = some [-99, -99]
#guard (exec Q (.mk [] [] [.effectMarker "unsupported"]) vectorState).isNone

-- A truly optional operation is absent; the kernel fails, with no fallback.
private def needsFma : QKernel := .mk [] []
  [.assign .real [] "a" (.fma .fp32 .nil .nil (.alg (.const 1))
    (.alg (.const 2)) (.alg (.const 3)))]
#guard (interp Q needsFma vectorState).isNone
-- A store whose address cannot be evaluated also fails.
#guard (interp Q (.mk [] [] [.store .nat [] (.region "x" (.ref .nat [] "unbound"))
    (.alg (.constNat 1)) .none]) vectorState).isNone

end VeriTile.Tests.InterpEvalSmoke
