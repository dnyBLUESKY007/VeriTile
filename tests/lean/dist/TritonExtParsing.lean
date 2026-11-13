import VeriTile.Dist.Frontend.TritonExt.Range

namespace VeriTile.Tests.Dist.TritonExtParsing
open VeriTile Triton

def rangeTest : ComputeKernel := triton {
  for i in tl.range(1, 4, 1) {
    x = i + 1
  }
}

example : rangeTest = (triton {
  for i in range(1, 4, 1) {
    x = i + 1
  }
} : ComputeKernel) := rfl

private def profile : CallSignature where
  constexpr := ["N"]

def namedStep (N : Nat) : ComputeKernel :=
  with_triton_signature profile in triton {
    for i in tl.range(0, N, step=1, loop_unroll_factor=1):
      x = i + 1
  }

def pythonStep (N : Nat) : ComputeKernel :=
  with_triton_signature profile in triton {
    for i in range(0, N, 1):
      x = i + 1
  }

example (N : Nat) : namedStep N = pythonStep N := rfl

-- Pinned range documentation says factors below two disable unrolling;
-- larger factors are also hints and leave the sequential program unchanged.
example : (triton {
  for i in tl.range(4, loop_unroll_factor=0) { x = i }
  for j in tl.range(4, loop_unroll_factor=4) { y = j }
} : ComputeKernel) = triton {
  for i in range(4) { x = i }
  for j in range(4) { y = j }
} := rfl

-- All positional arities, nesting, a runtime bound and a runtime stride use
-- precisely the existing range AST; no trip count is guessed by the adapter.
example : (triton {
  stop = tl.num_programs(0)
  stride = tl.program_id(0) + 1
  for i in tl.range(stop) {
    for j in tl.range(1, stop) {
      for k in tl.range(1, stop, stride) { x = i + j + k }
    }
  }
} : ComputeKernel) = triton {
  stop = tl.num_programs(0)
  stride = tl.program_id(0) + 1
  for i in range(stop) {
    for j in range(1, stop) {
      for k in range(1, stop, stride) { x = i + j + k }
    }
  }
} := rfl

-- The target also uses the Python range spelling with a named step. Both
-- keyword orders and the call-binding entrypoint preserve constexpr lowering.
example (N : Nat) : (with_triton_call profile call_args [] in triton {
  for i in tl.range(0, N, loop_unroll_factor=1, step=1):
    for j in range(0, N, step=1):
      x = i + j
} : ComputeKernel) = with_triton_call profile call_args [] in triton {
  for i in range(0, N, 1):
    for j in range(0, N, 1):
      x = i + j
} := rfl

example (N : Nat) : (with_triton_signature profile in triton {
  for i in tl.range(tl.cast(0, tl.int64), tl.cast(N, tl.int64), tl.cast(1, tl.int64)):
    x = i
} : ComputeKernel) = with_triton_signature profile in triton {
  for i in range(tl.cast(0, tl.int64), tl.cast(N, tl.int64), tl.cast(1, tl.int64)):
    x = i
} := rfl

/-- error: duplicate tl.range step -/
#guard_msgs in
example : ComputeKernel := triton {
  for i in tl.range(0, 4, step=1, step=2) { x = i }
}

/-- error: duplicate tl.range loop_unroll_factor -/
#guard_msgs in
example : ComputeKernel := triton {
  for i in tl.range(4, loop_unroll_factor=1, loop_unroll_factor=1) { x = i }
}

/-- error: positional tl.range bound after keyword -/
#guard_msgs in
example : ComputeKernel := triton {
  for i in tl.range(0, step=1, 4) { x = i }
}

/-- error: tl.range requires one to three bounds and at most one step -/
#guard_msgs in
example : ComputeKernel := triton {
  for i in tl.range(0, 4, 1, step=1) { x = i }
}

-- Capture parser errors inside elaboration so #guard_msgs can check them;
-- putting unparseable syntax directly after #guard_msgs aborts command parsing.
private def parseBody (body : String) : Lean.Elab.Command.CommandElabM Unit := do
  match Lean.Parser.runParserCategory (← Lean.getEnv) `term ("triton { " ++ body ++ " }") with
  | .ok _ => pure ()
  | .error message => throwError "{message}"

/-- error: <input>:1:40: expected ')' -/
#guard_msgs in
run_cmd parseBody "for i in tl.range(4, num_stages=2) { x = i }"

/-- error: <input>:1:45: expected ')' -/
#guard_msgs in
run_cmd parseBody "for i in tl.range(4, warp_specialize=True) { x = i }"

/-- error: <input>:1:34: expected ')' -/
#guard_msgs in
run_cmd parseBody "for i in tl.range(4, arg2=8) { x = i }"

/-- error: <input>:1:55: expected ')' -/
#guard_msgs in
run_cmd parseBody "for i in tl.range(4, disallow_acc_multi_buffer=True) { x = i }"

/-- error: <input>:1:37: expected ')' -/
#guard_msgs in
run_cmd parseBody "for i in tl.range(4, flatten=True) { x = i }"

/-- error: <input>:1:42: expected ')' -/
#guard_msgs in
run_cmd parseBody "for i in tl.range(4, disable_licm=True) { x = i }"

/-- error: <input>:1:9: expected '}' -/
#guard_msgs in
run_cmd parseBody "break"

/-- error: <input>:1:9: expected '}' -/
#guard_msgs in
run_cmd parseBody "continue"

example : (triton {
  pass
  for i in tl.range(4) { pass }
} : ComputeKernel) = triton { for i in range(4) { } } := rfl

example : (triton {
  v = tl.full([4], 1.0)
  x = v.sum(axis=0)
} : ComputeKernel) = triton {
  v = tl.full([4], 1.0)
  x = tl.sum(v, axis=0)
} := rfl

-- Real Python source spelling, including the receiver that Lean lexes as a
-- dotted identifier, must work through call-source normalization as well.
example : (with_triton_call ({} : CallSignature) call_args [] in triton {
  v = tl.full([4], 1.0)
  pass
  x = v.sum(axis=0)
} : ComputeKernel) = with_triton_call ({} : CallSignature) call_args [] in triton {
  v = tl.full([4], 1.0)
  x = tl.sum(v, axis=0)
} := rfl

example : (triton {
  v = tl.full([4], 1.0)
  x = (v + v).sum(axis=0)
} : ComputeKernel) = triton {
  v = tl.full([4], 1.0)
  x = tl.sum((v + v), axis=0)
} := rfl

example (N : Nat) : (with_triton_signature profile in triton {
  i = 0
  while i < N:
    pass
    i += 1
} : ComputeKernel) = with_triton_signature profile in triton {
  i = 0
  while i < N:
    i += 1
} := rfl

-- Non-affine polling control still needs the separately tracked Dist control
-- representation; accepting pass must not turn such a loop into a no-op.
/-- error: affine while requires a scalar counter < unchanged bound -/
#guard_msgs in
example : ComputeKernel := with_triton_signature ({} : CallSignature) in triton {
  while 1 != 0:
    pass
}

-- The conditional census rows already use existing annotation/augmentation
-- forms. Exercise them with a new spelling so the adapter cannot discard them.
example (N : Nat) : (with_triton_signature profile in triton {
  count: tl.constexpr = N
  x = 4
  for i in tl.range(count):
    x += 2
    x -= 1
} : ComputeKernel) = with_triton_signature profile in triton {
  count: tl.constexpr = N
  x = 4
  for i in range(count):
    x += 2
    x -= 1
} := rfl

end VeriTile.Tests.Dist.TritonExtParsing
