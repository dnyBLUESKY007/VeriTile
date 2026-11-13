/-
Descriptor call sites from Triton-distributed 7908e4ea9010bb2238f9f05b28c904fd4b71c60a:
- tutorials/07-overlapping-allgather-gemm.py:107–128,190–206
- tutorials/08-overlapping-gemm-reduce-scatter.py:167–188,241–257
- python/triton_dist/kernels/nvidia/allgather_gemm.py:220–241,289–304
- python/triton_dist/kernels/nvidia/gemm_reduce_scatter.py:262–270,302–318
https://github.com/ByteDance-Seed/Triton-distributed/tree/7908e4ea9010bb2238f9f05b28c904fd4b71c60a
Read 2026-11-13. Constructors, a/b loads and c stores are verbatim; host dimensions
and synthetic c loads supply intermediate values outside the distributed kernel.
-/
import VeriTile.Dist.Frontend.TritonExt.TensorDescriptor

namespace VeriTile.Tests.Dist.TensorDescriptorSource
open VeriTile Triton

private def signature : CallSignature where
  pointers := [("a_ptr", .fp32), ("b_ptr", .fp32), ("c_ptr", .fp32)]
  scalars := []
  constexpr := ["M", "N", "K", "BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K"]

def tutorialCalls (a_ptr b_ptr c_ptr : RegionName) (EPILOGUE_SUBTILE : Bool) : ComputeKernel :=
  let M := 8
  let N := 8
  let K := 8
  let BLOCK_SIZE_M := 2
  let BLOCK_SIZE_N := 4
  let BLOCK_SIZE_K := 4
  with_triton_signature signature in triton {
    a_desc = tl.make_tensor_descriptor(
        a_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[
            BLOCK_SIZE_M,
            BLOCK_SIZE_N if not EPILOGUE_SUBTILE else BLOCK_SIZE_N // 2,
        ],
    )
    offs_am = tl.program_id(0) * BLOCK_SIZE_M
    offs_bn = tl.program_id(1) * BLOCK_SIZE_N
    offs_k = tl.program_id(2) * BLOCK_SIZE_K
    a = a_desc.load([offs_am, offs_k])
    b = b_desc.load([offs_bn, offs_k])
    c0 = c_desc.load([offs_am, offs_bn])
    c1 = c_desc.load([offs_am, offs_bn])
    c = c_desc.load([offs_am, offs_bn])
    c_desc.store([offs_am, offs_bn], c0)
    c_desc.store([offs_am, offs_bn + BLOCK_SIZE_N // 2], c1)
    c_desc.store([offs_am, offs_bn], c)
  }

/-- The actual descriptor methods and their free-function spellings agree. -/
def methods (x : Region .fp32) : ComputeKernel := triton {
  d = tl.make_tensor_descriptor(x, shape=[8, 8], strides=[8, 1], block_shape=[2, 4])
  row = tl.program_id(0) * 2
  v = d.load([row, 4])
  d.store([row, 0], v)
}

def functions (x : Region .fp32) : ComputeKernel := triton {
  d = tl.make_tensor_descriptor(x, shape=[8, 8], strides=[8, 1], block_shape=[2, 4])
  row = tl.program_id(0) * 2
  v = tl.load_tensor_descriptor(d, [row, 4])
  tl.store_tensor_descriptor(d, [row, 0], v)
}

def blockPointers (x : Region .fp32) : ComputeKernel := triton {
  d = tl.make_block_ptr(base=x, shape=(8, 8), strides=(8, 1), offsets=(0, 0), block_shape=(2, 4))
  row = tl.program_id(0) * 2
  v = tl.load(tl.advance(d, [row, 4]), boundary_check=([0, 1] : List Nat), padding_option="zero")
  tl.store(tl.advance(d, [row, 0]), v, boundary_check=([0, 1] : List Nat))
}

example (x : Region .fp32) : methods x = functions x := rfl
-- Worked case: every statement, port, dtype and boundary check agrees. Thus the
-- descriptor load is the corresponding tl.load, not just a matching output.
example (x : Region .fp32) : methods x = blockPointers x := rfl

-- A wrong offset is a different program, even when tile shapes agree.
def wrongOffset (x : Region .fp32) : ComputeKernel := triton {
  d = tl.make_block_ptr(base=x, shape=(8, 8), strides=(8, 1), offsets=(0, 0), block_shape=(2, 4))
  row = tl.program_id(0) * 2
  v = tl.load(tl.advance(d, [row, 0]), boundary_check=([0, 1] : List Nat), padding_option="zero")
  tl.store(tl.advance(d, [row, 0]), v, boundary_check=([0, 1] : List Nat))
}
example (x : Region .fp32) : methods x ≠ wrongOffset x := by
  intro h
  cases h

/-- error: <input>:2:81: expected ')' -/
#guard_msgs in
example (x : RegionName) : ComputeKernel := with_triton_signature ({} : CallSignature) in triton {
  d = tl.make_tensor_descriptor(x, shape=[8, 8], strides=[8, 1], block_shape=[2, 4], padding_option="zero")
}

/-- error: tensor descriptor: shape, strides and block_shape must have the same rank (2–5) -/
#guard_msgs in
example (x : RegionName) : ComputeKernel := triton {
  d = tl.make_tensor_descriptor(x, shape=[8, 8], strides=[8], block_shape=[2, 4])
}

/-- error: tensor descriptor: one offset is required per dimension -/
#guard_msgs in
example (x : RegionName) : ComputeKernel := triton {
  d = tl.make_tensor_descriptor(x, shape=[8, 8], strides=[8, 1], block_shape=[2, 4])
  v = d.load([0])
}

/-- error: tensor descriptor: receiver must have a make_tensor_descriptor binding -/
#guard_msgs in
example (x : RegionName) : ComputeKernel := triton {
  v = x.load([0, 0])
}

/-- error: tensor descriptor: descriptor names cannot be rebound to other values -/
#guard_msgs in
example (x : RegionName) : ComputeKernel := triton {
  d = tl.make_tensor_descriptor(x, shape=[8, 8], strides=[8, 1], block_shape=[2, 4])
  d = tl.make_block_ptr(base=x, shape=(8, 8, 8), strides=(64, 8, 1), offsets=(0, 0, 0), block_shape=(2, 2, 2))
  v = d.load([0, 0])
}

private def loopSignature : CallSignature where
  pointers := [("x", .fp32)]
  scalars := []
  constexpr := ["M", "N", "BM", "BN"]

def descriptorLoop (x : RegionName) (M N BM BN : Nat) : ComputeKernel :=
  with_triton_signature loopSignature in triton {
    d = tl.make_tensor_descriptor(x, shape=[M, N], strides=[N, 1], block_shape=[BM, BN])
    for row in range(0, M, BM) {
      v = d.load([row, 0])
      d.store([row, 0], v)
    }
  }

def pointerLoop (x : RegionName) (M N BM BN : Nat) : ComputeKernel :=
  with_triton_signature loopSignature in triton {
    d = tl.make_block_ptr(base=x, shape=(M, N), strides=(N, 1), offsets=(0, 0), block_shape=(BM, BN))
    for row in range(0, M, BM) {
      v = tl.load(tl.advance(d, (row, 0)), boundary_check=([0, 1] : List Nat), padding_option="zero")
      tl.store(tl.advance(d, (row, 0)), v, boundary_check=([0, 1] : List Nat))
    }
  }

-- Early lowering preserves signature validation, store conversion and loop specialization.
example (x : RegionName) (M N BM BN : Nat) :
    descriptorLoop x M N BM BN = pointerLoop x M N BM BN := rfl

/-- error: pointer parameter x is missing from the call signature -/
#guard_msgs in
example (x : RegionName) : ComputeKernel := with_triton_signature ({} : CallSignature) in triton {
  d = tl.make_tensor_descriptor(x, shape=[8, 8], strides=[8, 1], block_shape=[2, 4])
  v = d.load([0, 0])
}

-- The call-argument entrypoint uses the same signature-aware lowering.
def descriptorCall (x : RegionName) : ComputeKernel :=
  with_triton_call ({pointers := [("x", .fp32)]} : CallSignature) call_args [] in triton {
    d = tl.make_tensor_descriptor(x, shape=[8, 8], strides=[8, 1], block_shape=[2, 4])
    v = tl.load_tensor_descriptor(d, [0, 0])
    tl.store_tensor_descriptor(d, [0, 0], v)
  }

def pointerCall (x : RegionName) : ComputeKernel :=
  with_triton_call ({pointers := [("x", .fp32)]} : CallSignature) call_args [] in triton {
    d = tl.make_block_ptr(base=x, shape=(8, 8), strides=(8, 1), offsets=(0, 0), block_shape=(2, 4))
    v = tl.load(tl.advance(d, (0, 0)), boundary_check=([0, 1] : List Nat), padding_option="zero")
    tl.store(tl.advance(d, (0, 0)), v, boundary_check=([0, 1] : List Nat))
  }
example (x : RegionName) : descriptorCall x = pointerCall x := rfl

-- Triton v3.5.1 core.py:2073 supports ranks 2–5; exercise the upper bound.
example (x : Region .fp32) : ComputeKernel := triton {
  d = tl.make_tensor_descriptor(x, shape=[8, 8, 8, 8, 8], strides=[4096, 512, 64, 8, 1], block_shape=[2, 2, 2, 2, 4])
  v = d.load([0, 0, 0, 0, 0])
  d.store([0, 0, 0, 0, 0], v)
}

/-- error: tensor descriptor: shape, strides and block_shape must have the same rank (2–5) -/
#guard_msgs in
example (x : RegionName) : ComputeKernel := triton {
  d = tl.make_tensor_descriptor(x, shape=[8], strides=[1], block_shape=[4])
}

/-- error: tensor descriptor: a name cannot change rank -/
#guard_msgs in
example (x : RegionName) : ComputeKernel := triton {
  d = tl.make_tensor_descriptor(x, shape=[8, 8], strides=[8, 1], block_shape=[2, 4])
  d = tl.make_tensor_descriptor(x, shape=[8, 8, 8], strides=[64, 8, 1], block_shape=[2, 2, 4])
}

def rebound (x y : Region .fp32) : ComputeKernel := triton {
  d = tl.make_tensor_descriptor(x, shape=[8, 8], strides=[8, 1], block_shape=[2, 4])
  d = tl.make_tensor_descriptor(y, shape=[16, 16], strides=[16, 1], block_shape=[2, 4])
  v = d.load([0, 4])
}
def reboundPointer (x y : Region .fp32) : ComputeKernel := triton {
  d = tl.make_block_ptr(base=x, shape=(8, 8), strides=(8, 1), offsets=(0, 0), block_shape=(2, 4))
  d = tl.make_block_ptr(base=y, shape=(16, 16), strides=(16, 1), offsets=(0, 0), block_shape=(2, 4))
  v = tl.load(tl.advance(d, (0, 4)), boundary_check=([0, 1] : List Nat), padding_option="zero")
}
example (x y : Region .fp32) : rebound x y = reboundPointer x y := rfl

-- Existing list-argument builtins and the admitted scalar source helper still parse.
example (word : Nat) : ComputeKernel := triton {
  zeros = tl.zeros([4])
  full = tl.full([4], 1)
  tl.source_helper uint_to_uniform_float = tl.uint_to_uniform_float
  word_arg = $((word : Nat))
  tl.source_integer word_arg = tl.uint32
  uniform = uint_to_uniform_float(word_arg)
}

end VeriTile.Tests.Dist.TensorDescriptorSource
