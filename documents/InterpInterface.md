# Interpreter interface

Status: draft of ticket I0.1, **not signed**. The Lean side lives in
`VeriTile/Staging/Interp/`; nothing here is a proved result yet.

The interpreter line builds one executable definition of kernel execution and
connects it to the semantics the theorems are about:

```text
GPU run  ≈  interp (some carrier)                 compared on tensor files
interp   =  FP.Structural.exec                    adequacy theorem
FP.Structural.exec (real algebra)  =  exec ∘ toAlgorithm?    bridge theorem
```

The first line is an experiment and never defines semantics. The other two are
theorems; their statements are frozen by this ticket.

## 1. Names and signatures

All names are in the namespace `VeriTile.Interp`. A stub is declared `opaque`
or with a `sorry` body in `VeriTile/Staging/Interp/Interface.lean`; the ticket
in the last column replaces it by a definition with the same name and type in
the main target, and deletes the stub in the same change.

| Name | Type | Meaning | Defined by |
| --- | --- | --- | --- |
| `QOp`, `QExpr`, `QStmt` | types | Mirrors of the upstream AST families whose floating literals are rationals instead of reals | I1.1 |
| `QKernel` | `Type` | Mirror of `ComputeKernel` | I1.1 |
| `QKernel.toReal` | `QKernel → ComputeKernel` | Replaces every rational literal `q` by the real `(q : ℝ)` and changes nothing else | I1.1 |
| `CAlgebra α` | `Type` | The operations of `FP.Structural.Algebra α`, except that `literal` takes a rational; every field is computable | I2.2 |
| `CAlgebra.lift` | `CAlgebra α → FP.Structural.Algebra α` | Same operations; on a real literal that is the image of a rational `q` it returns the `CAlgebra` literal of `q` | I2.2 |
| `State α` | `Type` | Executable state: registers, finite-map memory of typed cells, program ids, grid sizes, undefined-value table | I2.1 |
| `State.lift` | `State α → FP.Structural.State α` | The function-valued state that the executable state represents | I2.1 |
| `interp` | `[Inhabited α] → CAlgebra α → QKernel → State α → Option (State α)` | Executes one program instance; `none` is execution failure | I2.4 |
| `erase` | `FP.Structural.State ℝ → BlockState` | Forgets dtype tags: the state of the proof semantics that a real-carrier state stands for | I4.1 |
| `realAlgebra` | `FP.Structural.Algebra ℝ` | Literals as given, real arithmetic, casts and rounding as identity | I4.1 |
| `BridgePre` | `ComputeKernel → FP.Structural.State ℝ → Prop` | The named side conditions under which the two execution definitions agree | I4.2 |

`interp` executes one program instance of a launch, as `FP.Structural.exec`
does. Looping over the program ids of a grid belongs to the runner (I7.2).

## 2. Frozen statements

Adequacy (`VeriTile/Staging/Interp/Adequacy.lean`, proved by I3.3):

```lean
theorem interp_eq_structural_exec {α : Type} [Inhabited α]
    (M : CAlgebra α) (kq : QKernel) (s : State α) :
    (interp M kq s).map State.lift
      = FP.Structural.exec M.lift kq.toReal s.lift
```

For every carrier and every algebra, the interpreter and the structural
execution fail together or succeed with the same final state.

Bridge (`VeriTile/Staging/Interp/Bridge.lean`, proved by I4.3):

```lean
theorem bridge_exec {ck : ComputeKernel} {ak : AlgKernel}
    (s : FP.Structural.State ℝ)
    (h : ck.toAlgorithm? = .ok ak) (pre : BridgePre ck s) :
    (FP.Structural.exec realAlgebra ck s).map erase = exec ak (erase s)
```

For a source kernel whose dtype-erasing projection succeeds, structural
execution over the reals agrees with the proof semantics after erasure. The
statement covers only kernels whose projection succeeds.

`BridgePre` is part of the statement on purpose. Differences between the two
definitions (propagation of `⊥`, undefined values and `other`, missing backend
support, constructs whose projection fails) are expected to surface during the
proof. Each one becomes a named conjunct of `BridgePre` and an entry in
`documents/BridgeTheoremDifferences.md`, so the theorem's wording stays as
frozen and the conditions are reviewed where they are defined. `BridgePre` must
not be strengthened to something no bench kernel satisfies: I4.3 ships a
kernel and a state that satisfy it.

What the two statements do not say:

- Nothing about a GPU. Agreement with GPU runs is measured, not proved.
- Nothing about a particular carrier being IEEE-correct. `interp` over Lean
  `Float` approximates the real semantics; the bit-level carrier is the
  collaborator's model.
- Nothing about kernels that use constructs `FP.Structural.exec` rejects; on
  those both sides of the adequacy statement are `none`.

## 3. Tensor file format, version 1

A `.tensor` file holds one tensor. The development host, the Lean runner and
the GPU side all read and write exactly these bytes, and a tensor's `sha256`
in a request or a result is the SHA-256 of the whole file.

### Layout

All integers are little-endian.

| Offset | Size | Field |
| --- | --- | --- |
| 0 | 8 | Magic: the ASCII bytes `VTTENSR1` |
| 8 | 1 | Dtype code (table below) |
| 9 | 1 | Rank `r`, from 0 to 8 |
| 10 | 6 | Reserved, all zero |
| 16 | 8·r | Dimensions, one unsigned 64-bit integer each |
| 16 + 8·r | n·w | Elements in row-major order: the last index varies fastest |

`n` is the product of the dimensions (1 for rank 0) and `w` the element width.
The file length is exactly `16 + 8·r + n·w`. A reader rejects a wrong magic,
an unknown dtype code, a rank above 8, a non-zero reserved byte and any other
length.

### Dtypes

The name is the one used in `batch.json`.

| Code | Name | Width | Element bytes |
| --- | --- | --- | --- |
| 1 | `fp32` | 4 | IEEE 754 binary32 bit pattern |
| 2 | `fp16` | 2 | IEEE 754 binary16 bit pattern |
| 3 | `bf16` | 2 | bfloat16 bit pattern (the upper 16 bits of a binary32) |
| 4 | `fp64` | 8 | IEEE 754 binary64 bit pattern |
| 5 | `f8e4` | 1 | FP8 E4M3 bit pattern |
| 6 | `f8e5` | 1 | FP8 E5M2 bit pattern |
| 16 | `int8` | 1 | Two's complement |
| 17 | `int16` | 2 | Two's complement |
| 18 | `int32` | 4 | Two's complement |
| 19 | `int64` | 8 | Two's complement |
| 24 | `uint8` | 1 | Unsigned |
| 25 | `uint16` | 2 | Unsigned |
| 26 | `uint32` | 4 | Unsigned |
| 27 | `uint64` | 8 | Unsigned |
| 32 | `bool` | 1 | `0` or `1`; a reader rejects any other byte |

Floating elements are stored as bit patterns and are never normalised: the
sign of zero and the payload of a NaN survive a round trip.

### Sentinel fill

Before a kernel runs, each output buffer is filled with one sentinel bit
pattern, given in the request as a hexadecimal string of exactly `2·w` digits
after `0x`, most significant digit first (`0x7fc0dead` for `fp32`). After the
run, an element whose bytes still equal the sentinel counts as not written.
The write footprint of two runs is compared exactly on that basis.

Defaults when a request names none:

| Dtype | Sentinel | Kind |
| --- | --- | --- |
| `fp32` | `0x7fc0dead` | Quiet NaN with payload |
| `fp16` | `0x7ead` | Quiet NaN with payload |
| `bf16` | `0x7fcd` | Quiet NaN with payload |
| `fp64` | `0x7ff8dead0000dead` | Quiet NaN with payload |
| `f8e4` | `0x7f` | NaN |
| `f8e5` | `0x7f` | NaN |
| signed integers | `0x80…0` (the minimum value) | |
| unsigned integers | `0xdead…` truncated to the width, `0xad` for 8 bits | |
| `bool` | none: a boolean output needs a wider shadow buffer | |

Two limits follow from the method and are accepted. A kernel that writes the
sentinel bit pattern itself is indistinguishable from one that does not
write; generated inputs therefore never contain a sentinel pattern, and every
NaN they contain is the canonical quiet NaN (`0x7fc00000` for `fp32`). An
integer sentinel can collide with a legitimate result, so a request for an
integer output states a sentinel outside the kernel's result range.

### Not fixed here

- How inputs are generated from a seed: owned by `scripts/interp/gen_inputs.py` (I8.1), which must produce identical bytes on every machine.
- The directory layout of requests and results: `documents/GpuExchangeProtocol.md`.
- The export format of the symbolic carrier (I6.3).
