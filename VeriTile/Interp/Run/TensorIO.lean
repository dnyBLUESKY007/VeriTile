/-
VeriTile.Interp.Run.TensorIO

Reader and writer for `.tensor` files, version 1. The bytes are frozen in
`documents/InterpInterface.md`, section 3; the Python side of the same format
is `scripts/interp/tensorfile.py`. A tensor is kept as raw element bytes, so a
round trip never changes a bit and nothing here interprets a floating value.

Parsing and serialisation are pure functions on `ByteArray`; the `#guard`
commands at the end are the tests and fail the build when wrong. Lean core
only: no import.
-/

namespace VeriTile.Interp

/-! ### Bytes -/

/-- The `w` little-endian bytes of `n` (of `n` modulo `256 ^ w`). -/
private def leBytes (n w : Nat) : ByteArray :=
  ⟨(Array.range w).map fun i => (n >>> (8 * i)).toUInt8⟩

/-- The number whose little-endian bytes are `bytes`. -/
private def leNat (bytes : ByteArray) : Nat :=
  bytes.data.foldr (fun byte n => 256 * n + byte.toNat) 0

/-- Lowercase hexadecimal digits, two per byte. -/
def hex (bytes : ByteArray) : String :=
  String.ofList (bytes.toList.flatMap fun b =>
    [Nat.digitChar (b.toNat / 16), Nat.digitChar (b.toNat % 16)])

/-! ### SHA-256 (FIPS 180-4)

Neither Lean core nor Mathlib has one, and the format quotes a tensor by the
SHA-256 of its file. -/

private def sha256K : Array UInt32 := #[
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
  0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
  0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
  0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
  0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2]

/-- The 32 bytes of the SHA-256 digest of `data`. -/
def sha256 (data : ByteArray) : ByteArray := Id.run do
  let rotr (x n : UInt32) : UInt32 := (x >>> n) ||| (x <<< (32 - n))
  -- Padding: 0x80, zeros up to 56 modulo 64, then the bit length, big-endian.
  let mut msg := data.push 0x80
  for _ in [0:(119 - data.size % 64) % 64] do msg := msg.push 0
  msg := msg ++ ⟨(leBytes (8 * data.size) 8).data.reverse⟩
  let mut h : Array UInt32 := #[0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]
  for block in [0:msg.size / 64] do
    let mut w : Array UInt32 := (Array.range 16).map fun t =>
      (leNat ⟨(msg.extract (64 * block + 4 * t) (64 * block + 4 * t + 4)).data.reverse⟩).toUInt32
    for t in [16:64] do
      let (x, y) := (w[t - 15]!, w[t - 2]!)
      w := w.push (w[t - 16]! + (rotr x 7 ^^^ rotr x 18 ^^^ (x >>> 3)) + w[t - 7]!
        + (rotr y 17 ^^^ rotr y 19 ^^^ (y >>> 10)))
    let mut s := h
    for t in [0:64] do
      let (a, e) := (s[0]!, s[4]!)
      let t1 := s[7]! + (rotr e 6 ^^^ rotr e 11 ^^^ rotr e 25)
        + ((e &&& s[5]!) ^^^ (~~~e &&& s[6]!)) + sha256K[t]! + w[t]!
      let t2 := (rotr a 2 ^^^ rotr a 13 ^^^ rotr a 22)
        + ((a &&& s[1]!) ^^^ (a &&& s[2]!) ^^^ (s[1]! &&& s[2]!))
      s := #[t1 + t2, a, s[1]!, s[2]!, s[3]! + t1, e, s[5]!, s[6]!]
    h := h.zipWith (· + ·) s
  return h.foldl (fun out x => out ++ ⟨(leBytes x.toNat 4).data.reverse⟩) .empty

#guard hex (sha256 .empty) = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
#guard hex (sha256 "abc".toUTF8)
  = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
#guard hex (sha256 "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq".toUTF8)
  = "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"

/-! ### Dtypes -/

/-- Element types of a tensor file. The constructor names are the names used in `batch.json`. -/
inductive DType where
  | fp32 | fp16 | bf16 | fp64 | f8e4 | f8e5
  | int8 | int16 | int32 | int64
  | uint8 | uint16 | uint32 | uint64
  | bool
  deriving DecidableEq, Repr

namespace DType

/-- Code, element width in bytes and default sentinel of a dtype: the two tables of the format.
`bool` has no sentinel. -/
def info : DType → UInt8 × Nat × Option String
  | fp32 => (1, 4, "0x7fc0dead")
  | fp16 => (2, 2, "0x7ead")
  | bf16 => (3, 2, "0x7fcd")
  | fp64 => (4, 8, "0x7ff8dead0000dead")
  | f8e4 => (5, 1, "0x7f")
  | f8e5 => (6, 1, "0x7f")
  | int8 => (16, 1, "0x80")
  | int16 => (17, 2, "0x8000")
  | int32 => (18, 4, "0x80000000")
  | int64 => (19, 8, "0x8000000000000000")
  | uint8 => (24, 1, "0xad")
  | uint16 => (25, 2, "0xdead")
  | uint32 => (26, 4, "0xdeaddead")
  | uint64 => (27, 8, "0xdeaddeaddeaddead")
  | bool => (32, 1, none)

def code (d : DType) : UInt8 := d.info.1

/-- Element width in bytes. -/
def width (d : DType) : Nat := d.info.2.1

def defaultSentinel (d : DType) : Option String := d.info.2.2

/-- Every dtype, in the order of the format's table. -/
def all : List DType :=
  [fp32, fp16, bf16, fp64, f8e4, f8e5, int8, int16, int32, int64, uint8, uint16, uint32, uint64,
    bool]

theorem mem_all (d : DType) : d ∈ all := by cases d <;> decide

def ofCode? (code : UInt8) : Option DType := all.find? (·.code == code)

end DType

/-! ### Tensors -/

/-- One tensor: `data` holds the elements in row-major order, each little-endian. -/
structure Tensor where
  dtype : DType
  shape : List Nat
  data : ByteArray
  deriving DecidableEq

/-- Number of elements of a shape: the product of the dimensions, 1 for rank 0. -/
def elementCount (shape : List Nat) : Nat := shape.foldl (· * ·) 1

private def magic : ByteArray := "VTTENSR1".toUTF8

namespace Tensor

/-- The conditions under which a tensor has a file: rank at most 8, every dimension an unsigned
64-bit integer, exactly `n·w` data bytes, and only 0 or 1 in a `bool` tensor. -/
def check (t : Tensor) : Except String Unit := do
  unless t.shape.length ≤ 8 ∧ t.shape.all (· < 2 ^ 64) do throw s!"unsupported shape {t.shape}"
  let size := elementCount t.shape * t.dtype.width
  unless t.data.size = size do
    throw s!"shape {t.shape} at width {t.dtype.width} needs {size} data bytes, got {t.data.size}"
  if t.dtype = .bool ∧ t.data.data.any (· > 1) then throw "a bool element is neither 0 nor 1"

/-- The bytes of the tensor's file. -/
def toBytes (t : Tensor) : Except String ByteArray := do
  t.check
  let header := (magic.push t.dtype.code).push t.shape.length.toUInt8 ++ leBytes 0 6
  return t.shape.foldl (fun out dim => out ++ leBytes dim 8) header ++ t.data

/-- Parse the bytes of a tensor file. With `sha256?` (lowercase hexadecimal, of the whole file) or
`expected?` (dtype and shape), a file that differs from either is rejected. -/
def ofBytes (raw : ByteArray) (sha256? : Option String := none)
    (expected? : Option (DType × List Nat) := none) : Except String Tensor := do
  if let some digest := sha256? then
    unless hex (sha256 raw) = digest do throw s!"sha256 differs from the expected {digest}"
  unless raw.size ≥ 16 ∧ raw.extract 0 8 = magic do throw "missing VTTENSR1 magic"
  let some dtype := DType.ofCode? raw[8]! | throw s!"unknown dtype code {raw[8]!}"
  let rank := raw[9]!.toNat
  unless rank ≤ 8 do throw s!"rank {rank} is above 8"
  unless raw.extract 10 16 = leBytes 0 6 do throw "non-zero reserved byte"
  unless raw.size ≥ 16 + 8 * rank do throw "truncated dimensions"
  let shape := (List.range rank).map fun i => leNat (raw.extract (16 + 8 * i) (24 + 8 * i))
  let t : Tensor := ⟨dtype, shape, raw.extract (16 + 8 * rank) raw.size⟩
  t.check
  if let some (dtype, shape) := expected? then
    unless t.dtype = dtype do throw "dtype differs from the expected one"
    unless t.shape = shape do throw s!"shape {t.shape} differs from the expected {shape}"
  return t

/-- Read a tensor file; the optional arguments are those of `ofBytes`. -/
def read (path : System.FilePath) (sha256? : Option String := none)
    (expected? : Option (DType × List Nat) := none) : IO Tensor := do
  IO.ofExcept ((ofBytes (← IO.FS.readBinFile path) sha256? expected?).mapError
    fun e => s!"{path}: {e}")

/-- Write a tensor file and return its SHA-256, the value quoted in requests and results. -/
def write (path : System.FilePath) (t : Tensor) : IO String := do
  let raw ← IO.ofExcept (t.toBytes.mapError fun e => s!"{path}: {e}")
  IO.FS.writeBinFile path raw
  return hex (sha256 raw)

end Tensor

/-! ### Sentinel fill -/

/-- Element bytes of a sentinel given as `0x` and exactly `2·w` hexadecimal digits, most
significant first; without `sentinel?`, the default of the dtype. -/
def sentinelBytes (dtype : DType) (sentinel? : Option String := none) :
    Except String ByteArray := do
  let some text := sentinel? <|> dtype.defaultSentinel
    | throw "bool has no sentinel; use a wider shadow buffer"
  let digits? := match text.toList with
    | '0' :: 'x' :: digits =>
      if digits.length = 2 * dtype.width then
        digits.mapM fun c => "0123456789abcdef".toList.idxOf? (Char.toLower c)
      else none
    | _ => none
  let some digits := digits?
    | throw s!"sentinel {text} is not 0x followed by {2 * dtype.width} hexadecimal digits"
  return leBytes (digits.foldl (16 * · + ·) 0) dtype.width

/-- An output buffer in which every element is the sentinel. -/
def Tensor.filled (dtype : DType) (shape : List Nat) (sentinel? : Option String := none) :
    Except String Tensor := do
  let mark ← sentinelBytes dtype sentinel?
  return ⟨dtype, shape, Nat.repeat (· ++ mark) (elementCount shape) .empty⟩

/-! ### Tests

The expected hexadecimal strings marked "Python" were printed by
`scripts/interp/tensorfile.py` and `scripts/interp/gen_inputs.py`. -/

private def failsWith {α : Type} (result : Except String α) (reason : String) : Bool :=
  match result with
  | .error e => e == reason
  | .ok _ => false

/-- The file of a tensor, empty when it has none. -/
private def file (t : Tensor) : ByteArray := t.toBytes.toOption.getD .empty

/-- The tensor of the round-trip test in `scripts/interp/test_gen_inputs.py`. -/
private def sample (d : DType) : Tensor :=
  ⟨d, [2, 3], if d = .bool then ⟨#[1, 0, 1, 1, 0, 0]⟩
    else ⟨(Array.range (6 * d.width)).map (·.toUInt8 + 1)⟩⟩

-- Round trip on every dtype, also with the right hash, dtype and shape supplied.
#guard DType.all.all fun d =>
  let raw := file (sample d)
  raw.size = 32 + 6 * d.width ∧ (Tensor.ofBytes raw).toOption = some (sample d)
    ∧ (Tensor.ofBytes raw (hex (sha256 raw)) (some (d, [2, 3]))).toOption = some (sample d)

-- Rank 0, an empty tensor with the largest dimension, rank 8.
#guard [⟨.int8, [], ⟨#[0xff]⟩⟩, ⟨.fp64, [2 ^ 64 - 1, 0], .empty⟩,
    ⟨.bool, List.replicate 8 1, ⟨#[1]⟩⟩].all
  fun t => (Tensor.ofBytes (file t)).toOption = some t

-- Python: SHA-256 of the fifteen sample files concatenated in table order, so every code, every
-- width and the layout agree.
#guard hex (sha256 (DType.all.foldl (fun out d => out ++ file (sample d)) .empty))
  = "f7047d4e1be39d3f94baf9ecd47c00f9a18be7439099ac23e6fc0a99bb89b6bb"

-- Python: `gen_inputs.py --dtype fp32 --shape 3 --class normal --seed 1`, file bytes and hash.
#guard
  let raw := file ⟨.fp32, [3],
    ⟨#[0x00, 0x9d, 0x2a, 0x3f, 0x00, 0xf0, 0x63, 0xbd, 0x00, 0x5c, 0x0b, 0x3f]⟩⟩
  hex raw = "565454454e53523101010000000000000300000000000000009d2a3f00f063bd005c0b3f"
    ∧ hex (sha256 raw) = "13a74f020b5287d4db66f83e9e98fa309bf801fef0a0b854ada13d3f0a2b80c5"

-- Python: `gen_inputs.py --dtype fp64 --class normal --seed 6`, a `GOLDEN` entry of the test file.
#guard
  let raw := file ⟨.fp64, [], ⟨#[0x00, 0x00, 0x00, 0x00, 0x00, 0x5f, 0xe2, 0x3f]⟩⟩
  hex raw = "565454454e535231040000000000000000000000005fe23f"
    ∧ hex (sha256 raw) = "25d1aa3e427b2b8f287a635ae150858a9f8e12d601567c5fa4f7af3a788010ea"

-- Python: `tensorfile.filled('uint16', (2, 2))`, file bytes and hash.
#guard
  let raw := ((Tensor.filled .uint16 [2, 2]).toOption.map file).getD .empty
  hex raw = "565454454e535231190200000000000002000000000000000200000000000000addeaddeaddeadde"
    ∧ hex (sha256 raw) = "f398815af2700cb080e066c73f8ff496df78688be479c78da026cf77904ae40f"

-- Python: the default sentinels of the fourteen dtypes that have one, as element bytes.
#guard String.join (DType.all.map fun d => ((sentinelBytes d).toOption.map hex).getD "")
  = "addec07fad7ecd7fadde0000addef87f7f7f80008000000080000000000000008" ++
    "0adaddeaddeaddeaddeaddeaddeadde"

-- Sentinels: explicit, either letter case, and the malformed ones of the Python test.
#guard ((Tensor.filled .fp32 [4]).toOption.map (hex ·.data))
  = some "addec07faddec07faddec07faddec07f"
#guard ((Tensor.filled .int16 [1] "0x1234").toOption.map (hex ·.data)) = some "3412"
#guard ((sentinelBytes .uint16 "0xDeAd").toOption.map hex) = some "adde"
#guard ["0x12", "1234", "0x12345", "0x12g4", "0X1234"].all fun text =>
  failsWith (sentinelBytes .int16 text)
    s!"sentinel {text} is not 0x followed by 4 hexadecimal digits"
#guard failsWith (Tensor.filled .bool [1]) "bool has no sentinel; use a wider shadow buffer"

/-- A well-formed 30-byte file: `int16`, shape `[3]`. -/
private def good : ByteArray := file ⟨.int16, [3], ⟨#[1, 2, 3, 4, 5, 6]⟩⟩

-- What the reader rejects.
#guard (Tensor.ofBytes good).isOk
#guard failsWith (Tensor.ofBytes (good.set! 7 0x32)) "missing VTTENSR1 magic"
#guard failsWith (Tensor.ofBytes (good.extract 0 15)) "missing VTTENSR1 magic"
#guard failsWith (Tensor.ofBytes (good.set! 8 7)) "unknown dtype code 7"
#guard failsWith (Tensor.ofBytes (good.set! 9 9)) "rank 9 is above 8"
#guard failsWith (Tensor.ofBytes (good.set! 12 1)) "non-zero reserved byte"
#guard failsWith (Tensor.ofBytes (good.set! 9 2)) "truncated dimensions"
#guard failsWith (Tensor.ofBytes (good.extract 0 29))
  "shape [3] at width 2 needs 6 data bytes, got 5"
#guard failsWith (Tensor.ofBytes (good.push 0)) "shape [3] at width 2 needs 6 data bytes, got 7"
#guard failsWith (Tensor.ofBytes (good.set! 16 4)) "shape [4] at width 2 needs 8 data bytes, got 6"
#guard failsWith (Tensor.ofBytes ((file ⟨.uint8, [1], ⟨#[2]⟩⟩).set! 8 32))
  "a bool element is neither 0 nor 1"

-- A wrong SHA-256 (of a file that differs in one bit, or simply another value), a wrong dtype, a
-- wrong shape.
#guard failsWith (Tensor.ofBytes (good.set! 29 7) (hex (sha256 good)))
  s!"sha256 differs from the expected {hex (sha256 good)}"
#guard failsWith (Tensor.ofBytes good ("".pushn '0' 64))
  s!"sha256 differs from the expected {"".pushn '0' 64}"
#guard (Tensor.ofBytes good (hex (sha256 good)) (some (.int16, [3]))).isOk
#guard failsWith (Tensor.ofBytes good none (some (.uint16, [3])))
  "dtype differs from the expected one"
#guard failsWith (Tensor.ofBytes good none (some (.int16, [1, 3])))
  "shape [3] differs from the expected [1, 3]"

-- What the writer rejects.
#guard failsWith (Tensor.toBytes ⟨.uint8, List.replicate 9 1, ⟨#[0]⟩⟩)
  "unsupported shape [1, 1, 1, 1, 1, 1, 1, 1, 1]"
#guard failsWith (Tensor.toBytes ⟨.uint8, [2 ^ 64, 0], .empty⟩)
  "unsupported shape [18446744073709551616, 0]"
#guard failsWith (Tensor.toBytes ⟨.fp32, [2], ⟨#[0, 0, 0, 0]⟩⟩)
  "shape [2] at width 4 needs 8 data bytes, got 4"
#guard failsWith (Tensor.toBytes ⟨.bool, [1], ⟨#[2]⟩⟩) "a bool element is neither 0 nor 1"

end VeriTile.Interp
