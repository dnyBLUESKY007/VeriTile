# Multimem front-end documentation: guarantees and gaps

Research for S6.7 ([#352](https://github.com/dnyBLUESKY007/VeriTile/issues/352)).
Sources accessed **2026-11-12**. This document changes no semantics or assumption
registry. The governing decisions are [PrimitiveSources.md](PrimitiveSources.md),
sections 1, 6 and 9, observed at VeriTile
`a0ba977236322702b1d0d8b891a34a4e1c9b2074`.

**Result:** TileScale documents the intended data movement of its multicast
operations and some call requirements. Triton-distributed documents its multicast
pointer query, but its direct helpers have no usable front-end memory contract
in the sources checked. NVSHMEM describes the pointer's mapping and supported
platforms. None of these passages supplies the missing atomicity, cross-rank
ordering or exact reduction order of the direct helpers. These are separate
findings; partial documentation is not an all-schedule correctness guarantee.
Evidence and the bounded absence checks are listed below.

## Sources and search boundary

Repository sources were retrieved as text at these full pins, outside the
VeriTile checkout. Python was parsed with the standard-library AST to extract
signatures and docstrings, not imported or executed. No external files are
vendored. Source links below identify the exact passages; every source has the
access date above.

| ID | Primary source and portion inspected |
| --- | --- |
| TD-API | Triton-distributed `7908e4ea9010bb2238f9f05b28c904fd4b71c60a`, [language API](https://github.com/ByteDance-Seed/Triton-distributed/blob/7908e4ea9010bb2238f9f05b28c904fd4b71c60a/docs/python-api/triton-dist.language.rst), especially lines 282–284 and 517–519 |
| TD-SEM | Same pin, [front-end semantics](https://github.com/ByteDance-Seed/Triton-distributed/blob/7908e4ea9010bb2238f9f05b28c904fd4b71c60a/docs/python-api/triton-dist.semantics.rst); searched for multicast/multimem and checked the scope/order definitions |
| TD-HELPERS | Same pin, [CUDA helper declarations and docstrings](https://github.com/ByteDance-Seed/Triton-distributed/blob/7908e4ea9010bb2238f9f05b28c904fd4b71c60a/python/triton_dist/language/extra/cuda/language_extra.py#L279-L459); implementation bodies do not supply semantics here |
| TD-BIND | Same pin, [`remote_mc_ptr` binding](https://github.com/ByteDance-Seed/Triton-distributed/blob/7908e4ea9010bb2238f9f05b28c904fd4b71c60a/python/triton_dist/language/extra/cuda/libnvshmem_device.py#L201-L213); inspected only to identify the bound routine |
| TS-API | TileScale `71e71457d40ed6e6a2dad297936bfdbff21d104b`, [distributed API reference](https://github.com/tile-ai/tilescale/blob/71e71457d40ed6e6a2dad297936bfdbff21d104b/docs/distributed_api_reference.md#L242-L292), plus multicast allocation at lines 132–139 |
| TS-DOC | Same pin, [multimem front-end docstrings](https://github.com/tile-ai/tilescale/blob/71e71457d40ed6e6a2dad297936bfdbff21d104b/tilelang/language/distributed/multimem.py#L54-L143); accepted as lower-ranked documentation under PrimitiveSources section 1 |
| NV-PTR | [NVSHMEM Library Setup, Exit, and Query: `nvshmemx_mc_ptr`](https://docs.nvidia.com/nvshmem/api/latest/gen/api/setup.html#nvshmemx-mc-ptr), current unversioned `latest` page |
| NV-USE | [Using NVSHMEM: Communication Model](https://docs.nvidia.com/nvshmem/api/latest/using.html#communication-model), current `latest` page |
| NV-ORDER | [NVSHMEM Memory Ordering](https://docs.nvidia.com/nvshmem/api/latest/gen/api/ordering.html), table and `fence`/`quiet` descriptions |
| NV-MODEL | [NVSHMEM Memory Model: Visibility Guarantees](https://docs.nvidia.com/nvshmem/api/latest/gen/mem-model.html#visibility-guarantees) |
| NV-365 | [NVIDIA's NVSHMEM 3.6.5 release notes](https://github.com/NVIDIA/nvshmem/releases/tag/v3.6.5-0), Limitations |

“Not documented” below means not stated by these checked passages, not a claim
that no document anywhere exists. No instruction-set manual, assembly behavior,
compiler lowering or hardware experiment is used to fill a gap. Links to
lower-level explanations embedded in these pages were not followed.

## Triton-distributed

| Entry point | Observed documentation | What is not documented here |
| --- | --- | --- |
| `libshmem_device.remote_mc_ptr(team, ptr)` | TD-API describes a multicast pointer query. TD-BIND names `nvshmemx_mc_ptr` and preserves the input pointer type. NV-PTR supplies the address-level contract below. | Querying an address does not document a subsequent access's atomicity, memory order, completion or reduction value. |
| `multimem_st_b64`, `multimem_st_b32`, `multimem_st_v2`, `multimem_st_v4`, `multimem_st_p_b32` | Names/signatures exist in TD-HELPERS at lines 279, 284, 306, 338 and 356. Their AST docstrings are absent. TD-API and TD-SEM give no multicast-store contract. | Replica update behavior, atomicity, publication order, completion and masked/vector behavior are not documented by those front-end passages. Names and implementation strings are not guarantees. |
| `multimem_ld_reduce_p_v4` | Declaration at TD-HELPERS line 402; no docstring or contract in TD-API/TD-SEM. | Returned reduction, participant set, predicate behavior, precision and order are not documented. |
| `multimem_ld_reduce_v4` | Declaration at TD-HELPERS line 437. Its docstring delegates the operation to an excluded instruction-level explanation; it names four returned values and an accumulation option. The docstring's `acc_prec` parameter name also differs from the signature's `acc_dtype`. | No independent front-end reduction formula, snapshot, atomicity or memory-order guarantee. The parameter mismatch is a source observation, not a runtime defect claim. |
| `is_nvshmem_multimem_supported()` | TD-API names a capability query. | Its existence is not evidence that the project's node supports multicast or that any helper has stronger semantics. |

**Inference:** the ordinary access vocabulary of TD-SEM cannot by itself assign
atomicity or a scope/order to helpers for which the documentation gives no such
contract. Retain the existing weakest treatments in PrimitiveSources section 5.4;
this report does not reinterpret an internal `_semantic` compiler argument as a
user-facing memory-order parameter.

## TileScale

TS-API documents six public operations. The table separates their stated purpose
from guarantees that cannot be inferred merely from that purpose.

| Entry point | Documented purpose | Missing concurrent guarantee |
| --- | --- | --- |
| `T.multimem_ld_reduce(src, dst, reduce_op=ADD)` | Reduce-load multicast data into a local region (TS-API; TS-DOC lines 54–76). | No reduction tree/rank order, coherent snapshot, atomic load across replicas, or acquire effect is specified. |
| `T.multimem_st(src, dst)` | Broadcast local data into the multicast region (TS-API; TS-DOC lines 79–89). | No atomic broadcast, simultaneous visibility, release effect, or consumer synchronization is specified. |
| `T.multimem_red(src, dst, reduce_op=ADD)` | Reduce into multicast destinations without returning their old contents (TS-API; TS-DOC lines 92–105). | No atomic read-modify-write, inter-operation reduction order, or cross-rank publication guarantee is specified. |
| `T.multimem_tma_store(src, dst, reduce_op=None)` | Shared-memory source to multicast global destination, with an optional reduction (TS-API; TS-DOC lines 108–143). | A caller-managed local sequence is documented; a remote consumer's visibility and synchronization contract is not. |
| `T.multimem_signal(addr, value)` | Store a signal; `uint32` or `uint64` is inferred from the pointer (TS-API). | No atomicity or release effect is specified. |
| `T.multimem_signal_add(addr, value)` | Add to a signal; supported types are `uint32`, `int32`, `uint64` (TS-API). | No atomic fetch/update or ordering with earlier data writes is specified. |

TS-DOC requires matching region ranks, extents and scalar dtypes for direct
operations. Packed half/bfloat regions have additional even-extent, pair-alignment,
full-fragment and layout requirements. The bulk variant requires in-bounds,
contiguous, byte-addressable regions with 16-byte-aligned starts and a positive
16-byte-multiple transfer size representable by an unsigned 32-bit count. These
are documented call constraints, not a proof that any particular caller meets
them. See the cited docstrings for the complete restrictions.

TS-API distinguishes the enum from implemented operation combinations: the enum
contains ADD/MIN/MAX/NONE, while direct load-reduce/reduce-store expose ADD.
The bulk path supports ADD on float32/float16/bfloat16 and MIN/MAX on the two
half types. It requires compatible multicast hardware and a valid allocation;
the bulk path additionally requires CUDA Toolkit 13.1+. No capability was tested.

**Inference:** neither a signal's name nor a local store-arrive/wait sequence
establishes a remote happens-before edge. Keep PrimitiveSources section 5.5's
weakest treatments, including its explicit lack of a reduction order. Its section
9 item 9 already declines new assumptions for the specialized overlap variants.

## NVSHMEM: pointer contract versus operation contract

NV-PTR maps a local symmetric address and a valid team to the corresponding
replicas on that team's PEs. It identifies Hopper-or-later, third-generation
NVSwitch/NVLink4 systems as the supported platform; unsupported platforms return
null. The resulting address is not interchangeable with a symmetric address
argument to another NVSHMEM routine. NV-USE likewise conditions availability on
multicast capability and describes the intended load-reduce/store-broadcast use.
These are address and capability facts, not direct-helper concurrency semantics.

NV-ORDER includes memory stores in its fence/quiet table, but does not give a
front-end-specific contract for the TD or TS helpers above. **Inference:** one
cannot silently identify every helper with an NVSHMEM put, atomic, collective or
signal operation merely because it uses an address obtained from NVSHMEM. A
collective reduction's contract would not specify a direct helper's reduction.

The project's version of record is **3.6.5** (PrimitiveSources section 1).
The accessible API pages are `latest`, not a frozen 3.6.5 manual. A versioned
3.6.5 setup URL and an archive-index URL were unavailable in this retrieval;
no 3.6.5 API-text equivalence was established. NV-365 says synchronization gives
source/destination visibility rather than global visibility, whereas NV-MODEL's
current page has an NVLink-only exception. This report does not resolve that
existing discrepancy or promote newer documentation to a 3.6.5 guarantee.

## Disposition and unchecked work

Some front-end behavior **is documented**, so the ticket's conditional fallback
“if nothing documented is found” does not require inventing an assumption.
No new assumption is proposed or registered. In particular, the missing concurrent
guarantees do not override the user's existing decision against new assumptions
for specialized variants (PrimitiveSources section 9 item 9). If a future proof
needs more, its owning ticket must first identify the exact missing guarantee,
affected kernels and version, then obtain a separate semantic decision.

Not checked: GPU topology or multicast capability; runtime behavior; lowering;
all kernel-to-helper call graphs; numerical reduction agreement; Lean proofs;
or documentation outside the bounded source set. This is a documentation result,
not acceptance of a multimem implementation or any kernel.
