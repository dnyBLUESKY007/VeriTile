# Multimem front-end documentation: guarantees and gaps

Research for S6.7 ([#352](https://github.com/dnyBLUESKY007/VeriTile/issues/352)).
Revised for S6.8 ([#422](https://github.com/dnyBLUESKY007/VeriTile/issues/422)).
Sources rechecked **2026-11-13**. This document changes no semantics or assumption
registry. The governing decisions are [PrimitiveSources.md](PrimitiveSources.md),
sections 1, 6 and 9; this revision uses the same-branch source table.

**Result:** TileScale documents data movement and call requirements; the
Triton-distributed helpers still lack an independent front-end memory contract.
NVSHMEM **3.6.5** documents direct multicast load-reduce and store-broadcast as
access methods subject to its conflicting-access rule (NV-USE, Data Consistency).
That list does not name a reduce-store or provide reduction order or atomicity.
Its pointer contract concerns addresses returned by `nvshmemx_mc_ptr`, the route
bound by Triton-distributed; it does not cover TileScale's own allocator.
Identifying a particular front-end helper with one of these documented access
methods remains an **inference**, not a new documented helper guarantee. No
assumption or existing weakest treatment changes.

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
| NV-PTR | [NVSHMEM Library Setup, Exit, and Query: `nvshmemx_mc_ptr`](https://archive.docs.nvidia.com/nvshmem/api/3.6.5/gen/api/setup.html#nvshmemx-mc-ptr), frozen 3.6.5 page |
| NV-USE | [Using NVSHMEM: Communication Model](https://archive.docs.nvidia.com/nvshmem/api/3.6.5/using.html#communication-model), frozen 3.6.5 page |
| NV-ORDER | [NVSHMEM Memory Ordering](https://archive.docs.nvidia.com/nvshmem/api/3.6.5/gen/api/ordering.html), table and `fence`/`quiet` descriptions |
| NV-MODEL | [NVSHMEM Memory Model: Visibility Guarantees](https://archive.docs.nvidia.com/nvshmem/api/3.6.5/gen/mem-model.html#visibility-guarantees) |
| NV-VERSIONS | [Official version index](https://docs.nvidia.com/nvshmem/api/versions1.json), which links the 3.6.5 archive |
| NV-365 | [NVIDIA's NVSHMEM 3.6.5 release notes](https://github.com/NVIDIA/nvshmem/releases/tag/v3.6.5-0), Limitations |

“Not documented” below means not stated by these checked passages, not a claim
that no document anywhere exists. No instruction-set manual, assembly behavior,
compiler lowering or hardware experiment is used to fill a gap. Links to
lower-level explanations embedded in these pages were not followed.

## Triton-distributed

| Entry point | Observed documentation | What is not documented here |
| --- | --- | --- |
| `libshmem_device.remote_mc_ptr(team, ptr)` | TD-API describes a multicast pointer query. TD-BIND names `nvshmemx_mc_ptr`; preservation of the input pointer type is an observation of binding source, not documentation. NV-PTR supplies the address-level contract below. | Querying an address does not document a subsequent access's atomicity, memory order, completion or reduction value. |
| `multimem_st_b64`, `multimem_st_b32`, `multimem_st_v2`, `multimem_st_v4`, `multimem_st_p_b32` | Names/signatures exist in TD-HELPERS at lines 279, 284, 306, 338 and 356. Their AST docstrings are absent. TD-API and TD-SEM give no multicast-store contract. | Replica update behavior, atomicity, publication order, completion and masked/vector behavior are not documented by those front-end passages. Names and implementation strings are not guarantees. |
| `multimem_ld_reduce_p_v4` | Declaration at TD-HELPERS line 402; no docstring or contract in TD-API/TD-SEM. | Returned reduction, participant set, predicate behavior, precision and order are not documented. |
| `multimem_ld_reduce_v4` | Declaration at TD-HELPERS line 437. Its docstring delegates the operation to an excluded instruction-level explanation; it names four returned values and an accumulation option. The docstring's `acc_prec` parameter name also differs from the signature's `acc_dtype`. | No independent front-end reduction formula, snapshot, atomicity or memory-order guarantee. The parameter mismatch is a source observation, not a runtime defect claim. |
| `is_nvshmem_multimem_supported()` | TD-API names a capability query. | Its existence is not evidence that the project's node supports multicast or that any helper has stronger semantics. |

**Inference:** the ordinary access vocabulary of TD-SEM cannot by itself assign
atomicity or a scope/order to helpers for which the documentation gives no such
contract. Retain the existing weakest treatments in PrimitiveSources section 5.4;
this report does not reinterpret an internal `_semantic` compiler argument as a
user-facing memory-order parameter. Section 9 item 10 keeps undocumented helpers
outside the tutorials subject to a decision when a kernel in scope needs them.

## TileScale

TS-API [lines 132–139](https://github.com/tile-ai/tilescale/blob/71e71457d40ed6e6a2dad297936bfdbff21d104b/docs/distributed_api_reference.md#L132-L139)
requires the distributed VMM allocator and a successful capability probe for
`mcast_size`. Its allocator returns a multicast view and a local physical view;
each rank writes its contribution through the latter. This is TileScale
allocation, not an NVSHMEM symmetric-heap allocation.

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
direct operations and signals require SM90+ and CUDA Toolkit 12.1+, packed
half/bfloat load-reduce requires 12.2+, and the bulk path requires 13.1+
(TS-API lines 261–278). The [README, line 135](https://github.com/tile-ai/tilescale/blob/71e71457d40ed6e6a2dad297936bfdbff21d104b/README.md#L135)
and [release notes, lines 41–43](https://github.com/tile-ai/tilescale/blob/71e71457d40ed6e6a2dad297936bfdbff21d104b/docs/release_v0_0726.md#L41-L43)
class multicast/multimem as experimental and capability-gated, requiring compatible
GPUs, NVSwitch fabric, driver support and accessible IMEX. No capability was tested.

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

### Data Consistency and version comparison

NV-USE, [Data Consistency](https://archive.docs.nvidia.com/nvshmem/api/3.6.5/using.html#data-consistency),
contains the following access-list item and paragraphs (identical in `latest` 3.8.0):

> Direct multimem load reduce and store broadcast operations (for example, using a pointer that was returned by nvshmemx_mc_ptr)

> Two operations that were issued by the same PE or different PEs, and that access the same memory location in parallel, are in conflict when one or more of the operations performs a write. These conflicts result in undefined behavior in the OpenSHMEM memory model. An exception is made when the operations are a combination of AMOs or AMOs and wait/test operations. A second exception is made when the operations are a combination of signal updates and wait/test operations.

> Updates to globally accessible objects are unordered. A PE can enforce the ordering of its updates with respect to accesses performed by the target PE using the nvshmem_fence operation. When updates performed by a PE must be ordered or made visible to PEs other than the target PE, use the nvshmem_quiet operation.

**Inference:** the conflict rule applies to the documented direct access methods,
but neither exception makes an undocumented multicast helper an AMO or a signal
operation. The access list omits reduce-store and does not fix reduction order.
This is a constraint on the NVSHMEM-address route, not a contract for TileScale's
allocator or a license to strengthen the front-end rows.

All four NV API source pages above returned HTTP 200 and
`docsearch:version=3.6.5` on the access date. The official version index points
to this archive; the earlier unavailable-manual conclusion was wrong. Comparing
the same paths under [latest](https://docs.nvidia.com/nvshmem/api/latest/):

| Passage | 3.6.5 versus `latest` (3.8.0) |
| --- | --- |
| NV-PTR, `nvshmemx_mc_ptr` | Same pointer mapping, exclusive multicast use, hardware restriction, null return and prohibition on substituting the multicast address for a symmetric argument |
| NV-USE, Communication Model and Data Consistency | Same multicast capability condition, direct-access list and conflict/ordering passages used here |
| NV-ORDER | Same fence/quiet table and calling-thread fence rule; 3.6.5's quiet description explicitly grants visibility to all PEs on NVLink-only systems and destination visibility on mixed NVLink/InfiniBand systems. Latest instead states destination visibility without that exception and adds local/non-collective and non-notification wording. Latest also has GPU-thread notes and flush routines absent from 3.6.5; see PrimitiveSources E-NV2/E-NV22/E-NV23 |
| NV-MODEL, Visibility Guarantees | Same NVLink-only exception in both versions |
| NV-365, Limitations | The 3.6.5 release notes give only source/destination visibility, without the manual's NVLink-only exception |

The conflict is **within 3.6.5**, not evidence that the exception was added later.
[PrimitiveSources section 10](PrimitiveSources.md#10-still-open) compares the
five manual passages and the release limitation. D2 remains the user's decision;
this document chooses neither reading.

Source-ID mapping to PrimitiveSources: TD-API/TD-SEM keep their names;
TD-HELPERS = TD-LEC, TD-BIND = TD-SHC, TS-API keeps its name, TS-DOC = TS-MM,
NV-PTR = NV-SETUP, NV-USE keeps its name, NV-ORDER = NV-ORD, NV-MODEL = NV-MM,
NV-365 = NV-REL (3.6.5 release).

### Complete bounded front-end sweep

At each full pin above, the case-insensitive pattern
`multimem|multicast|mc_ptr` was searched over `docs/`, `README*`, and
`examples/**/README.md` (equivalent to `grep -rni 'multimem\|multicast\|mc_ptr'`
on that file set). There are **13 matching lines in Triton-distributed** and
**92 in TileScale**. The inventory below accounts for every matched line.
“Name only” means no additional accepted memory guarantee is used: launch names,
capability/test reports, local cluster operations, compiler or instruction-level
material do not supply inter-rank front-end semantics. No linked instruction
manual was followed and no instruction-level claim is cited.

| Pin / path | Matched lines | Disposition |
| --- | --- | --- |
| TD: [docs/kernels/nvidia/allgather.rst](https://github.com/ByteDance-Seed/Triton-distributed/blob/7908e4ea9010bb2238f9f05b28c904fd4b71c60a/docs/kernels/nvidia/allgather.rst#L32) | 32, 55, 57 | Name only; no additional inter-rank memory guarantee used |
| TD: [docs/kernels/nvidia/allreduce.rst](https://github.com/ByteDance-Seed/Triton-distributed/blob/7908e4ea9010bb2238f9f05b28c904fd4b71c60a/docs/kernels/nvidia/allreduce.rst#L15) | 15, 16 | Name only; no additional inter-rank memory guarantee used |
| TD: [docs/kernels/nvidia/index.rst](https://github.com/ByteDance-Seed/Triton-distributed/blob/7908e4ea9010bb2238f9f05b28c904fd4b71c60a/docs/kernels/nvidia/index.rst#L20) | 20, 22 | Name only; no additional inter-rank memory guarantee used |
| TD: [docs/python-api/triton-dist.language.rst](https://github.com/ByteDance-Seed/Triton-distributed/blob/7908e4ea9010bb2238f9f05b28c904fd4b71c60a/docs/python-api/triton-dist.language.rst#L282) | 282, 284, 517, 519 | Cited above for pointer/capability or API requirements |
| TD: [docs/testing.md](https://github.com/ByteDance-Seed/Triton-distributed/blob/7908e4ea9010bb2238f9f05b28c904fd4b71c60a/docs/testing.md#L85) | 85, 226 | Name only; no additional inter-rank memory guarantee used |
| TS: [README.md](https://github.com/tile-ai/tilescale/blob/71e71457d40ed6e6a2dad297936bfdbff21d104b/README.md#L135) | 135 | Cited above for pointer/capability or API requirements |
| TS: [README.md](https://github.com/tile-ai/tilescale/blob/71e71457d40ed6e6a2dad297936bfdbff21d104b/README.md#L28) | 28, 30, 51, 67, 142, 145, 146, 148, 149, 211, 270 | Name only; no additional inter-rank memory guarantee used |
| TS: [docs/distributed_api_reference.md](https://github.com/tile-ai/tilescale/blob/71e71457d40ed6e6a2dad297936bfdbff21d104b/docs/distributed_api_reference.md#L132) | 132, 135, 137, 242, 244, 245, 247, 251, 252, 253, 254, 255, 256, 259, 261, 264, 269, 290 | Cited above for pointer/capability or API requirements |
| TS: [docs/distributed_api_reference.md](https://github.com/tile-ai/tilescale/blob/71e71457d40ed6e6a2dad297936bfdbff21d104b/docs/distributed_api_reference.md#L21) | 21, 23, 60, 142, 341, 348, 384 | Name only; no additional inter-rank memory guarantee used |
| TS: [docs/index.md](https://github.com/tile-ai/tilescale/blob/71e71457d40ed6e6a2dad297936bfdbff21d104b/docs/index.md#L7) | 7 | Name only; no additional inter-rank memory guarantee used |
| TS: [docs/programming_guides/cluster_tma.md](https://github.com/tile-ai/tilescale/blob/71e71457d40ed6e6a2dad297936bfdbff21d104b/docs/programming_guides/cluster_tma.md#L1) | 1, 4, 29, 34, 40, 53, 54, 64, 70, 80, 94, 102, 107, 296, 312, 350 | Name only; no additional inter-rank memory guarantee used |
| TS: [docs/release_v0_0726.md](https://github.com/tile-ai/tilescale/blob/71e71457d40ed6e6a2dad297936bfdbff21d104b/docs/release_v0_0726.md#L41) | 41 | Cited above for pointer/capability or API requirements |
| TS: [docs/release_v0_0726.md](https://github.com/tile-ai/tilescale/blob/71e71457d40ed6e6a2dad297936bfdbff21d104b/docs/release_v0_0726.md#L99) | 99, 101, 103, 104, 106, 107, 115, 127, 128, 130, 133, 136, 137, 138, 148, 149, 153, 156, 157 | Name only; no additional inter-rank memory guarantee used |
| TS: [examples/distributed/allreduce/README.md](https://github.com/tile-ai/tilescale/blob/71e71457d40ed6e6a2dad297936bfdbff21d104b/examples/distributed/allreduce/README.md#L1) | 1, 3, 4, 10, 13, 16, 17, 22, 28, 29, 31, 32, 37, 39, 46, 50, 53, 62 | Name only; no additional inter-rank memory guarantee used |

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
