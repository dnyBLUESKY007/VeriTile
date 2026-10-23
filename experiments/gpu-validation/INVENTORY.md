# Inventory of GPU-dependent tests for VeriTile async/distributed work

Prepared 2026-10-23 for fork ticket #23 (hardware validation plan). Tickets: #23 hardware
validation plan; #24 cross-GPU visibility research (closed; its litmus list is the source of
rows 2-12); #25 trust-base representation (each `trust-base` row is a candidate opaque lemma
and its evidence record); #13 closed (per-platform axioms deferred; superseded by the sm_90
trust base). Lemma names are proposals for #25, not decisions.

Hardware: **1** = one H800; **2** = two H800 on one node over NVLink; **MN** = multi-node.
Status: **ready** = in this package; **later** = needs work not yet done; **blocked** = needs
software or access not yet available, or is out of first-version scope.

| # | Test (package name) | Purpose | Serves | HW | Software | Status |
|---|---|---|---|---|---|---|
| 1 | `selftest_read_before_wait`, `selftest_notify_before_read` | Harness can count stale and future outcomes | validity of every row below; #23 | 2 | this package (CuPy 14.2 + NVRTC) | ready |
| 2 | `full_tut01`, `full_tut01_warm` | MP "full" edge: `membar.sys; st.relaxed.sys` writer, `ld.global.acquire.gpu` reader, consumer-homed data | `sm90_nvlink_home_acquire_gpu` (A1/H1'); #23, #25 | 2 | this package | ready |
| 3 | `full_sys` | Same edge with `ld.global.acquire.sys`: PTX-sanctioned control | A0/P1 baseline; #23 | 2 | this package | ready |
| 4 | `full_nvshmem_wait`, `full_relaxed_sys_fence_gpu` | Reader = `.sys` poll + `.gpu` fence (NVSHMEM `wait_until` as implemented; dossier variant ii) | `sm90_nvlink_home_fence_gpu` (H1); #25 | 2 | this package | ready |
| 5 | `full_nccl_volatile` | NCCL Simple: volatile poll + volatile payload, no reader fence | `sm90_volatile_flag_sync`; #25 | 2 | this package | ready |
| 6 | `full_writer_gpu_fence` | Is the writer's `.sys` fence load-bearing? (`membar.gl` writer) | probe; #23 | 2 | this package | ready |
| 7 | `full_pos_relaxed`, `full_pos_relaxed_warm`, `full_pos_volatile_weakld`, `full_pos_no_writer_fence` | Positive controls: one knob weaker (no acquire / weak payload loads / no writer fence) | detectors for rows 2-6 | 2 | this package | ready |
| 8 | `same_tut01`, `same_pos_relaxed` | Same-GPU control: `.gpu` acquire is sanctioned on one device | baseline; #23 | 1 | this package | ready |
| 9 | `e2_remote_poll_sys_writer`, `e2_remote_poll_gpu_writer`, `e2_remote_poll_sys`, `e2_pos_relaxed` | Remote `ld.global.acquire.gpu` poll of a peer-homed flag, remote payload reads; timeouts = liveness (dossier tests 3-4) | `sm90_nvlink_remote_poll_gpu` (A2/H3); #25 | 2 | this package | ready |
| 10 | `empty_tut01`, `empty_sys`, `empty_pos_relaxed` | "Empty" edge, load-buffering shape: consumer reads, `.sys` release on local flag; producer polls remotely at `.gpu`, overwrites (dossier test 5) | `sm90_nvlink_remote_poll_empty_edge`, DeepEP `relaxed-head-WAR`; #25 | 2 | this package | ready (control may never fire) |
| 11 | `tut01_ring` | Tutorial 01 scopes on both edges, both directions at once | both tutorial lemmas; #25 illustration | 2 | this package | ready |
| 12 | `nc_deepep`, `nc_deepep_warm`, `nc_pos_plain`, `nc_pos_plain_warm` | DeepEP V1 `ld.global.nc.L1::no_allocate.L2::256B.v4.s32` reads of peer-written data after `ld.acquire.sys`; plain `.nc` as control (dossier variant v) | `sm90_nc_noalloc_coherent`; #25 | 2 | this package | ready |
| 13 | rows 2-12 with `--stress nvlink --jitter-ns 2000` | Same tests under NVLink traffic and random delays | strengthens rows 2-12 | 2 | this package | ready |
| 14 | Cross-process / distinct-VA variant of rows 2-12 | Producer and consumer in two processes (CUDA IPC or VMM/NVSHMEM mapping): PTX alias/proxy rule and cross-process `.sys` | H2 (alias); needed before rows 2-12 cover tutorial 01's real setting; #25 | 2 | CuPy IPC or cuda-python VMM; launcher | later (phase 1b) |
| 15 | Memory-sync-domain variant (dossier test 6) | Producer launched in `cudaLaunchMemSyncDomainRemote` | domain premise of A1; #25 | 2 | `cudaLaunchKernelEx` attributes (cuda-python `cuda.core` or C++) | later |
| 16 | Cache-path probe (dossier test 7) | `ld.ca` vs `ld.cg` timing; Nsight Compute `l1tex`, `lts` `aperture_peer`, NVLink sections | explains A1/A2 mechanism | 2 | Nsight Compute; profiling permission | later |
| 17 | SASS inspection of phase-1 kernels | Confirm cache modes (e.g. `LDG.E.NA...CONSTANT`) and that ptxas kept each access | validity of rows 2-12 | none | `cuobjdump`/`nvdisasm` (CUDA Toolkit) | later |
| 18 | NVSHMEM `fence` vs `putmem_nbi` ordering | nbi put payload, `nvshmem_fence`, signal; reader `signal_wait_until`; controls: blocking put + fence (documented), `quiet` (documented), no fence (positive) | N2' (docs conflict, G7); NVSHMEM-based targets | 2 | NVSHMEM 3.x (`nvidia-nvshmem-cu12` 3.8.0 + `nvshmem4py-cu12` 0.4.1, or source) + bootstrap | blocked (NVSHMEM not installed). Over NVLink P2P, nbi and blocking puts share one ld/st path in NVSHMEM 3.8.0, so a pass cannot settle the documentation conflict |
| 19 | NVSHMEM API reader vs raw poll on the symmetric heap | `signal_wait_until` (N5) vs plain `ld.acquire.gpu` poll, with `nvshmem_ptr` peer mappings | N5; H1/H1' + H2 in the tutorial's real memory | 2 | NVSHMEM | blocked |
| 20 | Triton-distributed tutorial 01 (and 05, 07, 08) at `7908e4ea` | Run as written, with `"sys"` waits, more repeats, under stress | end-to-end check of rows 2, 10, 11 | 2 | Triton-distributed build (its Triton + NVSHMEM), PyTorch | later / blocked on build |
| 21 | TileScale DeepEP intranode port (`@6a7fe38`, `T.ld(nc=True)`) and DeepEP V1 intranode with/without `DISABLE_AGGRESSIVE_PTX_INSTRS` | Production-shaped `.nc` use | `sm90_nc_noalloc_coherent`; target 4 | 2 | TileScale pre-release / DeepEP v1.2.1 builds | later |
| 22 | Differential tests of VeriTile's sequential semantics vs real Triton | Run TritonBench-G kernels on GPU (or `TRITON_INTERPRET=1` subset on CPU) and compare with the Lean interpreter | sequential semantics adequacy; #23 | 1 | Triton, PyTorch, Lean build, a comparison harness (none exists) | later |
| 23 | mbarrier / TMA single-GPU litmus | Phase/parity waits, `expect_tx`, `cp.async.bulk` + `fence.proxy.async`, TMA store wait | async-proxy model (decision pending) | 1 | this harness (NVRTC sm_90/sm_90a) | later |
| 24 | TileLang `T.ws` / TMA examples | Run warp-specialized TMA examples | async user-code semantics | 1 | TileLang 0.1.15 | later |
| 25 | FlashInfer `dcp_lse_reduce`, vLLM/FlashInfer custom all-reduce volatile barriers | Production distributed kernels from documented primitives | targets 5+ | 2 | FlashInfer, NCCL | later |
| 26 | NVLink 128-byte store granularity (LL128 / Lamport) | Single-copy atomicity beyond 8 bytes | `nvlink-128b-store-granularity`, A3 | 2 | this harness | later (may not be needed) |
| 27 | Forward progress / co-residency of spin waits | Hang detection under oversubscription | launch assumption (§7.7 item 4) | 1-2 | this harness | later (tests can show hangs, not their absence) |
| 28 | Multi-node NVLink+IB / IBGDA visibility (N6 split) | Visibility of NVSHMEM updates across nodes | N6 platform split | MN | NVSHMEM + IB | blocked (out of first-version scope; one node only) |
