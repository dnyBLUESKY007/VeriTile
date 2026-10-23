// VeriTile phase-1 litmus kernels (sm_90, compiled at run time by NVRTC through CuPy).
//
// One protocol, many variants. Every test is the Triton-distributed tutorial 01 slot
// protocol between one PRODUCER block and one CONSUMER block, repeated ROUNDS times:
//
//   flag (u64, one per instance)   2k   = "slot empty, round k may be written"
//                                  2k+1 = "slot full, round k may be read"
//   payload (1 KiB = 8 lines of 128 B) holds the value k+1 in every 32-bit word in round k.
//
//   PRODUCER, round k                         CONSUMER, round k
//   wait flag >= 2k      (P_POLL)             wait flag >= 2k+1   (C_POLL)   <- "full" edge
//   store payload = k+1  (weak st)            load payload        (C_LD)
//   bar.sync                                  bar.sync
//   P_FENCE ; flag = 2k+1 (P_FLAG_ST)         C_FENCE ; flag = 2k+2 (C_FLAG_ST) -> "empty" edge
//
// The consumer checks every payload word it read in round k:
//   == k+1  -> ok
//   <  k+1  -> STALE  (the "full" edge failed: message-passing violation)
//   >  k+1  -> FUTURE (the "empty" edge failed: the producer overwrote the slot before the
//                      consumer's earlier loads were satisfied: load-buffering shape)
//
// litmus.py picks each knob with -D<KNOB>=<n>; the exact PTX string of every choice is
// written next to it below and also recorded in each results/<test>.json.
// Placement (which GPU runs which block, which GPU's memory holds flag+payload) is chosen
// by the host through the pointer tables; the kernel code is the same for every placement.

typedef unsigned int u32;
typedef unsigned long long u64;

#ifndef P_POLL
#define P_POLL 0      // producer's wait for "empty" (see poll_load)
#endif
#ifndef P_FENCE
#define P_FENCE 0     // producer's fence before the "full" flag store (see fence_op)
#endif
#ifndef P_FLAG_ST
#define P_FLAG_ST 0   // producer's "full" flag store (see flag_store)
#endif
#ifndef C_POLL
#define C_POLL 0      // consumer's wait for "full"
#endif
#ifndef C_LD
#define C_LD 0        // consumer's payload load (see payload_load)
#endif
#ifndef C_FENCE
#define C_FENCE 0     // consumer's fence before the "empty" flag store
#endif
#ifndef C_FLAG_ST
#define C_FLAG_ST 0   // consumer's "empty" flag store
#endif
#ifndef C_WARM
#define C_WARM 0      // 1: consumer loads the payload once BEFORE waiting, to put old lines in L1
#endif
#ifndef C_ORDER
#define C_ORDER 0     // 0 normal; 1 SELF-TEST read before wait; 2 SELF-TEST notify before read
#endif

#define LANES 32
#define WORDS_PER_LANE 8              // 32 lanes * 8 words * 4 B = 1 KiB payload
#define SELFTEST_DELAY_NS 20000u      // C_ORDER==2: give the producer time to overwrite

// out[] layout per (instance, role): OUT_STRIDE x u64
#define OUT_STRIDE 16
enum { O_ROUNDS = 0, O_OK, O_STALE, O_FUTURE, O_MIXED, O_TIMEOUT, O_FIRST_ROUND, O_FIRST_VAL,
       O_SINK };

// ---------------------------------------------------------------------------------------
// Flag polling loads. "acquire" makes later loads of this thread wait for this load and
// (if morally strong with the writer's release) see what the writer published.
// ---------------------------------------------------------------------------------------
template <int V>
__device__ __forceinline__ u64 poll_load(const u64* p) {
  u64 v;
  if (V == 0) {
    // Tutorial 01 dl.wait(..., "gpu", "acquire"): acquire at DEVICE scope. Its scope does
    // not include threads on the other GPU, so PTX gives no ordering across GPUs.
    asm volatile("ld.global.acquire.gpu.b64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  } else if (V == 1) {
    // Control: acquire at SYSTEM scope. Morally strong with a .sys release on any GPU.
    asm volatile("ld.global.acquire.sys.b64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  } else if (V == 2) {
    // NVSHMEM wait_until as implemented (v3.8.0): a volatile poll; the fence comes after
    // the loop (poll_fence). volatile == relaxed at system scope in PTX.
    asm volatile("ld.volatile.global.b64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  } else if (V == 3) {
    // Cross-GPU dossier variant (ii): relaxed system-scope poll; fence after the loop.
    asm volatile("ld.relaxed.sys.global.b64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  } else if (V == 4) {
    // NCCL Simple protocol: volatile poll, NO fence; payload loads must then be volatile.
    asm volatile("ld.volatile.global.b64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  } else {
    // V == 5, POSITIVE CONTROL: relaxed device-scope poll with no acquire and no fence.
    // Nothing orders the later weak payload loads after it.
    asm volatile("ld.relaxed.gpu.global.b64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  }
  return v;
}

// Fence issued once after the poll loop has seen the target value.
template <int V>
__device__ __forceinline__ void poll_fence() {
  if (V == 2) {
    // __threadfence() as NVSHMEM's nvshmemi_transfer_syncapi_update_mem uses it:
    // membar.gl == fence.sc.gpu on sm_70+ (a DEVICE-scope fence).
    asm volatile("membar.gl;" ::: "memory");
  } else if (V == 3) {
    // Device-scope acquire-release fence after a relaxed .sys read: an acquire pattern
    // (relaxed read, then fence), but only at .gpu scope.
    asm volatile("fence.acq_rel.gpu;" ::: "memory");
  }
}

// Spin until *p >= target or max_spin polls; returns false on timeout.
template <int V>
__device__ __forceinline__ bool wait_ge(const u64* p, u64 target, u64 max_spin) {
  for (u64 i = 0; i < max_spin; ++i) {
    if (poll_load<V>(p) >= target) {
      poll_fence<V>();
      return true;
    }
  }
  return false;
}

// ---------------------------------------------------------------------------------------
// Writer-side fences. "fence + relaxed store" is PTX's release pattern (PTX ISA §8.8).
// ---------------------------------------------------------------------------------------
template <int V>
__device__ __forceinline__ void fence_op() {
  if (V == 0) {
    // Tutorial 01 notify(comm_scope="intra_node"): membar.sys == fence.sc.sys on sm_70+.
    asm volatile("membar.sys;" ::: "memory");
  } else if (V == 1) {
    // NCCL postPeer: fence_acq_rel_sys().
    asm volatile("fence.acq_rel.sys;" ::: "memory");
  } else if (V == 2) {
    // Writer-weak probe: membar.gl == fence.sc.gpu, a DEVICE-scope fence only.
    asm volatile("membar.gl;" ::: "memory");
  } else if (V == 4) {
    // Dossier test 3 writer: device-scope acquire-release fence, a peer-local .gpu release.
    asm volatile("fence.acq_rel.gpu;" ::: "memory");
  }
  // V == 3, POSITIVE CONTROL: no fence at all; the flag store may overtake the payload.
}

template <int V>
__device__ __forceinline__ void flag_store(u64* p, u64 v) {
  if (V == 0) {
    // Tutorial 01: relaxed store at SYSTEM scope.
    asm volatile("st.relaxed.sys.global.b64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
  } else if (V == 1) {
    // Relaxed store at DEVICE scope (dossier test 3, a peer-local .gpu writer).
    asm volatile("st.relaxed.gpu.global.b64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
  } else {
    // V == 2: volatile store, as NVSHMEM's P2P signal_op(SET) does.
    asm volatile("st.volatile.global.b64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
  }
}

// ---------------------------------------------------------------------------------------
// Payload accesses: 128-bit vectors, two per lane (8 words).
// ---------------------------------------------------------------------------------------
__device__ __forceinline__ void payload_store(u32* p, u32 v) {
  // Weak (plain) vector store, like tl.store; it may be cached and reordered freely
  // until a release orders it.
  asm volatile("st.global.v4.b32 [%0], {%1, %1, %1, %1};" :: "l"(p), "r"(v) : "memory");
}

template <int V>
__device__ __forceinline__ void payload_load(const u32* p, u32& a, u32& b, u32& c, u32& d) {
  if (V == 0) {
    // Weak vector load, like tl.load. On sm_90 it may be served from the SM's L1, which is
    // not kept coherent with other SMs or GPUs; an acquire before it is what makes it fresh.
    asm volatile("ld.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(p) : "memory");
  } else if (V == 1) {
    // NCCL style: volatile (relaxed .sys) payload loads, never served from a stale L1 line.
    asm volatile("ld.volatile.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(p) : "memory");
  } else if (V == 2) {
    // DeepEP V1 ld_nc_global(int4) verbatim: read-only (non-coherent) path, do not allocate
    // in L1, prefetch 256 B into L2. PTX says .nc data must not change during the kernel;
    // DeepEP uses it on data a peer writes and reports it "tested to be guaranteed" on Hopper.
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.s32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(p) : "memory");
  } else {
    // V == 3, POSITIVE CONTROL for V == 2: the same read-only path WITHOUT L1::no_allocate.
    asm volatile("ld.global.nc.v4.s32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "l"(p) : "memory");
  }
}

__device__ __forceinline__ void block_sync() {
  // bar.sync 0 == __syncthreads(): all 32 threads arrive; orders memory among them.
  asm volatile("bar.sync 0;" ::: "memory");
}
__device__ __forceinline__ void warp_sync() {
  // bar.warp.sync: tutorial 01 runs it after lane 0's wait so the other lanes follow it.
  asm volatile("bar.warp.sync 0xffffffff;" ::: "memory");
}

__device__ __forceinline__ u32 xorshift(u32& s) {
  s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s;
}
__device__ __forceinline__ void jitter(u32& s, u32 jitter_ns) {
  if (jitter_ns) __nanosleep(xorshift(s) % jitter_ns);
}

__device__ void producer(u64* flag, u32* payload, u64* out, u32 rounds, u64 max_spin,
                         u32 jitter_ns, u32 seed) {
  __shared__ volatile int abort_;
  const u32 lane = threadIdx.x;
  u32 rng = seed | 1u;
  u64 done = 0;
  if (lane == 0) abort_ = 0;
  for (u32 k = 0; k < rounds; ++k) {
    jitter(rng, jitter_ns);
    if (lane == 0 && !wait_ge<P_POLL>(flag, 2ull * k, max_spin)) abort_ = 1;   // "empty"?
    warp_sync();
    if (abort_) break;
    u32* w = payload + lane * WORDS_PER_LANE;
    payload_store(w, k + 1);
    payload_store(w + 4, k + 1);
    block_sync();                       // tutorial: __syncthreads() before the fence
    if (lane == 0) {
      fence_op<P_FENCE>();              // release pattern, part 1
      flag_store<P_FLAG_ST>(flag, 2ull * k + 1);   // part 2: "full"
    }
    ++done;
  }
  if (lane == 0) {
    out[O_ROUNDS] = done;
    out[O_TIMEOUT] = abort_ ? 1 : 0;
  }
}

__device__ void consumer(u64* flag, const u32* payload, u64* out, u32 rounds, u64 max_spin,
                         u32 jitter_ns, u32 seed) {
  __shared__ volatile int abort_;
  const u32 lane = threadIdx.x;
  const u32* w = payload + lane * WORDS_PER_LANE;
  u32 rng = seed | 1u;
  u64 cnt[4] = {0, 0, 0, 0};            // ok, stale, future, mixed
  u64 first_round = ~0ull, first_val = 0, done = 0;
  u32 a, b, c, d, e, f, g, h;
  u32 sink = 0;   // ptxas deletes a load whose value is never used, so warm-up values land here
  if (lane == 0) abort_ = 0;
  for (u32 k = 0; k < rounds; ++k) {
    if (C_WARM) {                                  // pull the previous round's lines into L1
      payload_load<0>(w, a, b, c, d);
      payload_load<0>(w + 4, e, f, g, h);
      sink ^= a ^ b ^ c ^ d ^ e ^ f ^ g ^ h;
    }
    if (C_ORDER == 1) {                            // SELF-TEST: read before waiting
      payload_load<C_LD>(w, a, b, c, d);
      payload_load<C_LD>(w + 4, e, f, g, h);
    }
    jitter(rng, jitter_ns);
    if (lane == 0 && !wait_ge<C_POLL>(flag, 2ull * k + 1, max_spin)) abort_ = 1;  // "full"?
    warp_sync();
    if (abort_) break;
    if (C_ORDER == 2) {                            // SELF-TEST: say "empty" before reading
      block_sync();
      if (lane == 0) { fence_op<C_FENCE>(); flag_store<C_FLAG_ST>(flag, 2ull * k + 2); }
      __nanosleep(SELFTEST_DELAY_NS);
    }
    if (C_ORDER != 1) {
      payload_load<C_LD>(w, a, b, c, d);
      payload_load<C_LD>(w + 4, e, f, g, h);
    }
    u32 lo = min(min(min(a, b), min(c, d)), min(min(e, f), min(g, h)));
    u32 hi = max(max(max(a, b), max(c, d)), max(max(e, f), max(g, h)));
    lo = __reduce_min_sync(0xffffffffu, lo);
    hi = __reduce_max_sync(0xffffffffu, hi);
    const u32 want = k + 1;
    int kind = (lo == want && hi == want) ? 0 : (hi <= want ? 1 : (lo >= want ? 2 : 3));
    cnt[kind]++;
    if (kind && first_round == ~0ull) { first_round = k; first_val = ((u64)lo << 32) | hi; }
    if (C_ORDER != 2) {
      block_sync();                    // all lanes finished reading round k
      if (lane == 0) {
        fence_op<C_FENCE>();
        flag_store<C_FLAG_ST>(flag, 2ull * k + 2);  // "empty": round k+1 may be written
      }
    }
    ++done;
  }
  if (lane == 0) {
    out[O_ROUNDS] = done;
    out[O_OK] = cnt[0]; out[O_STALE] = cnt[1]; out[O_FUTURE] = cnt[2]; out[O_MIXED] = cnt[3];
    out[O_TIMEOUT] = abort_ ? 1 : 0;
    out[O_FIRST_ROUND] = first_round; out[O_FIRST_VAL] = first_val;
  }
  sink = __reduce_xor_sync(0xffffffffu, sink);
  if (lane == 0) out[O_SINK] = sink;
}

// Stress block: volatile read-modify-write traffic over a large buffer (local L2 or, when the
// host passes a peer buffer, NVLink) until every litmus block on this GPU has finished.
__device__ void stress(volatile u32* scratch, u64 words, const volatile u32* done_count,
                       u32 n_litmus_blocks, u64 cap, u32 seed) {
  __shared__ volatile int stop;
  u32 rng = (seed ^ (blockIdx.x * 2654435761u) ^ threadIdx.x) | 1u;
  if (threadIdx.x == 0) stop = 0;
  warp_sync();
  for (u64 i = 0; i < cap && !stop; ++i) {
    u64 j = xorshift(rng) % words;
    scratch[j] = scratch[(j * 7 + 1) % words] + 1;
    if ((i & 255) == 0 && threadIdx.x == 0 && *done_count >= n_litmus_blocks) stop = 1;
    warp_sync();
  }
}

// One launch per GPU. Block b plays roles[b] (0 producer, 1 consumer, 2 stress) for
// instance insts[b]. flag_ptr/payload_ptr hold device addresses that may be on the peer GPU.
extern "C" __global__ void litmus(const int* roles, const int* insts, const u64* flag_ptr,
                                  const u64* payload_ptr, u64* out, u32* done_count,
                                  u32 n_litmus_blocks, u64 scratch_ptr, u64 scratch_words,
                                  u32 rounds, u64 max_spin, u32 jitter_ns, u32 seed,
                                  u64 stress_cap) {
  const int role = roles[blockIdx.x];
  const int inst = insts[blockIdx.x];
  const u32 s = seed ^ (u32)(inst * 0x9E3779B9u) ^ (u32)role;
  if (role == 2) {
    stress((volatile u32*)scratch_ptr, scratch_words, done_count, n_litmus_blocks,
           stress_cap, s);
    return;
  }
  u64* flag = (u64*)flag_ptr[inst];
  u32* payload = (u32*)payload_ptr[inst];
  u64* o = out + (u64)(inst * 2 + role) * OUT_STRIDE;
  if (role == 0) producer(flag, payload, o, rounds, max_spin, jitter_ns, s);
  else consumer(flag, payload, o, rounds, max_spin, jitter_ns, s);
  block_sync();
  if (threadIdx.x == 0) atomicAdd(done_count, 1u);
}
