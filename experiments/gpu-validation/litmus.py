"""Phase-1 cross-GPU / same-GPU PTX litmus tests for the VeriTile sm_90 trust base.

  uv run --no-sync python litmus.py list                 # tests, kinds, what each serves
  uv run --no-sync python litmus.py compile-only         # NVRTC sm_90 compile + PTX check, no GPU
  uv run --no-sync python litmus.py run all [--quick]    # run tests, write results/<test>.json
  uv run --no-sync python litmus.py run full_tut01 --rounds 1000000
  uv run --no-sync python litmus.py run all --stress nvlink --jitter-ns 2000 --tag stress
  uv run --no-sync python litmus.py summarize            # verdict table, results/summary.json

Protocol and every PTX line: kernels.cu. Test meaning and pass/fail rules: README.md.
"""

import argparse
import hashlib
import json
import pathlib
import sys
import time
import traceback

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results"
SRC = (HERE / "kernels.cu").read_text()

# Exact PTX of every knob value, as written in kernels.cu (checked by compile-only).
PTX = {
    "P_POLL": {0: ["ld.global.acquire.gpu.b64"], 1: ["ld.global.acquire.sys.b64"],
               5: ["ld.relaxed.gpu.global.b64"]},
    "C_POLL": {0: ["ld.global.acquire.gpu.b64"], 1: ["ld.global.acquire.sys.b64"],
               2: ["ld.volatile.global.b64", "membar.gl"],
               3: ["ld.relaxed.sys.global.b64", "fence.acq_rel.gpu"],
               4: ["ld.volatile.global.b64"], 5: ["ld.relaxed.gpu.global.b64"]},
    "P_FENCE": {0: ["membar.sys"], 1: ["fence.acq_rel.sys"], 2: ["membar.gl"], 3: [],
                4: ["fence.acq_rel.gpu"]},
    "C_FENCE": {0: ["membar.sys"], 1: ["fence.acq_rel.sys"], 2: ["membar.gl"], 3: [],
                4: ["fence.acq_rel.gpu"]},
    "P_FLAG_ST": {0: ["st.relaxed.sys.global.b64"], 1: ["st.relaxed.gpu.global.b64"],
                  2: ["st.volatile.global.b64"]},
    "C_FLAG_ST": {0: ["st.relaxed.sys.global.b64"], 1: ["st.relaxed.gpu.global.b64"],
                  2: ["st.volatile.global.b64"]},
    "C_LD": {0: ["ld.global.v4.b32"], 1: ["ld.volatile.global.v4.b32"],
             2: ["ld.global.nc.L1::no_allocate.L2::256B.v4.s32"], 3: ["ld.global.nc.v4.s32"]},
}
ALWAYS_PTX = ["st.global.v4.b32", "bar.sync 0", "bar.warp.sync"]

# Default knobs: every edge NOT under test uses the PTX-sanctioned .sys/.sys pairing.
BASE = dict(P_POLL=1, P_FENCE=0, P_FLAG_ST=0, C_POLL=1, C_LD=0, C_FENCE=0, C_FLAG_ST=0,
            C_WARM=0, C_ORDER=0)

# kind: sanctioned      PTX promises the forbidden outcome never occurs (harness/hardware check)
#       trust-base      candidate opaque lemma; zero violations + fired controls = evidence
#       probe           measurement; neither outcome admits anything by itself
#       positive-control  one knob weaker than its target; should show violations
#       self-test       deliberately broken program order; MUST show violations
# placement: cross = producer GPU0, consumer GPU1; e2 = producer GPU1, consumer GPU0;
#            same = both on GPU0; ring = half the instances each way (tutorial 01's ring).
# home: GPU whose memory holds flag+payload: the consumer's (tutorial) or the producer's.
T = {}


def t(name, kind, serves, why, placement="cross", home="consumer", controls=(), **knobs):
    T[name] = dict(kind=kind, serves=serves, why=why, placement=placement, home=home,
                   controls=list(controls), knobs={**BASE, **knobs})


L_FULL = "lemma sm90_nvlink_home_acquire_gpu (A1/H1', 'full' edge); #23, #25"
L_EMPTY = "lemma sm90_nvlink_remote_poll_empty_edge (H3 on the WAR edge); #23, #25"
L_E2 = "lemma sm90_nvlink_remote_poll_gpu (A2/H3); #23, #25"

# --- "full" edge: message passing, forbidden = consumer sees full but stale payload ---
t("full_tut01", "trust-base", L_FULL, "Tutorial 01 full edge: membar.sys; st.relaxed.sys "
  "writer, ld.global.acquire.gpu reader, weak payload loads", controls=["full_pos_relaxed"],
  C_POLL=0)
t("full_tut01_warm", "trust-base", L_FULL, "As full_tut01, reader first loads the payload "
  "weakly so old lines sit in L1 (dossier variant i)", controls=["full_pos_relaxed_warm"],
  C_POLL=0, C_WARM=1)
t("full_sys", "sanctioned", "A0/P1 baseline (PTX 8.9.4, CUDA P2P 3.4.2.3); harness check",
  "Control: .sys release, ld.global.acquire.sys reader (dossier variant iv)",
  controls=["full_pos_relaxed"], C_POLL=1)
t("full_nvshmem_wait", "trust-base", "lemma sm90_nvlink_home_fence_gpu (H1, NVSHMEM "
  "wait_until as implemented); #23, #25", "Reader: ld.volatile poll, then membar.gl",
  controls=["full_pos_relaxed"], C_POLL=2)
t("full_relaxed_sys_fence_gpu", "trust-base", "lemma sm90_nvlink_home_fence_gpu (H1, "
  "dossier variant ii); #23, #25", "Reader: ld.relaxed.sys poll, then fence.acq_rel.gpu",
  controls=["full_pos_relaxed"], C_POLL=3)
t("full_nccl_volatile", "trust-base", "lemma sm90_volatile_flag_sync (NCCL Simple, "
  "dossier variant iii); #23, #25", "Writer fence.acq_rel.sys; reader volatile poll and "
  "volatile payload loads, no fence", controls=["full_pos_volatile_weakld"],
  P_FENCE=1, C_POLL=4, C_LD=1)
t("full_writer_gpu_fence", "probe", "Is the writer's .sys fence load-bearing? (dossier "
  "test 2); #23", "Writer membar.gl instead of membar.sys; reader acquire.sys",
  controls=["full_pos_no_writer_fence"], P_FENCE=2)
t("full_pos_relaxed", "positive-control", "detector for full_* tests", "Reader "
  "ld.relaxed.gpu poll with no acquire (one knob weaker than full_tut01)", C_POLL=5)
t("full_pos_relaxed_warm", "positive-control", "detector for full_tut01_warm",
  "As full_pos_relaxed with the L1 warm-up load", C_POLL=5, C_WARM=1)
t("full_pos_volatile_weakld", "positive-control", "detector for full_nccl_volatile",
  "NCCL-style volatile poll but WEAK payload loads", P_FENCE=1, C_POLL=4, C_LD=0)
t("full_pos_no_writer_fence", "positive-control", "detector for full_writer_gpu_fence",
  "Writer drops the fence entirely; reader acquire.sys", P_FENCE=3)
t("selftest_read_before_wait", "self-test", "harness: STALE counting works",
  "Consumer reads the payload BEFORE waiting for full", C_ORDER=1)

# --- same-GPU controls (PTX: .gpu acquire is morally strong within one GPU) ---
t("same_tut01", "sanctioned", "same-GPU baseline: .gpu scope suffices on one device",
  "Tutorial 01 full-edge PTX with producer and consumer on GPU 0",
  placement="same", controls=["same_pos_relaxed"], C_POLL=0, P_POLL=0)
t("same_pos_relaxed", "positive-control", "detector for same_tut01",
  "Same GPU, reader ld.relaxed.gpu with no acquire", placement="same", C_POLL=5, P_POLL=0)

# --- remote device-scope poll of a peer-homed flag (E2) ---
t("e2_remote_poll_sys_writer", "trust-base", L_E2, "Data+flag on GPU1, written locally "
  "with membar.sys; st.relaxed.sys; GPU0 polls remotely with ld.global.acquire.gpu and reads "
  "remotely (dossier test 4)", placement="e2", home="producer",
  controls=["e2_pos_relaxed"], C_POLL=0)
t("e2_remote_poll_gpu_writer", "probe", "A2 with a .gpu writer (dossier test 3; dossier "
  "advises not adopting A2); #23", "As e2_remote_poll_sys_writer but writer "
  "fence.acq_rel.gpu; st.relaxed.gpu", placement="e2", home="producer",
  controls=["e2_pos_relaxed"], C_POLL=0, P_FENCE=4, P_FLAG_ST=1)
t("e2_remote_poll_sys", "sanctioned", "A0/P1 baseline for remote reads",
  "Remote poll with ld.global.acquire.sys", placement="e2", home="producer",
  controls=["e2_pos_relaxed"], C_POLL=1)
t("e2_pos_relaxed", "positive-control", "detector for e2_* tests",
  "Remote poll with ld.relaxed.gpu, no acquire", placement="e2", home="producer", C_POLL=5)

# --- "empty" edge: load buffering, forbidden = consumer's round-k read returns round k+1 ---
t("empty_tut01", "trust-base", L_EMPTY, "Tutorial 01 empty edge: consumer reads, bar.sync, "
  "membar.sys; st.relaxed.sys (local flag); producer polls it remotely with "
  "ld.global.acquire.gpu, then overwrites", controls=["empty_pos_relaxed"], P_POLL=0)
t("empty_sys", "sanctioned", "A0/P1 baseline for the WAR edge",
  "Producer's remote poll uses ld.global.acquire.sys", controls=["empty_pos_relaxed"],
  P_POLL=1)
t("empty_pos_relaxed", "positive-control", "detector for empty_* tests (may never fire: "
  "load buffering may not be observable on this hardware)", "Consumer drops the fence "
  "before 'empty'; producer polls with ld.relaxed.gpu", P_POLL=5, C_FENCE=3)
t("selftest_notify_before_read", "self-test", "harness: FUTURE counting works",
  "Consumer signals 'empty' BEFORE reading, then sleeps 20 us", C_ORDER=2)

# --- tutorial 01 as written: both edges, ring over both GPUs ---
t("tut01_ring", "trust-base", f"{L_FULL} + {L_EMPTY}", "Tutorial 01 scopes on both edges; "
  "half the instances GPU0->GPU1, half GPU1->GPU0", placement="ring",
  controls=["full_pos_relaxed", "empty_pos_relaxed"], P_POLL=0, C_POLL=0)

# --- DeepEP V1: ld.global.nc.L1::no_allocate on data a peer writes during the kernel ---
L_NC = "lemma sm90_nc_noalloc_coherent (DeepEP V1 'tested to be guaranteed on Hopper'); #23, #25"
t("nc_deepep", "trust-base", L_NC, "Reader ld.global.acquire.sys, then DeepEP's exact "
  "ld.global.nc.L1::no_allocate.L2::256B payload loads", controls=["nc_pos_plain"],
  C_POLL=1, C_LD=2)
t("nc_deepep_warm", "trust-base", L_NC, "As nc_deepep with an L1 warm-up weak load first",
  controls=["nc_pos_plain_warm"], C_POLL=1, C_LD=2, C_WARM=1)
t("nc_pos_plain", "positive-control", "detector for nc_deepep (dossier variant v)",
  "Same, payload via plain ld.global.nc (L1 allocation allowed)", C_POLL=1, C_LD=3)
t("nc_pos_plain_warm", "positive-control", "detector for nc_deepep_warm",
  "As nc_pos_plain with the warm-up load", C_POLL=1, C_LD=3, C_WARM=1)

STRIDE = 4096          # bytes per instance slot: flag in line 0, payload in lines 1..8
PAYLOAD_OFF = 128
OUT_STRIDE = 16     # u64 counters per (instance, role); matches kernels.cu
DEFAULTS = dict(rounds=200_000, instances=64)
QUICK = dict(rounds=5_000, instances=16)


def ptx_of(knobs):
    out = {k: PTX[k][v] for k, v in knobs.items() if k in PTX}
    if knobs["C_WARM"]:
        out["C_WARM"] = ["ld.global.v4.b32 (warm-up, value folded into a sink)"]
    return out


def options(knobs):
    return ("-std=c++17",) + tuple(f"-D{k}={v}" for k, v in sorted(knobs.items()))


def compile_only(arch):
    """Compile every test's source with NVRTC; no GPU or driver needed."""
    from cupy_backends.cuda.libs import nvrtc
    RESULTS.mkdir(exist_ok=True)
    ptx_dir = RESULTS / "ptx"
    ptx_dir.mkdir(exist_ok=True)
    report, ok_all = {}, True
    for name, spec in T.items():
        entry = {}
        for target in (f"compute_{arch}", f"sm_{arch}"):
            prog = nvrtc.createProgram(SRC, "kernels.cu", [], [])
            try:
                # -ftz=true: CuPy's RawModule adds it at run time, so mirror it here.
                nvrtc.compileProgram(prog, list(options(spec["knobs"])) + ["-ftz=true", f"-arch={target}"])
                if target.startswith("compute"):
                    ptx = nvrtc.getPTX(prog)
                    ptx = ptx.decode() if isinstance(ptx, bytes) else ptx
                    (ptx_dir / f"{name}.ptx").write_text(ptx)
                    want = [s.split(" (")[0] for v in ptx_of(spec["knobs"]).values() for s in v] + ALWAYS_PTX
                    missing = sorted({s for s in want if s not in ptx})
                    entry["ptx_missing"] = missing
                    entry["ptx_ok"] = not missing
                else:
                    cubin = nvrtc.getCUBIN(prog)
                    entry["cubin_bytes"] = len(cubin)
                    entry["cubin_sha256"] = hashlib.sha256(cubin).hexdigest()
                entry[target] = "ok"
            except nvrtc.NVRTCError as e:
                entry[target] = f"FAILED: {e}\n{nvrtc.getProgramLog(prog)}"
            finally:
                nvrtc.destroyProgram(prog)
        good = all(entry.get(x) == "ok" for x in (f"compute_{arch}", f"sm_{arch}")) \
            and entry.get("ptx_ok", False)
        ok_all &= good
        report[name] = entry
        print(f"{'ok  ' if good else 'FAIL'} {name:30s} sm_{arch} cubin "
              f"{entry.get('cubin_bytes', '-')} B; missing PTX: {entry.get('ptx_missing')}")
    (RESULTS / f"compile_sm{arch}.json").write_text(json.dumps(
        {"nvrtc": ".".join(map(str, nvrtc.getVersion())), "arch": arch, "tests": report},
        indent=2))
    return 0 if ok_all else 1


def plan(spec, n_inst):
    """Per instance: (producer GPU, consumer GPU, home GPU)."""
    p = spec["placement"]
    rows = []
    for i in range(n_inst):
        if p == "cross":
            pd, cd = 0, 1
        elif p == "e2":
            pd, cd = 1, 0
        elif p == "same":
            pd, cd = 0, 0
        else:  # ring
            pd, cd = (0, 1) if i % 2 == 0 else (1, 0)
        rows.append((pd, cd, cd if spec["home"] == "consumer" else pd))
    return rows


def enable_peer_access(cp):
    rt = cp.cuda.runtime
    for a, b in ((0, 1), (1, 0)):
        if not rt.deviceCanAccessPeer(a, b):
            raise RuntimeError(f"GPU {a} cannot access GPU {b} (no P2P); cannot run cross-GPU tests")
        with cp.cuda.Device(a):
            try:
                rt.deviceEnablePeerAccess(b)
            except rt.CUDARuntimeError as e:
                if "AlreadyEnabled" not in str(e):
                    raise


def run_one(name, a, env):
    import cupy as cp
    import numpy as np

    spec = T[name]
    rows = plan(spec, a.instances)
    devs = sorted({d for r in rows for d in r[:2]})
    if max(d for r in rows for d in r) >= env["gpu_count"]:
        return {"test": name, "status": "skipped", "reason": "needs 2 GPUs"}
    if spec["placement"] != "same" or a.stress == "nvlink":
        enable_peer_access(cp)

    # Home buffers: one zeroed buffer per GPU holding the slots of the instances homed there.
    slot, homes = {}, {}
    for i, (_, _, h) in enumerate(rows):
        slot[i] = sum(1 for j in range(i) if rows[j][2] == h)
    for h in sorted({r[2] for r in rows}):
        n = sum(1 for r in rows if r[2] == h)
        with cp.cuda.Device(h):
            homes[h] = cp.zeros(n * STRIDE // 4, dtype=cp.uint32)
    flag_ptr = np.array([homes[r[2]].data.ptr + slot[i] * STRIDE for i, r in enumerate(rows)],
                        dtype=np.uint64)
    payload_ptr = flag_ptr + np.uint64(PAYLOAD_OFF)

    launches = []
    for d in devs:
        blocks = [(0, i) for i, r in enumerate(rows) if r[0] == d] + \
                 [(1, i) for i, r in enumerate(rows) if r[1] == d]
        n_lit = len(blocks)
        blocks += [(2, 0)] * (a.stress_blocks if a.stress != "none" else 0)
        with cp.cuda.Device(d):
            sms = cp.cuda.runtime.getDeviceProperties(d)["multiProcessorCount"]
            if len(blocks) > sms * 8:
                raise RuntimeError(f"{len(blocks)} blocks on GPU {d} exceed the co-residency "
                                   f"guard ({sms} SMs x 8); lower --instances/--stress-blocks")
            kern = cp.RawModule(code=SRC, options=options(spec["knobs"])).get_function("litmus")
            scratch_dev = d if a.stress != "nvlink" else 1 - d
            scratch = None
            if a.stress != "none":
                with cp.cuda.Device(scratch_dev):
                    scratch = cp.zeros(a.stress_mib * 2**20 // 4, dtype=cp.uint32)
            launches.append(dict(
                dev=d, kern=kern, grid=len(blocks), n_lit=n_lit, scratch=scratch,
                roles=cp.asarray(np.array([b[0] for b in blocks], dtype=np.int32)),
                insts=cp.asarray(np.array([b[1] for b in blocks], dtype=np.int32)),
                fptr=cp.asarray(flag_ptr), pptr=cp.asarray(payload_ptr),
                out=cp.zeros(a.instances * 2 * OUT_STRIDE, dtype=cp.uint64),
                done=cp.zeros(1, dtype=cp.uint32),
                stream=cp.cuda.Stream(non_blocking=True)))
    for d in range(env["gpu_count"]):
        cp.cuda.Device(d).synchronize()

    t0 = time.time()
    for L in launches:
        with cp.cuda.Device(L["dev"]):
            sp = L["scratch"]
            L["kern"]((L["grid"],), (32,), (
                L["roles"], L["insts"], L["fptr"], L["pptr"], L["out"], L["done"],
                np.uint32(L["n_lit"]), np.uint64(sp.data.ptr if sp is not None else 0),
                np.uint64(sp.size if sp is not None else 0), np.uint32(a.rounds),
                np.uint64(a.max_spin), np.uint32(a.jitter_ns), np.uint32(a.seed),
                np.uint64(a.stress_cap)), stream=L["stream"])
    for L in launches:
        with cp.cuda.Device(L["dev"]):
            L["stream"].synchronize()
    elapsed = time.time() - t0

    out = np.zeros((a.instances, 2, OUT_STRIDE), dtype=np.uint64)
    for L in launches:
        with cp.cuda.Device(L["dev"]):
            out += cp.asnumpy(L["out"]).reshape(a.instances, 2, OUT_STRIDE)
    cons, prod = out[:, 1, :], out[:, 0, :]
    hist = {k: int(cons[:, j].sum()) for k, j in (("ok", 1), ("stale", 2), ("future", 3),
                                                   ("mixed", 4))}
    firsts = [{"instance": i, "round": int(cons[i, 6]), "min_word": int(cons[i, 7] >> 32),
               "max_word": int(cons[i, 7] & 0xFFFFFFFF), "expected": int(cons[i, 6]) + 1}
              for i in range(a.instances) if cons[i, 6] != 2**64 - 1][:10]
    return {
        "test": name, "status": "ran", "kind": spec["kind"], "serves": spec["serves"],
        "description": spec["why"], "controls": spec["controls"], "tag": a.tag,
        "placement": spec["placement"], "home": spec["home"],
        "instances": [{"producer_gpu": r[0], "consumer_gpu": r[1], "home_gpu": r[2]}
                      for r in rows[:2]] + (["..."] if len(rows) > 2 else []),
        "knobs": spec["knobs"], "ptx": ptx_of(spec["knobs"]),
        "iterations_planned": a.rounds * a.instances, "rounds": a.rounds,
        "n_instances": a.instances,
        "iterations_completed": int(cons[:, 0].sum()),
        "histogram": hist, "bad_count": hist["stale"] + hist["future"] + hist["mixed"],
        "timeouts": {"producer": int(prod[:, 5].sum()), "consumer": int(cons[:, 5].sum())},
        "first_bad": firsts,
        "stress": a.stress, "stress_blocks": a.stress_blocks if a.stress != "none" else 0,
        "jitter_ns": a.jitter_ns, "seed": a.seed, "max_spin": a.max_spin,
        "elapsed_s": round(elapsed, 3), "env": env,
    }


def run(a):
    import env_check
    if a.quick:
        a.rounds = a.rounds or QUICK["rounds"]
        a.instances = a.instances or QUICK["instances"]
    a.rounds = a.rounds or DEFAULTS["rounds"]
    a.instances = a.instances or DEFAULTS["instances"]
    names = list(T) if a.tests == ["all"] else a.tests
    unknown = [n for n in names if n not in T]
    if unknown:
        sys.exit(f"unknown tests: {unknown}; see `litmus.py list`")
    RESULTS.mkdir(exist_ok=True)
    env = env_check.collect()
    failures = 0
    for name in names:
        suffix = f"__{a.tag}" if a.tag else ""
        try:
            res = run_one(name, a, env)
        except Exception as e:  # keep going; the error is part of the result
            failures += 1
            res = {"test": name, "status": "error", "tag": a.tag, "error": repr(e),
                   "traceback": traceback.format_exc(), "env": env}
        (RESULTS / f"{name}{suffix}.json").write_text(json.dumps(res, indent=2))
        if res["status"] == "ran":
            h = res["histogram"]
            print(f"{name+suffix:38s} {res['kind']:16s} iters={res['iterations_completed']:>10} "
                  f"stale={h['stale']} future={h['future']} mixed={h['mixed']} "
                  f"timeouts={sum(res['timeouts'].values())} {res['elapsed_s']}s", flush=True)
        else:
            print(f"{name+suffix:38s} {res['status'].upper()}: "
                  f"{res.get('reason') or res.get('error')}", flush=True)
    return 1 if failures else 0


def verdicts(results):
    """results: {(test, tag): json}. Returns {(test, tag): verdict string}."""
    fired = {k: r.get("status") == "ran" and r["bad_count"] > 0 for k, r in results.items()}
    v = {}
    for (name, tag), r in results.items():
        if r.get("status") != "ran":
            v[(name, tag)] = r.get("status", "?").upper()
            continue
        kind, bad, tmo = r["kind"], r["bad_count"], sum(r["timeouts"].values())
        ctl = [fired.get((c, tag)) for c in r["controls"]]
        if kind == "self-test":
            v[(name, tag)] = "OK (detects)" if bad else "HARNESS-BROKEN"
        elif kind == "positive-control":
            v[(name, tag)] = "FIRED" if bad else "SILENT"
        elif bad:
            v[(name, tag)] = "FAIL (violations)"
        elif tmo:
            v[(name, tag)] = "INCOMPLETE (timeouts)"
        elif kind == "sanctioned":
            v[(name, tag)] = "PASS"
        elif ctl and all(ctl):
            v[(name, tag)] = "PASS (evidence)" if kind == "trust-base" else "NONE-SEEN"
        else:
            v[(name, tag)] = "INCONCLUSIVE (control silent or missing)"
    return v


def summarize(_a):
    results = {}
    for f in sorted(RESULTS.glob("*.json")):
        if f.name in ("env.json", "summary.json") or f.name.startswith("compile_"):
            continue
        r = json.loads(f.read_text())
        results[(r["test"], r.get("tag") or "")] = r
    v = verdicts(results)
    rows = []
    for (name, tag), r in results.items():
        rows.append(dict(test=name, tag=tag, kind=r.get("kind"), verdict=v[(name, tag)],
                         iterations=r.get("iterations_completed"), bad=r.get("bad_count"),
                         histogram=r.get("histogram"), timeouts=r.get("timeouts"),
                         controls=r.get("controls"), serves=r.get("serves")))
        print(f"{name:30s} {tag or '-':8s} {r.get('kind') or '':16s} {v[(name, tag)]:40s} "
              f"iters={r.get('iterations_completed')} bad={r.get('bad_count')}")
    broken = [r for r in rows if r["verdict"] == "HARNESS-BROKEN"]
    if broken:
        print("\nWARNING: a self-test did not fire; treat every verdict as untrustworthy.")
    (RESULTS / "summary.json").write_text(json.dumps(rows, indent=2))
    return 1 if broken else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    c = sub.add_parser("compile-only")
    c.add_argument("--arch", default="90")
    r = sub.add_parser("run")
    r.add_argument("tests", nargs="+", help="test names, or 'all'")
    r.add_argument("--quick", action="store_true", help=f"defaults {QUICK}")
    r.add_argument("--rounds", type=int, help=f"rounds per instance (default {DEFAULTS['rounds']})")
    r.add_argument("--instances", type=int, help=f"producer/consumer pairs (default {DEFAULTS['instances']})")
    r.add_argument("--max-spin", type=int, default=1 << 24, help="polls before a wait times out")
    r.add_argument("--stress", choices=["none", "local", "nvlink"], default="none",
                   help="extra blocks hammering a large buffer on this GPU (local) or the peer (nvlink)")
    r.add_argument("--stress-blocks", type=int, default=64)
    r.add_argument("--stress-mib", type=int, default=256, help="stress buffer size (> L2)")
    r.add_argument("--stress-cap", type=int, default=1 << 26, help="safety bound on stress loop")
    r.add_argument("--jitter-ns", type=int, default=0, help="random sleep 0..N ns per round")
    r.add_argument("--seed", type=int, default=12345)
    r.add_argument("--tag", default="", help="suffix for result files, e.g. 'stress'")
    sub.add_parser("summarize")
    a = ap.parse_args()
    if a.cmd == "list":
        for n, s in T.items():
            print(f"{n:30s} {s['kind']:16s} {s['placement']:5s} {s['why']}\n{'':30s} serves: {s['serves']}")
        return 0
    if a.cmd == "compile-only":
        return compile_only(a.arch)
    if a.cmd == "run":
        return run(a)
    return summarize(a)


if __name__ == "__main__":
    sys.exit(main())
