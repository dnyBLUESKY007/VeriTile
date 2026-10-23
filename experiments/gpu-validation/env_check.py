"""Print and record the GPU environment: results/env.json.

Usage: uv run --no-sync python env_check.py
Exits non-zero if fewer than two GPUs are visible or GPU 0 and GPU 1 cannot access each other.
"""

import datetime
import json
import pathlib
import platform
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
RESULTS = HERE / "results"


def _run(cmd):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return (p.stdout + p.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"unavailable: {e}"


def package_commit():
    head = _run(["git", "-C", str(HERE), "rev-parse", "HEAD"])
    dirty = _run(["git", "-C", str(HERE), "status", "--porcelain", "--", "."])
    if not head or " " in head:
        return {"commit": "unknown", "dirty": None}
    return {"commit": head, "dirty": bool(dirty)}


def _ver(v):
    return f"{v // 1000}.{(v % 1000) // 10}"


def collect():
    """Lightweight facts recorded in every result file (no nvidia-smi)."""
    import cupy as cp
    from cupy_backends.cuda.libs import nvrtc

    rt = cp.cuda.runtime
    n = rt.getDeviceCount()
    gpus = []
    for d in range(n):
        p = rt.getDeviceProperties(d)
        name = p["name"].decode() if isinstance(p["name"], bytes) else p["name"]
        gpus.append({
            "index": d,
            "name": name,
            "compute_capability": f"{p['major']}.{p['minor']}",
            "sm_count": p["multiProcessorCount"],
            "pci_bus_id": p.get("pciBusID"),
            "l2_bytes": p.get("l2CacheSize"),
        })
    peer = {}
    for a in range(n):
        for b in range(n):
            if a != b:
                peer[f"{a}->{b}"] = bool(rt.deviceCanAccessPeer(a, b))
    return {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python": platform.python_version(),
        "cupy": cp.__version__,
        "driver_cuda_version": _ver(rt.driverGetVersion()),
        "runtime_cuda_version": _ver(rt.runtimeGetVersion()),
        "nvrtc_version": ".".join(map(str, nvrtc.getVersion())),
        "gpu_count": n,
        "gpus": gpus,
        "peer_access": peer,
        "package": package_commit(),
    }


def main():
    RESULTS.mkdir(exist_ok=True)
    try:
        env = collect()
    except Exception as e:  # no driver / no GPU: still record what nvidia-smi says
        env = {"error": repr(e), "gpu_count": 0, "package": package_commit()}
    env["nvidia_smi"] = _run(["nvidia-smi"])
    env["nvidia_smi_query"] = _run(["nvidia-smi", "--query-gpu=index,name,driver_version,"
                                    "memory.total,clocks.max.sm,persistence_mode,compute_mode,mig.mode.current",
                                    "--format=csv"])
    env["nvidia_smi_topo"] = _run(["nvidia-smi", "topo", "-m"])
    env["nvidia_smi_nvlink"] = _run(["nvidia-smi", "nvlink", "--status"])
    env["uv_version"] = _run(["uv", "--version"])
    env["os"] = platform.platform()
    (RESULTS / "env.json").write_text(json.dumps(env, indent=2))

    if "error" in env:
        print(f"PROBLEM: CUDA is not usable: {env['error']}\n"
              f"nvidia-smi says:\n{env['nvidia_smi']}\nwrote {RESULTS / 'env.json'}",
              file=sys.stderr)
        return 1
    print(f"CuPy {env['cupy']}  NVRTC {env['nvrtc_version']}  "
          f"driver supports CUDA {env['driver_cuda_version']}  runtime {env['runtime_cuda_version']}")
    for g in env["gpus"]:
        print(f"GPU {g['index']}: {g['name']}  sm_{g['compute_capability'].replace('.', '')}  "
              f"{g['sm_count']} SMs")
    print("peer access:", env["peer_access"])
    print("\n--- nvidia-smi topo -m ---\n" + env["nvidia_smi_topo"])
    print(f"\nwrote {RESULTS / 'env.json'}")

    problems = []
    if env["gpu_count"] < 2:
        problems.append("fewer than 2 GPUs visible (check CUDA_VISIBLE_DEVICES)")
    elif not (env["peer_access"].get("0->1") and env["peer_access"].get("1->0")):
        problems.append("GPU 0 and GPU 1 cannot access each other (no P2P)")
    if any(g["compute_capability"] != "9.0" for g in env["gpus"][:2]):
        problems.append("GPU 0/1 are not compute capability 9.0 (Hopper); results do not "
                        "count for the sm_90 trust base")
    for p in problems:
        print("PROBLEM:", p, file=sys.stderr)
    return 1 if env["gpu_count"] < 2 or "P2P" in " ".join(problems) else 0


if __name__ == "__main__":
    sys.exit(main())
