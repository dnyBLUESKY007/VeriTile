"""CPU-only self-check of litmus.py's placement and verdict logic (no GPU, no CuPy needed).

Usage: uv run --no-sync python check_logic.py
"""

import litmus as L


def fake(name, bad=0, timeouts=0, tag=""):
    s = L.T[name]
    return {"status": "ran", "kind": s["kind"], "controls": s["controls"], "tag": tag,
            "bad_count": bad, "timeouts": {"producer": timeouts, "consumer": 0}}


def main():
    # Every control named by a test exists and is a positive control.
    for n, s in L.T.items():
        for c in s["controls"]:
            assert L.T[c]["kind"] == "positive-control", (n, c)
        for k, v in s["knobs"].items():
            assert k in ("C_WARM", "C_ORDER") or v in L.PTX[k], (n, k, v)

    assert L.plan(L.T["full_tut01"], 2) == [(0, 1, 1), (0, 1, 1)]          # data on consumer
    assert L.plan(L.T["e2_remote_poll_sys_writer"], 1) == [(1, 0, 1)]      # data on producer
    assert L.plan(L.T["same_tut01"], 1) == [(0, 0, 0)]
    assert L.plan(L.T["tut01_ring"], 2) == [(0, 1, 1), (1, 0, 0)]

    r = {(n, ""): fake(n) for n in ("full_tut01", "full_pos_relaxed", "full_sys",
                                     "selftest_read_before_wait", "empty_tut01",
                                     "empty_pos_relaxed", "e2_remote_poll_gpu_writer",
                                     "e2_pos_relaxed")}
    r[("full_pos_relaxed", "")]["bad_count"] = 5
    r[("selftest_read_before_wait", "")]["bad_count"] = 9
    r[("e2_pos_relaxed", "")]["bad_count"] = 1
    v = L.verdicts(r)
    assert v[("full_tut01", "")] == "PASS (evidence)", v
    assert v[("full_sys", "")] == "PASS"
    assert v[("selftest_read_before_wait", "")] == "OK (detects)"
    assert v[("empty_pos_relaxed", "")] == "SILENT"
    assert v[("empty_tut01", "")].startswith("INCONCLUSIVE")       # its control never fired
    assert v[("e2_remote_poll_gpu_writer", "")] == "NONE-SEEN"     # probe, control fired

    r[("full_tut01", "")]["bad_count"] = 1
    assert L.verdicts(r)[("full_tut01", "")] == "FAIL (violations)"
    r[("full_tut01", "")].update(bad_count=0, timeouts={"producer": 1, "consumer": 0})
    assert L.verdicts(r)[("full_tut01", "")] == "INCOMPLETE (timeouts)"
    # A control from another tag (e.g. stress) does not count.
    r2 = {("full_tut01", "stress"): fake("full_tut01", tag="stress"),
          ("full_pos_relaxed", ""): fake("full_pos_relaxed", bad=3)}
    assert L.verdicts(r2)[("full_tut01", "stress")].startswith("INCONCLUSIVE")
    print(f"check_logic: ok ({len(L.T)} tests)")


if __name__ == "__main__":
    main()
