"""GPU-side runner of the exchange protocol (documents/GpuExchangeProtocol.md).

Run from the repository root with `scripts` on the module path:

    PYTHONPATH=scripts python -m gpurun check-env <batch-dir>
    PYTHONPATH=scripts python -m gpurun run <batch-dir> --out <result-dir>

`gpurun` reads a request directory and writes a result directory. It never
judges: no file it writes says whether a result is right.

A test's `entry` is `<path inside the repository>::<function>`. The function
is a launch harness kept next to the kernel:

    def run(inputs, outputs, launch): ...

`inputs` and `outputs` map tensor names to `torch` tensors on the GPU; every
output arrives filled with its sentinel and is written in place. `launch` is
the test's `launch` object. The harness binds arguments and launches the
kernel (`kernel[grid](...)` for Triton, the compiled callable for TileLang).
A `torch` reference entry has the same signature.
"""
