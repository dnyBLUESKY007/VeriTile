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

In a `dist` batch a test runs as `ranks` processes, one GPU each, and its
`entry` is called once in every rank (see `dist.py`):

    def run(rank, ranks, inputs, outputs, launch): ...

The operator, never a request, sets through the environment:

    CUDA_VISIBLE_DEVICES  the GPUs gpurun may use; rank r of a `dist` test uses visible GPU r
    GPURUN_NODES          number of nodes gpurun may use (default 1); `requires.nodes`
                          above it is refused; reported as `nodes` in env.json
    GPURUN_HOSTS          comma-separated host names of those nodes, this node first, for a
                          launcher that must name them; removed from the logs
"""
