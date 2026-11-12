"""Per-rank harness of the `dist-env-check` batch: start the distributed runtime, report, stop.

It launches no kernel. It has not run on the development host, which has no GPU.
"""


def init_finalize(rank, ranks, inputs, outputs, launch):
    from triton_dist.utils import finalize_distributed, initialize_distributed
    group = initialize_distributed()  # the torch process group over all ranks, then NVSHMEM
    # What the runtime reports, not what gpurun passed in: the dev side compares the two.
    outputs['world'][0] = group.rank()
    outputs['world'][1] = group.size()
    finalize_distributed()
