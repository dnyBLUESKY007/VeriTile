"""Per-rank harness of the `internode-logical-env-001` batch: does a launch come up as the nodes it was asked for?

Every rank starts the runtime through the pinned `initialize_distributed`,
reports the topology it was given, puts one tensor into the next rank's
symmetric memory and stops. It launches no kernel. It has not run on the
development host, which has no GPU.

What the development side expects of rank r of 2 (the GPU side never judges):

    topology   rank, world size, LOCAL_WORLD_SIZE, nodes, LOCAL_RANK, CUDA device index
               two logical nodes: r, 2, 1, 2, 0, 0        one node: r, 2, 2, 1, r, r
    received   four times 100 + the other rank
"""
import os


def put_barrier(rank, ranks, inputs, outputs, launch):
    import nvshmem.core
    import torch
    from triton_dist.utils import (TorchStreamWrapper, finalize_distributed, initialize_distributed,
                                   nvshmem_barrier_all_on_stream, nvshmem_create_tensor, nvshmem_free_tensor_sync)
    group = initialize_distributed()  # selects GPU LOCAL_RANK; the torch process group over all ranks, then NVSHMEM
    world, local = int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_WORLD_SIZE'])
    device = torch.cuda.current_device()
    # What the launcher and the runtime report, not what gpurun passed in: the dev side compares the two.
    seen = [group.rank(), group.size(), local, world // local, int(os.environ['LOCAL_RANK']), device]
    outputs['topology'].copy_(torch.tensor(seen, dtype=torch.int32))
    print(f'rank {seen[0]} of {seen[1]}: LOCAL_WORLD_SIZE={local} nodes={seen[3]} LOCAL_RANK={seen[4]} '
          f'device={device} {torch.cuda.get_device_name(device)}', flush=True)

    sent = nvshmem_create_tensor((4, ), torch.int64)  # symmetric: a put needs both ends in the NVSHMEM heap
    received = nvshmem_create_tensor((4, ), torch.int64)
    sent.fill_(100 + rank)
    received.fill_(-1)
    stream = torch.cuda.current_stream()
    nvshmem_barrier_all_on_stream(stream)  # every rank has filled `received` before a put can land in it
    nvshmem.core.put(received, sent, remote_pe=(rank + 1) % ranks, stream=TorchStreamWrapper(stream))
    nvshmem_barrier_all_on_stream(stream)  # every put has landed
    torch.cuda.synchronize()
    outputs['received'].copy_(received)

    nvshmem_free_tensor_sync(sent)
    nvshmem_free_tensor_sync(received)
    finalize_distributed()
