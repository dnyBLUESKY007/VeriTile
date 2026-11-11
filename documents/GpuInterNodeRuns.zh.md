[English](GpuInterNodeRuns.md) | [中文](GpuInterNodeRuns.zh.md)

# 跨节点程序在单机上的覆盖范围

Issue：[#353](https://github.com/dnyBLUESKY007/VeriTile/issues/353)。检查日期：2026-11-10。状态：源码检查完成；kernel 运行验证等待 GPU 交换执行器。尚未安排第二个物理节点。

## 硬件与范围

所检查的主机可见六张 H800 PCIe，计算能力均为 9.0。NVLink 卡对为 0–1、2–3、4–5。每个分布式请求计划只使用两张卡；本次环境检查使用当时空闲的 4–5 卡对。设备选择不代表资源预留，每次执行前都必须重新检查占用。

下述源码观察均针对 [Triton-distributed `7908e4ea9010bb2238f9f05b28c904fd4b71c60a`](https://github.com/ByteDance-Seed/Triton-distributed/tree/7908e4ea9010bb2238f9f05b28c904fd4b71c60a)。它们说明控制流和配置约束，不代表运行成功或数值正确。

普通单机双进程启动时，`WORLD_SIZE=2`、`LOCAL_WORLD_SIZE=2`、`NNODES=1`。名称带有 `inter_node` 的程序此时可能只执行退化的单节点路径，跳过它原本需要验证的跨节点通信。

## 源码发现

| 程序 | 单机两卡的源码评估 | 未覆盖部分或阻碍 |
| --- | --- | --- |
| [Tutorial 03](https://github.com/ByteDance-Seed/Triton-distributed/blob/7908e4ea9010bb2238f9f05b28c904fd4b71c60a/tutorials/03-inter-node-allgather.py) | 1D all-gather 通过 NVSHMEM 向另一 rank 发送。2D all-gather 接受 `NNODES=1`，执行节点内 peer 分支。尚未实测运行。 | `NNODES=1` 时不覆盖 2D 的相同 local rank、不同节点发送及其接收／转发顺序。 |
| [Tutorial 03a](https://github.com/ByteDance-Seed/Triton-distributed/blob/7908e4ea9010bb2238f9f05b28c904fd4b71c60a/tutorials/03a-inter-node-allgather.py) | 此固定版本不是 NVIDIA 教程：导入 `pyrocshmem`，使用 ROCSHMEM 设备上下文和信号操作。 | 无法原样在 H800/CUDA 环境执行。需要 AMD/ROCm 主机或单独审阅的 NVIDIA 移植。本次未移植。 |
| [Tutorial 06](https://github.com/ByteDance-Seed/Triton-distributed/blob/7908e4ea9010bb2238f9f05b28c904fd4b71c60a/tutorials/06-inter-node-reduce-scatter.py) | 代码明确支持 `nnodes == 1`，此时返回本地 reduce-scatter 缓冲区。尚未实测运行。 | 只有 `nnodes > 1` 才启动 `kernel_inter_node_p2p_for_same_local_rank`；普通单机双 rank 不会执行该 kernel。 |
| [跨节点 all-to-all kernel](https://github.com/ByteDance-Seed/Triton-distributed/blob/7908e4ea9010bb2238f9f05b28c904fd4b71c60a/python/triton_dist/kernels/nvidia/all_to_all_vdev_2d_offset_inter_node.py) | 核心上下文可以描述 `world_size=2, local_world_size=2`。现有测试调用 `create_context` 时未覆盖默认的 `local_world_size=8`。 | 现有测试直接使用双 rank 会违反 `world_size % local_world_size == 0`。开发端适配器必须显式传入 2。只有一个逻辑节点时跳过异节点接收等待分支。本次未修改测试或 kernel。 |
| [跨节点序列并行 all-gather attention kernel](https://github.com/ByteDance-Seed/Triton-distributed/blob/7908e4ea9010bb2238f9f05b28c904fd4b71c60a/python/triton_dist/kernels/nvidia/sp_ag_attention_inter_node.py) | 现有测试由 `WORLD_SIZE / LOCAL_WORLD_SIZE` 得到 `NNODES`，并将二者传入上下文。单机双 rank 配置执行节点内 all-gather 路径。尚未实测运行。 | `NNODES=1` 时不覆盖异节点 put 及跨节点接收／转发同步。 |

## 启动约束

固定版本的 `scripts/launch.sh` 识别 `--nproc_per_node=2`（必须带 `=`）。默认分支通过 `nvidia-smi` 统计物理设备，因此在这台六卡机器上仅设置 `CUDA_VISIBLE_DEVICES` 不够。后续开发端请求必须明确选择两张可见卡和两个进程。

不能只修改 `LOCAL_WORLD_SIZE` 来伪造拓扑：初始化会将 `LOCAL_RANK` 映射到 CUDA 设备，对称 peer 缓冲区的构造也依赖节点内分组。以后如在一台物理主机上模拟两个逻辑节点，需要明确且经过审阅的 launcher 与 batch 描述。即使它执行了异节点源码分支，也不能证明跨主机网卡、RDMA/IBGDA、隔离、延迟或故障行为。

## 受影响请求与完成门槛

本次检查未提交或运行新批次。下表是建议的 batch ID，并非已存在的 manifest 或结果：

| 建议批次 | 计划检查 | 当前处置 |
| --- | --- | --- |
| `internode-ag03-single-host-001` | Tutorial 03 的双本地 rank 1D/2D 路径 | 等待 `gpurun` 和开发端固定 manifest；只覆盖单节点。 |
| `internode-rs06-single-host-001` | Tutorial 06 的双本地 rank 路径 | 等待 `gpurun` 和 manifest；不覆盖其跨节点 P2P kernel。 |
| `internode-a2av-single-host-001` | 显式使用 `local_world_size=2` 的 all-to-all 核心 | 等待 `gpurun` 及开发端适配器／manifest；现有测试假定 8。 |
| `internode-sp-attn-single-host-001` | 双本地 rank 序列并行 attention | 等待 `gpurun` 和 manifest；异节点分支仍未覆盖。 |
| `internode-ag03a-rocm-001` | 原样执行 tutorial 03a | H800 无法提供；需要 AMD 环境或明确审阅的移植。 |

协议 v1 下，要求两个物理节点的请求仍应返回 `skipped_env`。不能把逻辑节点实验报告为双物理节点结果。

本文不构成 #353 运行验收通过。在上述可执行请求产出结果、开发端记录比较与覆盖边界之前，或用户明确接受其他处置之前，Issue 保持开放。用户已明确要求首个正式批次等待 `gpurun`；本次检查不绕过这一依赖。
