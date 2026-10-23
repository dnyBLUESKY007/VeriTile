[English](README.md) | [中文](README.zh.md)

# VeriTile GPU 验证第一阶段：在 2×H800 上运行 PTX litmus 测试

本包运行一组小型 **litmus 测试**，检查两块通过 NVLink 相连的 Hopper（sm_90）GPU 之间如何对内存操作排序。VeriTile 是一个面向 GPU kernel 的 Lean 验证器。它的第一版把 PTX 与 NVSHMEM 文档没有承诺的硬件行为建模为具名的 **trust-base 引理**。只有当覆盖该引理的运行时测试在这类机器上通过后，证明中才可以使用它。这些测试就是为此提供证据。

运行测试不需要理解背后的理论。一个脚本完成全部工作，并生成一个 `.tar.gz` 文件供回传。

## 需要准备什么

- 一台 Linux 服务器，配有**两块通过 NVLink 相连的 NVIDIA H800（或 H100）GPU**，测试期间保持空闲（没有其他任务，MIG 关闭）。
- **NVIDIA 驱动**。用 `nvidia-smi` 查看：
  - 默认的 CUDA 12 版本需要驱动 **>= 525**（更新的驱动同样可用）；
  - 只有选择 CUDA 13 版本时才需要驱动 **>= 580**。
  **不需要**安装 CUDA Toolkit。CUDA 编译器（NVRTC）从 NVIDIA 的 Python wheel 安装。
- 能访问 `pypi.org`、`github.com` 和 `astral.sh`（如果系统没有 Python 3.10-3.12，还需要能从 `github.com` releases 下载 Python）。
- 约 1 GB 磁盘空间用于 Python 环境。
- `git`、`bash`、`tar`。

## 第 1 步：安装 uv

[uv](https://docs.astral.sh/uv/) 会安装合适的 Python，以及 `uv.lock` 中锁定的精确包版本。锁文件由 uv 0.13.1 生成，请使用该版本或更新版本：

```bash
curl -LsSf https://astral.sh/uv/0.13.1/install.sh | sh
# 然后新开一个 shell，或执行：source $HOME/.local/bin/env
uv --version
```

## 第 2 步：获取代码

```bash
git clone --branch gpu-validation https://github.com/dnyBLUESKY007/VeriTile.git
cd VeriTile/experiments/gpu-validation
```

## 第 3 步：运行

```bash
./run_all.sh --quick     # 先做冒烟测试，约 1-3 分钟（估计值）
./run_all.sh             # 完整运行，约 5-20 分钟（估计值，尚未实测）
```

改用 CUDA 13 wheel（驱动 >= 580）：`CUDA_VARIANT=cu13 ./run_all.sh`。

如果通过 NVLink 相连的两块 GPU 不是 GPU 0 和 GPU 1，先选定它们，例如 `CUDA_VISIBLE_DEVICES=2,3 ./run_all.sh`。

`run_all.sh` 依次执行：

1. `uv sync --locked --extra cu12`：按 `uv.lock` 的精确锁定创建 `.venv/`，包含 Python 3.12、CuPy 14.2.0、NumPy 和 NVRTC。
2. `check_logic.py`：在 CPU 上自检统计代码。
3. `litmus.py compile-only`：为 `sm_90` 编译全部 27 个测试 kernel（不需要 GPU），并检查生成的 PTX 中包含每条预期的 PTX 指令。
4. `env_check.py`：把 GPU 名称、compute capability、驱动/运行时/NVRTC 版本、`nvidia-smi`、`nvidia-smi topo -m`、NVLink 状态和 peer access 记录到 `results/env.json`。如果没有两块可互访的 GPU，脚本在此停止。
5. `litmus.py run all`：在空闲机器上运行全部测试。
6. `litmus.py run all --stress nvlink --jitter-ns 2000 --tag stress`：再次运行全部测试，同时用额外的 block 制造 NVLink 流量，并在每轮加入随机延迟。
7. `litmus.py summarize`：打印结论表并写入 `results/summary.json`。
8. 把 `results/` 打包成 `veritile-gpu-validation-results-<UTC 时间>.tar.gz`。即使某一步失败也会打包。

需要时可以单独运行：

```bash
uv run --no-sync python litmus.py list                    # 列出全部测试及其用途
uv run --no-sync python litmus.py run full_tut01 --rounds 1000000
uv run --no-sync python litmus.py summarize
```

## 第 4 步：回传

请回传 **`.tar.gz` 文件**和 `run_all.sh` 的**完整终端输出**（直接复制，或运行 `./run_all.sh 2>&1 | tee run_all.out` 后发送 `run_all.out`）。运行成功时不需要其他内容。结果中包含 GPU 型号、驱动和 PCI 总线 ID，不包含主机名或用户名。

## 测试做什么

每个测试都是 Triton-distributed tutorial 01 的 slot 协议，重复很多次：

```
PRODUCER（一块 GPU）                        CONSUMER（同一块或另一块 GPU）
等待 flag == "empty"（2k）                  等待 flag == "full"（2k+1）    <- "full" 边
写 payload = k+1（1 KiB）                   读 payload
fence；flag = "full"（2k+1）                fence；flag = "empty"（2k+2） <- "empty" 边
```

consumer 检查第 k 轮读到的 payload：

- 等于 `k+1`：正常；
- 更小（**stale**）："full" 边失效，即 message passing 违例；
- 更大（**future**）：producer 在 consumer 的读取完成之前就覆盖了 slot，即 "empty" 边上的 load buffering 违例。

默认每个测试运行 **64 对 producer/consumer × 200,000 轮 = 1280 万次迭代**（`--quick`：16 × 5,000）。每个测试只改变它研究的那几条 PTX 指令；`kernels.cu` 逐行解释了每条 PTX。每个结果文件都记录了实际使用的 PTX。

| 类别 | 含义 | 预期结果 |
|---|---|---|
| `self-test` | 故意打乱程序顺序 | **必须**出现违例，否则测试框架有问题 |
| `sanctioned` | PTX 文档禁止该坏结果 | 零违例 |
| `trust-base` | 候选 trust-base 引理（PTX 未承诺的行为） | 零违例，且其 positive control 触发 |
| `positive-control` | 比某个 trust-base 或 sanctioned 测试弱一步 | 应出现违例，说明该测试**能够**检测到违例 |
| `probe` | 仅测量 | 两种结果都有信息量 |

测试分组（`litmus.py list` 会逐个列出）：

- **"full" 边，GPU0 -> GPU1，数据在 consumer 的显存中：** `full_tut01`（tutorial 01：写方 `membar.sys; st.relaxed.sys`，读方 `ld.global.acquire.gpu`）、`full_tut01_warm`、`full_sys`（`ld.global.acquire.sys` 对照）、`full_nvshmem_wait`（`ld.volatile` + `membar.gl`，NVSHMEM 的实现）、`full_relaxed_sys_fence_gpu`、`full_nccl_volatile`、`full_writer_gpu_fence`（probe）、positive control `full_pos_*`，以及 `selftest_read_before_wait`。
- **同 GPU 对照：** `same_tut01`、`same_pos_relaxed`。
- **对 peer 显存中 flag 的远程 device scope 轮询：** `e2_remote_poll_sys_writer`、`e2_remote_poll_gpu_writer`（probe）、`e2_remote_poll_sys`、`e2_pos_relaxed`。
- **"empty" 边（load buffering）：** `empty_tut01`、`empty_sys`、`empty_pos_relaxed`、`selftest_notify_before_read`。
- **原样的 tutorial 01，两个方向同时进行：** `tut01_ring`。
- **DeepEP V1 用 `ld.global.nc.L1::no_allocate.L2::256B` 读取 peer 写入的数据：** `nc_deepep`、`nc_deepep_warm`，positive control `nc_pos_plain`、`nc_pos_plain_warm`。

## 如何阅读结果

`results/summary.txt` 中每个测试、每一轮（`-` = 空闲，`stress` = 加压）各占一行：

| 结论 | 含义 |
|---|---|
| `OK (detects)` | self-test 触发：测试框架能统计违例 |
| `HARNESS-BROKEN` | 某个 self-test 没有触发：**本次运行的所有结论都不可信** |
| `PASS` | sanctioned 测试：零违例 |
| `PASS (evidence)` | trust-base 测试：零违例、零超时，且列出的每个 positive control 在同一轮中都触发了 |
| `INCONCLUSIVE (control silent or missing)` | 零违例，但某个 positive control 从未触发，所以该测试尚未证明自己能检测到这种失效 |
| `FAIL (violations)` | 观察到被禁止的结果。JSON 中的 `first_bad` 给出示例 |
| `INCOMPLETE (timeouts)` | 部分等待超时（默认 2^24 次轮询）。可能是 liveness 问题，或 GPU 不空闲 |
| `FIRED` / `SILENT` | positive control 出现 / 未出现违例 |
| `NONE-SEEN` | probe：零违例，且其对照触发了 |

每个 `results/<test>[__stress].json` 包含：测试名、类别、所服务的引理或 ticket、每个旋钮的精确 PTX、计划与完成的迭代次数、结果直方图（`ok`、`stale`、`future`、`mixed`）、`bad_count`、超时次数、最早的若干坏轮次、stress 与 jitter 设置、耗时、GPU 与驱动信息、UTC 时间戳，以及本包的 git commit。

## 接纳 trust-base 引理的通过/失败规则

只有当服务于某个引理的每个测试都满足以下**全部**条件时，才能依据本包接纳该引理：

1. 同一次运行中所有 self-test 均为 `OK (detects)`。
2. 所有 sanctioned 测试均为 `PASS`。sanctioned 测试出现违例说明测试框架或机器有问题，必须先排查，之后才能接纳任何引理。
3. 该测试在空闲轮和 stress 轮中**都**是 `PASS (evidence)`，规模不低于默认值。

零观察违例是**证据，不是证明**：之后的运行、换一个驱动或换一种负载仍可能出现违例。positive control 从未触发的测试是**不确定的（inconclusive）**：它没有证明自己能看到所要找的失效，因此不能支持该引理。任何 `FAIL` 都会阻止接纳该引理；依赖它的 kernel 仍报告为存在竞争。每个被接纳的引理都要记录测试名、commit、GPU、驱动、CUDA/NVRTC 版本、日期和结果文件。

## 第一阶段的已知局限

- 单进程、统一虚拟地址：producer 与 consumer 对共享缓冲区使用**同一个虚拟地址**。NVSHMEM 和 Triton-distributed 使用不同进程、不同映射，所以第一阶段不检验 PTX 的 alias/proxy 问题，也不检验跨进程的 `.sys` scope。这里的通过只覆盖单进程 `cudaMalloc` 的 peer 内存。
- 默认 memory synchronization domain；使用 `cudaMalloc` 内存，而不是 NVSHMEM symmetric heap。
- 没有检查 SASS。ptxas 可以合法地变换 weak load（它会删除结果未被使用的 load，因此预热 load 的值被汇入一个 sink）。
- positive control 可能因编译器或硬件而触发。两者都是 PTX 模型允许的，因此无论哪种情况，都说明测试框架能看到弱结果。
- `empty_pos_relaxed` 可能永远不触发：这类硬件上可能观察不到 load buffering。此时 `empty_tut01` 和 `tut01_ring` 在 "empty" 边上保持不确定。

## 出错时怎么办

请回传 `.tar.gz`（失败时也会生成），并填写以下模板：

```text
### What failed
<一行，例如 "uv sync failed" 或 "full_tut01 ERROR">

### Full command
<实际输入的完整命令，包括环境变量>

### Full stderr / terminal output
<粘贴全部输出，而不只是最后几行>

### results/env.json
<作为附件，或直接粘贴>

### nvidia-smi
<以下命令的输出：nvidia-smi ; nvidia-smi topo -m>

### uv
<以下命令的输出：uv --version ; uv tree>

### Anything unusual
<GPU 上的其他任务、容器、代理、CUDA_VISIBLE_DEVICES 等>
```

常见问题：

- `cudaErrorInsufficientDriver`：驱动版本低于所选版本的要求。请使用默认的 `cu12` 版本，或报告驱动版本。
- `cannot access GPU 1 (no P2P)`：两块 GPU 不能互访；查看 `nvidia-smi topo -m`，并用 `CUDA_VISIBLE_DEVICES` 选一对 NVLink 相连的 GPU。
- `uv sync --locked` 失败：请报告；不要修改 `uv.lock` 或 `pyproject.toml`。
- 大量 `INCOMPLETE (timeouts)`：确认没有其他任务占用 GPU 后重新运行。
