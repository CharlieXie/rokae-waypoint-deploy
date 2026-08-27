# Rokae 双臂策略服务器 — 环境配置与启动手册

这份手册的目标：在**你的 GPU 机器**上，从零把"策略服务器"跑起来，并用随包的录像做一次对拍，
确认数字与我们这边**逐位相同**，然后再接真机。所有命令都在**本包根目录**执行。

本包 2026-08-26 在一台全新的最小虚拟环境里按本文 §2 的命令逐条验证过，随后又由**另一位同事独立复验**
（新建环境、对包零修改）：起服务器、跑参考客户端，8 项对拍指标与 `data/expected_val_ep2.json`
逐位一致，42 次重规划的每一行中间量也逐位一致。

术语：**策略服务器** = 加载模型、通过 WebSocket 接收观测、返回关节目标的进程；
**pi0.5 底座** = Physical Intelligence 的 π0.5 视觉-语言-动作基础模型（PaliGemma 主干 + 动作专家），
本包的检查点是在它之上训练的 LoRA（低秩适配）增量，所以底座权重必须在场。

---

## 0. 硬件与系统要求

| 项 | 要求 | 说明 |
|---|---|---|
| GPU | NVIDIA，显存 ≥ 10 GB（服务器实测占 7.8 GB） | 我们的验证在 RTX 5090 上做；30/40 系也可以，只要驱动够新 |
| NVIDIA 驱动 | **≥ 570**（支持 CUDA 12.8） | `nvidia-smi` 右上角的 "CUDA Version" ≥ 12.8 即可。torch 轮子自带 CUDA 运行库，**不需要**单独装 CUDA toolkit |
| 操作系统 | Linux x86_64，glibc ≥ 2.35（Ubuntu 22.04+） | 我们用的是 Ubuntu 24.04 / glibc 2.39 |
| Python | **3.11.x** | 其它版本没验证过；frozen 清单是在 3.11.16 上生成的 |
| 磁盘 | **≥ 22 GB** | **峰值出现在解压时**：tar 包 5.7 GB + 解出来的 7.1 GB + Python 环境约 7 GB ≈ 20 GB，留 2 GB 余量。确认完整性后删掉 tar 包，稳态约 14 GB。**余量要按峰值算，不是按稳态差值算** |
| 网络 | 装环境时要能访问 PyPI 与 `download.pytorch.org`（或用你们的镜像）；**运行时不需要网络**（分词器随包提供） |

---

## 0.5 收到包先验完整性（**别跳过，1 分钟**）

本包 7.1 GB，其中单个文件最大 6.8 GB。**传输被截断是最常见的故障**，而它的表现是
"服务器能起、但对拍数字全错"——排查手册会把你引向别的方向，白白浪费几小时。先花一分钟排除它：

```bash
cd <本包根目录>
sha256sum -c SHA256SUMS --quiet   # 清单覆盖包内全部文件；--quiet 只在出错时打印
```

**没有任何输出 = 全部通过**（去掉 `--quiet` 会逐行打印每个文件的 OK）。
任何一行 `FAILED` → 重新传输该文件，不要试图绕过。
（等装完环境后，`python scripts/check_env.py --full` 也会做同样的校验。）

> **解压前先看一眼磁盘。** 如果你收到的是 `.tar.gz`，解压后 **tar 包和解出来的目录会同时存在**
> （5.7 GB + 7.1 GB = 12.8 GB）。校验通过后可以先删掉 tar 包，再去建 §2 的 Python 环境（约 7 GB）。

---

## 1. 先自检：也许你已经有可用的环境

如果你机器上已经有一个 openpi / pi0.5 的 Python 环境，**先别装**，激活它然后跑：

```bash
cd <本包根目录>
export PYTHONPATH=$PWD/src:$PWD/packages/openpi-client/src
export OPENPI_DATA_HOME=$PWD/.openpi_cache      # 与 serve.sh 保持一致；见 §2 第 5 步的注意事项
python scripts/check_env.py
```

它会逐项打印 `OK` / `FAIL`：Python 版本、每个依赖包及版本（torch 2.7.1 / transformers 4.53.2 等）、
CUDA 是否可用、**transformers 覆盖层是否已打**（见 §2 第 4 步，缺了不报错但数字会错）、
包内文件是否齐全、分词器缓存是否就位。

* 全部 `OK` → 跳到 §3。
* torch / transformers 版本不同 → **不要在原环境里升降级**（会破坏你原来的项目），按 §2 新建一个。
* 只有"transformers overlay"或"tokenizer in cache"失败 → 只做 §2 的第 4、5 步。

---

## 2. 从零建环境（约 10 分钟，下载约 3 GB）

推荐用 [`uv`](https://docs.astral.sh/uv/)（快、可锁版本）；没有 uv 也可以用 `python3.11 -m venv` + `pip`，命令等价写在后面。

> **六步必须按顺序做**，尤其第 2 步要在第 3 步之前——原因见第 3 步下面的警告。

**第 1 步：建虚拟环境（Python 3.11）**
```bash
cd <本包根目录>
uv venv --python 3.11 .venv            # 没有 3.11 时 uv 会自动下载
source .venv/bin/activate
# pip 等价：python3.11 -m venv .venv && source .venv/bin/activate && pip install -U pip
```
> **pip 路径需要你自己先装好 Python 3.11**（很多机器上默认没有 `python3.11`，包括我们的验证机）。
> Ubuntu：`sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt install python3.11 python3.11-venv`。
> 我们只完整验证过 uv 路径；用 uv 的话不需要预装 Python 3.11，它会自己下载。
>
> 虚拟环境建在包根目录下最省事，但它有 7 GB。**如果你之后要把这个包再转发给别人，
> 记得排除它**（见 §5 末尾的"转发前清理"）。

**第 2 步：装 torch（必须从 cu128 索引装，否则拿到的是 CPU 版或旧 CUDA 版）**
```bash
uv pip install --index-url https://download.pytorch.org/whl/cu128 "torch==2.7.1"
# pip 等价：pip install --index-url https://download.pytorch.org/whl/cu128 "torch==2.7.1"
```

**第 3 步：装其余依赖（精确清单见 `requirements-infer.txt`；逐版本冻结见 `requirements-infer-frozen.txt`）**
```bash
uv pip install -r requirements-infer.txt
# pip 等价：pip install -r requirements-infer.txt
```
> ⚠️ **必须先做第 2 步。** `requirements-infer.txt` 里也有一行 `torch==2.7.1`：先做第 2 步的话，
> 装好的 `2.7.1+cu128` 已满足这个约束、不会被覆盖；**顺序反了就会从默认索引装到非 cu128 的 torch**，
> 然后在 RTX 50 系上撞 `no kernel image is available`。

要 100% 复现我们的版本就用冻结清单：`uv pip install -r requirements-infer-frozen.txt`（其中 torch 一行带 `+cu128`，需要
加 `--extra-index-url https://download.pytorch.org/whl/cu128`）。
（参考：独立复验用的是非冻结的 `requirements-infer.txt`，装出来只有 2 个无关紧要的包与冻结清单版本不同，
对拍数字仍然逐位相同——所以冻结清单不是必需的，只是更保险。）

**第 4 步：覆盖 transformers（关键，静默陷阱）**

模型代码依赖对 HuggingFace `transformers` 4.53.2 的 5 个文件的补丁（Gemma/PaliGemma/SigLIP 的注意力实现）。
**不打补丁不会报错，但注意力语义不对，输出数字会悄悄变错。** 每次重装 transformers 后都要重做：
```bash
SP=$(python -c "import transformers, pathlib; print(pathlib.Path(transformers.__file__).parent)")
cp -r src/openpi/models_pytorch/transformers_replace/* "$SP/"
```

**第 5 步：分词器文件放进缓存**

代码首次运行会从 Google Cloud Storage 下载 PaliGemma 的 SentencePiece 分词器；为了不依赖外网，本包在
`assets/big_vision/paligemma_tokenizer.model` 附带了同一个文件（sha256 `8986bb4f…8fc6`）。放到代码查找的缓存目录：
```bash
export OPENPI_DATA_HOME=$PWD/.openpi_cache          # 不设则默认 ~/.cache/openpi
mkdir -p $OPENPI_DATA_HOME/big_vision
cp assets/big_vision/paligemma_tokenizer.model $OPENPI_DATA_HOME/big_vision/
```
> ⚠️ **`export` 只对当前终端有效。** 后面的 §3、§4 会让你"另开一个终端"——新终端里这个变量就没了。
> `serve.sh` 内部会自己把它默认成 `<包根>/.openpi_cache`，所以**服务器照常能起**；
> 但如果你在新终端里再跑一次 `check_env.py` 而没有重新 `export`，这一项可能会报 FAIL。
> **省事的做法**：把 `export OPENPI_DATA_HOME=<包根绝对路径>/.openpi_cache` 写进你的 `~/.bashrc`，
> 或者每次新开终端都照 §3/§4 的命令块重新 `export` 一次（那两节已经带上了这一行）。

**第 6 步：自检**
```bash
export PYTHONPATH=$PWD/src:$PWD/packages/openpi-client/src
python scripts/check_env.py            # 期望最后一行 ALL OK
python scripts/check_env.py --full     # 可选：再加一遍 SHA256SUMS 校验（约 1 分钟）
```

---

## 3. 启动服务器

```bash
cd <本包根目录>; source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 ./serve.sh                       # 默认检查点 8800、端口 8000
# 等价的完整命令：
# export PYTHONPATH=$PWD/src:$PWD/packages/openpi-client/src
# export OPENPI_DATA_HOME=$PWD/.openpi_cache
# python -m openpi.waypoint.rokae_policy serve --config configs/rokae_tokenar_infer.yaml \
#        --checkpoint checkpoints/8800_vlm0.0148_ae0.0040 --port 8000
```
看到 `RokaeWaypointPolicy ready … (3x s)` 和 `server listening on 0.0.0.0:8000` 即就绪（加载约 30 s，显存约 7.8 GB）。
备选检查点：`checkpoints/4000_vlm0.0385_ae0.0071`（离线指标与 8800 在噪声内，见 README）。
可选参数：`--stall-stop-replans N`、`--max-steps`、`--max-replans`、`--execute-waypoints {1,2}`、`--terminal-stop-agree N`，含义见 `docs/17-rokae-robot-client.md` §5–6。

**配置文件里的路径是相对包根目录的**（`configs/rokae_tokenar_infer.yaml`：`models/pi05_base`、`data/dataset_statistics.json`），
所以务必在包根目录启动；`serve.sh` 会自己 `cd`。

### 3b. 启动 block_ar 服务器（2026-08-27 增量包）

```bash
cd <本包根目录>; source .venv/bin/activate          # 同一个环境，依赖没有变化
CUDA_VISIBLE_DEVICES=0 ./serve_blockar.sh              # 默认检查点 checkpoints/blockar_8800_vlm0.0338_ae0.0045、端口 8000
# 等价：CONFIG=configs/rokae_blockar_infer.yaml ./serve.sh checkpoints/blockar_8800_vlm0.0338_ae0.0045 8000
# 两种规划器同时起（两张卡）：CUDA_VISIBLE_DEVICES=1 ./serve.sh checkpoints/8800_vlm0.0148_ae0.0040 8001
```
就绪标志多一行 `checkpoint …: block planner tensors=yes, planner_mode=block_ar`，然后是 `RokaeWaypointPolicy ready: planner_mode=block_ar decode=block …`。

**配置与检查点必须是同一种架构。** 服务器启动时读检查点文件头核对：block_ar 检查点配 token_ar 配置会直接报
`ValueError: … contains block_planner.* tensors … but the config says planner_mode='token_ar'` 并退出——这是故意的：
这种配错在旧版不会报错，只是静默跳过 3 个张量，然后输出看似正常、其实毫无意义的动作。反向配错也会被拦下。

block_ar 服务器多出的行为只有一条：响应里 `plan_ends_in` 可能是整数（模型认为再走几个路径点就结束）。默认只报告、不停止；
`--terminal-stop-agree 2` 可让它在连续两次规划意见一致时停止（`docs/17` §6 有离线模拟的数字，**对比实验期间不要开**）。

备选检查点：`checkpoints/blockar_4000_vlm0.1391_ae0.0070`（第 4000 步；规划器离线更准、动作专家更差、端到端打平，取舍见 README 目录表和目录内 `ORIGIN.txt`）。
起法：`./serve_blockar.sh checkpoints/blockar_4000_vlm0.1391_ae0.0070 8000`。它**没有对拍期望数字**（§4b 的期望只对默认检查点成立）：
换用后只能靠 `sha256sum -c SHA256SUMS` 和 `check_env.py` 保证文件正确，对拍只能看"能跑通、数字与默认检查点同一量级"。
**对比实验中途不要换检查点**——换了就等于换了一个模型，之前的对照局作废（见 `docs/PROMPT_FOR_BLOCKAR_EXPERIMENT.md`）。

> **多卡机器上后台起服务器**：`setsid nohup … &` 之后 shell 里的 `$!` **不是**服务器的 PID
> （中间隔了一层瞬间退出的壳），拿它去 `kill` 会打在一个已不存在的号上、静默"成功"，
> 而真正的服务器带着约 8 GB 权重继续占着卡。正确写法：
> ```bash
> setsid nohup bash -c 'echo $$ > serve.pid; exec ./serve.sh' > serve.log 2>&1 &
> ```
> （`exec` 原地替换，所以 `$$` 从头到尾就是服务器的 PID。）收尾核查见 §6。

---

## 4. 对拍（不接真机，验证环境 + 协议）

另开一个终端（同一环境）：
```bash
cd <本包根目录>; source .venv/bin/activate
export PYTHONPATH=$PWD/src:$PWD/packages/openpi-client/src
export OPENPI_DATA_HOME=$PWD/.openpi_cache
python scripts/rokae_reference_client.py schema --host 127.0.0.1 --port 8000 --npz data/val_ep2_pepper_banana.npz
python scripts/rokae_reference_client.py run    --host 127.0.0.1 --port 8000 --npz data/val_ep2_pepper_banana.npz --out ref.json
```

`run` 结束会打印一个 summary。**用这段脚本逐键对照**（它会自动跳过两个不该比较的键）：

```bash
python - <<'EOF'
import json
a = json.load(open("ref.json"))["summary"]
b = json.load(open("data/expected_val_ep2.json"))["summary"]
bad = 0
for k in b:
    if k in ("npz", "round_trip_ms_mean"):        # 见下方说明，这两项不比较
        continue
    same = a[k] == b[k]
    bad += not same
    print(f"{k:26s} got={a[k]!s:24s} expected={b[k]!s:24s} {'SAME' if same else 'DIFF'}")
print("\n对拍通过" if not bad else f"\n对拍失败：{bad} 项不一致")
EOF
```

**8 项应当全部 SAME**（同一 GPU 型号逐位相同；不同 GPU 允许 ±0.005 rad 的浮点差）：

| 指标 | 期望值 |
|---|---|
| frames（录像总帧数） | 592 |
| replans（重规划次数） | 42 |
| reached_t（录像推进到的帧号） | 609 |
| left_joint_mae_rad / right_joint_mae_rad（左/右臂关节平均绝对误差，弧度） | 0.0329 / 0.0440 |
| grip_acc（夹爪开合一致率） | 0.939 |
| max_first_row_jump_rad（首行跳变：动作第一帧相对当前关节的最大跳跃） | 0.230 |
| max_tick_jump_rad（逐拍跳变：相邻两帧之间的最大跳跃） | 0.057 |

**两项不参与比较，不是 bug：**
* `round_trip_ms_mean`（往返耗时）与机器负载有关——我们在被训练占满的 GPU 上是 2.4 s，独占会快得多。
* `npz`（录像文件路径）记录的是**当时命令行里怎么写的路径**，属于溯源信息不是指标。
  `ref.json` 里还有一个 `server_metadata` 字段（服务器自报的检查点、协议约定等），同样是溯源信息，
  也不参与比较。

不一致 → 先看 `check_env.py` 是否全 OK（尤其覆盖层），再对照 `docs/17` §8。

**可选：再验一次 `execute_waypoints=2`**（一次执行两个路径点的减半停顿模式）
```bash
python scripts/rokae_reference_client.py run --host 127.0.0.1 --port 8000 \
       --npz data/val_ep2_pepper_banana.npz --execute-waypoints 2 --out ref_ew2.json
# 然后把上面那段比较脚本里的两个文件名换成 ref_ew2.json / data/expected_val_ep2_ew2.json
```
期望：21 次重规划、0.0415 / 0.0508、夹爪 0.880、首行跳变 0.244、逐拍跳变 **0.161**。
注意最后一项**超过了 `docs/17` §7 建议的 0.1 rad 逐拍上限**，超出点出在两段的拼接处——
真机上用 `execute_waypoints=2` 时必须在拼接处做限速插值。

### 4b. block_ar 对拍（2026-08-27 增量包）

起 block_ar 服务器（§3b）后，同样的两条命令，输出文件换名、期望文件换成 block_ar 的：

```bash
python scripts/rokae_reference_client.py schema --host 127.0.0.1 --port 8000 --npz data/val_ep2_pepper_banana.npz
python scripts/rokae_reference_client.py run    --host 127.0.0.1 --port 8000 --npz data/val_ep2_pepper_banana.npz --out ref_blockar.json
# 比较：把 §4 那段脚本里的 "ref.json" 换成 "ref_blockar.json"、"data/expected_val_ep2.json" 换成 "data/expected_val_ep2_blockar.json"
```

期望（block_ar 检查点 8800 步，同一局录像；以 `data/expected_val_ep2_blockar.json` 为准）：

| 指标 | 期望值 |
|---|---|
| frames（录像总帧数） | 592 |
| replans（重规划次数） | 38 |
| reached_t（录像推进到的帧号） | 606 |
| left_joint_mae_rad / right_joint_mae_rad（左/右臂关节平均绝对误差，弧度） | 0.0236 / 0.0258 |
| grip_acc（夹爪开合一致率） | 0.943 |
| max_first_row_jump_rad（首行跳变：动作第一帧相对当前关节的最大跳跃） | 0.110 |
| max_tick_jump_rad（逐拍跳变：相邻两帧之间的最大跳跃） | 0.025 |

可选：`--execute-waypoints 2` 对照 `data/expected_val_ep2_blockar_ew2.json`，期望 19 次重规划、0.0344 / 0.0381、夹爪 0.908、首行跳变 0.063、逐拍跳变 **0.074**。

`schema` 输出里 `metadata["planner_mode"]` 应为 `block_ar`；响应多一个键 `plan_ends_in`。往返耗时会明显短于 token_ar（规划一次约 0.2–0.3 s）。

---

## 5. 常见问题

| 现象 | 处理 |
|---|---|
| `ModuleNotFoundError: openpi` / `Unknown robot type: rokae` | 没设 `PYTHONPATH`（§2 第 6 步） |
| `torch.cuda.is_available()` 为 False | 驱动 < 570，或装成了 CPU 版 torch（重做 §2 第 2 步） |
| `no kernel image is available for execution on the device` | torch 的 CUDA 版本不支持你的 GPU 架构；50 系必须 cu128。常见原因是 §2 第 2、3 步顺序反了 |
| 启动时尝试联网下载 `paligemma_tokenizer.model` | §2 第 5 步没做，或 `OPENPI_DATA_HOME` 指到了别处 |
| `check_env.py` 说 `tokenizer in cache` FAIL，但服务器能正常起 | 新终端里 `OPENPI_DATA_HOME` 丢了（§2 第 5 步的警告）。按 §1/§4 的命令块重新 `export` 即可 |
| 对拍数字不一致但都能跑 | 依次查：① `sha256sum -c SHA256SUMS`（传输截断）；② 覆盖层没打（§2 第 4 步）；③ `check_env.py` 其它 FAIL 项 |
| 对拍时 `npz` 那一行显示 DIFF | 正常，它是溯源信息不是指标（§4）。用 §4 那段比较脚本就会自动跳过 |
| 第二局刚开始就 `done=True, step_budget` | 客户端每局第一次请求没带 `reset: true`（`docs/17` §1） |
| 服务器启动即报 `ValueError: … block_planner.* tensors … planner_mode` | 配置与检查点架构不配。block_ar 检查点要配 `configs/rokae_blockar_infer.yaml`（`serve_blockar.sh` 已配好），token_ar 检查点配 `configs/rokae_tokenar_infer.yaml`（`serve.sh`） |
| `check_env.py` 报 `matching block_ar checkpoint(s): NONE` | `checkpoints/blockar_8800_vlm0.0338_ae0.0045/` 没复制进来（增量包叠加不完整），重新 `bash apply_delta.sh` |
| block_ar 对拍时 `plan_ends_in` 有整数、`done` 一直 False | 正常：模型的结束信号默认只报告不停止（`docs/17` §6） |
| 往返 > 3 秒 | 常见原因是这张 GPU 同时被别的任务占用；看响应里的 `planner_ms` / `ae_ms` 区分是模型慢还是网络慢 |

**转发前清理。** 跑过一遍之后，包目录里会多出这些**运行产物**，把包再转给别人之前请排除掉：

```bash
tar czf rokae_deploy.tar.gz \
    --exclude='.venv' --exclude='.openpi_cache' --exclude='__pycache__' \
    --exclude='ref*.json' --exclude='*.log' --exclude='serve.pid' \
    rokae_tokenar_deploy_20260826/
```
（`.venv` 有 7 GB 且绑死在你的机器路径上，转发出去对别人没有任何用处。）

---

## 6. 收尾：停服务器 + 退场核查

在共享的多卡机器上尤其重要——**`kill` 返回 0 不等于进程没了，更不等于显存还回来了**。
按顺序做完这四步再离开：

```bash
PID=$(cat serve.pid)                     # §3 那种启动方式记下来的真实 PID

# 1. 杀之前先确认这个 PID 确实是你的服务器（命令行 + 工作目录双条件）
tr '\0' ' ' < /proc/$PID/cmdline; echo; readlink /proc/$PID/cwd

# 2. 停，并轮询到它真的消失（不要只看 kill 的返回值）
kill $PID
for i in $(seq 1 30); do kill -0 $PID 2>/dev/null || { echo "已退出"; break; }; sleep 1; done

# 3. 确认显存已经还回来（应该看不到你那约 7.8 GB 了）
nvidia-smi --query-compute-apps=pid,used_memory --format=csv

# 4. 确认没有残留（不要用 pgrep -f，它会匹配到你自己这条命令）
ps -eo pid,args | grep -E '^ *[0-9]+ [^ ]*python[0-9.]* -m openpi\.waypoint\.rokae_policy'
```

第 4 步用 `ps -eo args` 并**锚定行首**、要求命令行以 python 解释器开头，是为了区分
"**是**这个程序"和"命令行里**提到**了这个程序"——任何 shell、`grep`、监控脚本的命令行里都可能
出现 `rokae_policy` 这个词，`pgrep -f rokae_policy` 会把它们（以及执行这条检查的 shell 自己）全都算进来。
