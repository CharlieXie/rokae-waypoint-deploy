# 任务 prompt：验证 Rokae 策略服务器部署包能否在本机跑起来

> 把这份文件整份粘进一个新对话的第一条消息即可。它同时适用于：
> （a）我们自己在另一台机器 / 另一个会话里复验；（b）机器人端同学的 AI 助手在他们的 GPU 机器上首次部署。

---

## 0. 你的任务

有一个**自包含的部署包**，里面是一套已经训练好的机器人策略模型和它的推理服务器。
你要做的是：**在这台机器上把服务器跑起来，并用包里自带的录像做一次"对拍"，确认算出来的数字与包里
给定的期望值一致**。你不需要接触真实机器人，也不需要训练任何东西。

包的位置：**本文件所在目录的上一级**（即 `docs/` 的父目录）。
如果你是被直接粘贴这段文字、手上没有这个文件，请让交给你这份任务的人告诉你包解压到了哪里。

包里有**两种规划器**（2026-08-27 起）：token_ar（`./serve.sh`，配置 `configs/rokae_tokenar_infer.yaml`）和
block_ar（`./serve_blockar.sh`，配置 `configs/rokae_blockar_infer.yaml`）。**两种都要验一遍**，一次起一个（或两张卡各起一个、换端口）。

**成功的判据只有一个**：`scripts/rokae_reference_client.py run` 打印的 summary 里，
`frames`、`replans`、`reached_t`、`left_joint_mae_rad`、`right_joint_mae_rad`、`grip_acc`、
`max_first_row_jump_rad`、`max_tick_jump_rad` **八项**与期望文件的 `summary` 相同——token_ar 对
`data/expected_val_ep2.json`，block_ar 对 `data/expected_val_ep2_blockar.json`
（同型号 GPU 上应逐位相同；不同型号 GPU 允许 ±0.005 rad 的浮点差）。

summary 里另外两个键**不参与比较**，它们显示 DIFF 是正常的：
`round_trip_ms_mean`（往返耗时，与机器负载有关）、`npz`（录像文件路径，是溯源信息不是指标）。
第 4 步给的比较脚本已经自动跳过这两项。

---

## 1. 背景（够你理解每一步在干什么）

* **模型**：两阶段"路径点"策略。**规划器**（planner）看三路相机图像 + 当前关节状态 + 任务句，
  输出接下来 7 个**路径点**（waypoint = 目标关节角 + 两个夹爪开合 + 到达所需帧数）；
  **动作专家**（action expert）把"当前状态 → 第 1 个路径点"这一段展开成逐帧（30 Hz）的关节目标。
* **底座**：模型建立在 π0.5（pi zero point five，Physical Intelligence 的视觉-语言-动作基础模型）之上。
  包里的检查点只有 **LoRA（低秩适配）增量**（约 98 MB），必须叠加在包内的底座权重
  `models/pi05_base`（6.8 GB）上才能用——所以两者缺一不可。
* **两种规划器**：token_ar 一次写一个"词"（规划一次约 1.6 s），block_ar 一次写完一整个路径点（约 0.2–0.3 s）。
  同一批数据、同一个随机种子训练，检查点不同、配置不同、底座和依赖相同。**配置与检查点必须同架构**：
  服务器启动时会读检查点文件头核对，配错直接报 `ValueError` 退出（配错不报错的话输出是看似正常的垃圾动作）。
* **服务器**：一个 WebSocket + msgpack 的进程，一个请求 = 一帧观测，一个响应 = 接下来若干帧的关节目标。
* **对拍**（英文里没有对应的固定词，指"用同一份输入在两边各跑一遍、比较数字是否一致"）：
  包里带了一局真实示教录像 `data/val_ep2_pepper_banana.npz`（592 帧）和我们这边跑出的期望数字。
  参考客户端会把这局录像当作机器人：把每一帧的图像和状态发给服务器，按服务器返回的帧数推进录像，
  把模型给的动作与录像里人类示教的动作作比较。

---

## 2. 硬约束

1. **只在部署包目录里操作。** 不要改包外的任何东西，不要动这台机器上别的进程。
2. **GPU：一张卡一个推理进程。** 启动前先 `nvidia-smi` 看清楚哪张卡空闲；服务器约占 **7.8 GB** 显存。
   如果这台机器上正在跑训练或别人的任务，**先确认还有余量**，并用 `CUDA_VISIBLE_DEVICES=<空闲卡号>` 指定卡。
3. **不要杀任何不是你自己启动的进程。** 找进程时不要用 `pgrep -f`（会误伤同名命令行），
   用 `/proc/<pid>/cwd` + 命令行双条件确认；你自己启动服务器时用
   `setsid nohup bash -c 'echo $$ > /tmp/serve.pid; exec ./serve.sh ...' &` 这种写法拿到**真实 PID**
   （`setsid nohup … & ` 后面的 `$!` 拿到的不是最终进程的 PID）。
4. **测完把服务器停掉**，并确认 `nvidia-smi` 上显存已经释放。
5. **如实汇报**：失败就把**原始报错**贴出来，不要凭猜测改代码绕过去。如果你为了跑通改了包里的文件，
   必须在汇报里逐条列出改了什么、为什么。

---

## 3. 步骤

### 第 1 步：读文档 + 环境自检（**先自检，不要急着装东西**）

```bash
cd <包目录>
sha256sum -c SHA256SUMS --quiet   # 先验完整性：覆盖包内全部文件，没有输出就是全部通过（约 1 分钟）
cat README.md      # 目录说明
cat SETUP.md       # 环境配置手册，后面每一步都以它为准
```

**`SHA256SUMS` 这一步别跳过。** 这个包 7.1 GB、单文件最大 6.8 GB，传输截断的表现是
"服务器能起、但对拍数字全错"——那会把你引向排查覆盖层，白白浪费很久。

如果这台机器上**已经有**一个装了 PyTorch 的 Python 环境（尤其是已经跑过 openpi / π0.5 的环境），
先激活它，然后：

```bash
export PYTHONPATH=$PWD/src:$PWD/packages/openpi-client/src
export OPENPI_DATA_HOME=$PWD/.openpi_cache     # 与 serve.sh 一致；不设的话下面这项可能误报 FAIL
python scripts/check_env.py
```

它会逐行打印 `OK` / `FAIL`：Python 版本、每个依赖包及版本、CUDA 是否可用、
**transformers 覆盖层是否已打**、包内文件是否齐全、分词器缓存是否就位。

* 全部 OK → 直接跳到第 3 步。
* 版本对不上 → **不要在现有环境里升级/降级**（会破坏这台机器上原有的项目），按 `SETUP.md` §2 新建一个虚拟环境。
* 只有 "transformers overlay" 或 "tokenizer in cache" 失败 → 只补 `SETUP.md` §2 的第 4、5 步。

### 第 2 步：按 SETUP.md §2 建环境（只在需要时做）

严格照 `SETUP.md` 的命令执行，**不要自己换版本**。其中最容易被跳过、且**出错时不报错**的一步是：

> **第 4 步：把 `src/openpi/models_pytorch/transformers_replace/` 覆盖进 site-packages 的 transformers。**
> 不覆盖不会报错，但注意力（attention）的语义是错的，模型输出会悄悄变错——对拍数字不一致时先查这一条。

装完再跑一次 `python scripts/check_env.py`，要求最后一行是 `ALL OK`。

### 第 3 步：启动服务器

```bash
nvidia-smi                       # 先确认哪张卡空闲、余量是否 > 10 GB
CUDA_VISIBLE_DEVICES=<空闲卡号> ./serve.sh          # token_ar：默认检查点 8800、端口 8000
# 第二种规划器（验完 token_ar 并停掉后再起，或另一张卡换端口）：
# CUDA_VISIBLE_DEVICES=<空闲卡号> ./serve_blockar.sh   # block_ar：默认检查点 blockar_8800_vlm0.0338_ae0.0045、端口 8000
```
就绪的标志是日志里出现 `RokaeWaypointPolicy ready: ...` 和 `server listening on 0.0.0.0:8000`（加载约 30 秒）；
block_ar 还会先打印一行 `checkpoint …: block planner tensors=yes, planner_mode=block_ar`。

### 第 4 步：对拍

另开一个终端（同一个 Python 环境、同一个包目录）：

```bash
export PYTHONPATH=$PWD/src:$PWD/packages/openpi-client/src
export OPENPI_DATA_HOME=$PWD/.openpi_cache
# 4.1 先看协议：打印服务器元数据、请求/响应每个字段的形状与类型、一次往返耗时
python scripts/rokae_reference_client.py schema --host 127.0.0.1 --port 8000 --npz data/val_ep2_pepper_banana.npz
# 4.2 再跑整局录像，输出 ref.json
python scripts/rokae_reference_client.py run    --host 127.0.0.1 --port 8000 --npz data/val_ep2_pepper_banana.npz --out ref.json
```

比较（把两个 JSON 的 `summary` 逐键对照，或直接用下面这段）：

```bash
python - <<'EOF'
import json
a=json.load(open("ref.json"))["summary"]; b=json.load(open("data/expected_val_ep2.json"))["summary"]
for k in b:
    if k in ("npz","round_trip_ms_mean"): continue
    print(f"{k:26s} got={a[k]!s:22s} expected={b[k]!s:22s} {'SAME' if a[k]==b[k] else 'DIFF'}")
EOF
```

期望值（token_ar 检查点 8800，`data/expected_val_ep2.json`）：`frames` 592、`replans` 42、`reached_t` 609、`left_joint_mae_rad` 0.0329、
`right_joint_mae_rad` 0.0440、`grip_acc` 0.939、`max_first_row_jump_rad` 0.230、`max_tick_jump_rad` 0.057。

block_ar 服务器：同样的两条命令，输出文件换成 `ref_blockar.json`，比较脚本里的期望文件换成 `data/expected_val_ep2_blockar.json`。
期望值（block_ar 检查点 8800 步）：

| 指标 | 期望值 |
|---|---|
| frames（录像总帧数） | 592 |
| replans（重规划次数） | 38 |
| reached_t（录像推进到的帧号） | 606 |
| left_joint_mae_rad / right_joint_mae_rad（左/右臂关节平均绝对误差，弧度） | 0.0236 / 0.0258 |
| grip_acc（夹爪开合一致率） | 0.943 |
| max_first_row_jump_rad（首行跳变：动作第一帧相对当前关节的最大跳跃） | 0.110 |
| max_tick_jump_rad（逐拍跳变：相邻两帧之间的最大跳跃） | 0.025 |

此外 `schema` 的输出里，`metadata["planner_mode"]` 应为 `block_ar`，响应多一个键 `plan_ends_in`（整数或 `None`），`budget` 多三个键。

### 第 5 步：收尾

**`kill` 返回 0 不等于进程没了，更不等于显存还回来了。** 按顺序做完这四步：

```bash
PID=$(cat serve.pid)                     # 第 2 节第 3 条那种启动方式记下的真实 PID

# 1. 杀之前确认这个 PID 确实是你的服务器（命令行 + 工作目录双条件）
tr '\0' ' ' < /proc/$PID/cmdline; echo; readlink /proc/$PID/cwd

# 2. 停，并轮询到它真的消失
kill $PID
for i in $(seq 1 30); do kill -0 $PID 2>/dev/null || { echo "已退出"; break; }; sleep 1; done

# 3. 确认显存已释放（应该看不到你那约 7.8 GB 了）
nvidia-smi --query-compute-apps=pid,used_memory --format=csv

# 4. 确认没有残留——注意不要用 pgrep -f，它会匹配到你自己这条命令
ps -eo pid,args | grep -E '^ *[0-9]+ [^ ]*python[0-9.]* -m openpi\.waypoint\.rokae_policy'
```

第 4 步锚定行首、要求命令行以 python 解释器开头，是为了区分"**是**这个程序"和
"命令行里**提到**了这个程序"——任何 shell、`grep`、监控脚本都可能在命令行里出现 `rokae_policy` 这个词。

---

## 4. 已知的坑（对拍不一致时按顺序排查）

| 现象 | 原因 / 处理 |
|---|---|
| `ModuleNotFoundError: openpi` 或 `Unknown robot type: rokae` | 没设 `PYTHONPATH`（必须同时包含 `<包>/src` 和 `<包>/packages/openpi-client/src`） |
| `torch.cuda.is_available()` 为 False | NVIDIA 驱动低于 570，或装成了 CPU 版 torch（必须从 `https://download.pytorch.org/whl/cu128` 索引装 `torch==2.7.1`） |
| `no kernel image is available for execution on the device` | torch 的 CUDA 版本不支持这块 GPU 的架构；RTX 50 系必须用 cu128 轮子 |
| 启动时想联网下载 `paligemma_tokenizer.model` | `SETUP.md` §2 第 5 步没做（包里 `assets/big_vision/` 自带这个文件），或 `OPENPI_DATA_HOME` 指错了 |
| 能跑起来，但对拍数字不一致 | **十有八九是 transformers 覆盖层没打**（`SETUP.md` §2 第 4 步）。`check_env.py` 会明确指出这一项 |
| 第二局刚开始就返回 `done=True, done_reason="step_budget"` | 客户端每局的第一次请求必须带 `reset: true`；参考客户端已经这样做了（细节见 `docs/17-rokae-robot-client.md` §1） |
| 往返耗时 > 3 秒 | 正常现象之一是这张 GPU 同时被别的任务占用；看响应里的 `planner_ms` / `ae_ms` 区分是模型慢还是网络慢 |
| `check_env.py` 报 `tokenizer in cache` FAIL，但服务器能正常起 | 新开的终端里 `OPENPI_DATA_HOME` 丢了（它只对当时那个终端有效）。按第 1 步 / 第 4 步的命令块重新 `export` 即可，不是环境坏了 |
| 对拍时 `npz` 那一行显示 DIFF | 正常，它是溯源信息不是指标（见第 0 节）。用第 4 步给的比较脚本会自动跳过 |
| 数字不一致，且以上都排除了 | 回到第 1 步跑一次 `sha256sum -c SHA256SUMS`——传输截断也会表现成"能跑但数字错" |
| 服务器启动即报 `ValueError: … block_planner.* tensors … planner_mode` | 配置与检查点架构不配（block_ar 检查点配了 token_ar 配置，或反过来）。用 `serve_blockar.sh` / `serve.sh` 各自的默认组合 |
| block_ar 对拍时 `plan_ends_in` 出现整数、`done` 却一直 False | 正常：模型的结束信号默认只报告不停止（`docs/17` §6） |

---

## 5. 汇报格式（请照这个结构回复）

1. **结论**：token_ar 与 block_ar 各自对拍通过 / 不通过（八项指标逐项列出 got vs expected）。
2. **环境**：GPU 型号与驱动版本、Python 版本、是复用了已有环境还是新建的、`check_env.py` 的最后一行。
3. **耗时与资源**：模型加载耗时、服务器显存占用、`round_trip_ms_mean`。
4. **过程中遇到的问题**：每个问题贴**原始报错**，以及你是怎么解决的。
5. **你对包/文档做的任何修改**：逐条列出（理想情况是"无"）。
6. **文档缺陷**：`SETUP.md` 里有没有哪一步说得不清楚、命令跑不通、或者少了前置条件——这条对我们最有价值。
