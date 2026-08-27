# 任务 prompt：叠加 block_ar 增量包，并在真机上做 token_ar 与 block_ar 的对比实验

> 把这份文件整份粘进一个新对话的第一条消息即可。适用对象：机器人端同学的 AI 助手（或同学本人）。
> 前提：你们已经有 2026-08-26 的部署包 `rokae_tokenar_deploy_20260826`，并且已经按它的 `SETUP.md`
> 跑通过一次对拍（`scripts/rokae_reference_client.py run` 的 8 项指标与期望一致）。如果还没有，先做那一步。

---

## 0. 你的任务（分三段，按顺序）

1. **叠加增量包**：把本增量包叠到旧包目录上，校验完整性，跑环境自检。
2. **起第二种规划器的服务器并对拍**：用 `serve_blockar.sh` 起 block_ar 服务器，用旧包里同一局录像对拍，
   确认 8 项指标与本包给的 `data/expected_val_ep2_blockar.json` 一致（同型号 GPU 逐位相同；不同型号允许 ±0.005 rad）。
3. **真机对比实验**：同样的任务、同样的摆放，两种规划器交替各跑若干局，按第 4 节记录，按第 6 节汇报。

你不需要训练任何东西，也不需要改代码。**如果为了跑通改了任何文件，汇报里必须逐条列出。**

---

## 1. 背景：两种规划器是什么，离线对比说了什么

* 整套策略是"两阶段"：**规划器**看三路相机 + 当前关节角 + 任务句，给出接下来 7 个**路径点**（目标关节角 +
  两个夹爪开合 + 到达帧数）；**动作专家**把"当前状态 → 第 1 个路径点"展开成逐帧（30 Hz）关节目标。
  机器人执行完一段再拍一次观测、重新规划。**每到一个路径点就重新规划**，这一点两种规划器都一样。
* 旧包里的 **token_ar**：规划器一次写一个"词"（一个路径点是 19 个词，7 个路径点要 133 步），慢但前后连贯。
* 本包里的 **block_ar**：一次写完一整个路径点的 19 个词（只要 8 步），快得多，但同一路径点内各关节是独立猜的。
  两者用**同一批数据、同一个随机种子**训练，除规划器结构外没有任何差别。
* 离线验证集（15 局没见过的演示、2172 次规划）上的对比（block_ar 第 8800 步 对 token_ar 第 8800 步）：
  * 规划一次耗时 **171 ms 对 1653 ms**（同一台空闲 RTX 5090；换一台机器绝对值会变，我们另一台单卡 5090 上 block_ar 是约 300 ms）。
    两种规划器之后都还要跑一次动作专家（约 0.4 s），所以真机上每段之间的停顿大约从 2 s 降到 0.7 s。
  * 下一个路径点的关节误差：两臂平均低 28%（右臂低 43%），15 局验证演示里 15 局都更好。
  * 远处路径点（第 3–7 个）block_ar 更粗（整条 7 点路径左臂误差高 25%）。**所以必须保持"每段重规划"，不要用一次执行
    多个路径点的模式做正式对比**（`execute_waypoints=2` 只用于对拍验证，不用于真机对比）。
  * block_ar 会在约 9% 的规划里给出"任务快结束了"的标记（响应里的 `plan_ends_in`），token_ar 从不给。这个标记位置常常不准，
    所以**默认只报告不停止**，停止仍靠预算（见第 3 步的说明）。
  * 两者的执行层（动作专家）误差都在 0.5° 级，不是瓶颈。
  * 这些都是离线数字，**真机能不能用、哪个更好，只有你们这次实验能回答**。

---

## 2. 硬约束

1. **只在部署包目录里操作**，不要动机器人端别的东西；不要修改协议或服务器代码。
2. **一张卡一个推理进程**，每个服务器约 7.8 GB 显存。两种规划器如果要同时起（推荐，省去反复加载的 40 s），
   用两张卡、两个端口；只有一张卡就一次起一个。
3. **不要杀不是你自己启动的进程**；找进程用 `/proc/<pid>/cwd` + 命令行双条件，不要用 `pgrep -f`。
4. **真机安全层由机器人端负责**（旧包 `docs/17` §7 的五条：首行跳变 > 0.3 rad 拒绝、逐拍跳变 > 0.1 rad 拒绝、关节限位、
   夹爪只收 0/1、急停与总时长上限）。两种规划器的安全检查**必须完全一样**，否则对比没有意义。
5. **对比实验期间不要开 `--terminal-stop-agree`**（保持默认 0）：两种规划器都靠预算与人工判断停止，条件才对等。
   `plan_ends_in` 只记录，不据此操作。
6. **如实汇报**：失败贴原始报错；不要凭猜测绕过。

---

## 3. 步骤

### 第 1 步：叠加增量包并校验

```bash
cd <本增量包目录>
bash apply_delta.sh <旧包目录>          # 复制文件 + 用本包的 SHA256SUMS 校验合并后的整个目录（约 1 分钟）
cd <旧包目录>; source .venv/bin/activate # 旧包的 Python 环境直接复用，依赖没有任何变化
export PYTHONPATH=$PWD/src:$PWD/packages/openpi-client/src
export OPENPI_DATA_HOME=$PWD/.openpi_cache
python scripts/check_env.py              # 期望最后一行 ALL OK
```

自检脚本这次会自动识别 `checkpoints/` 下每个检查点是 token_ar 还是 block_ar，并检查它和 `configs/` 下每个配置是否配对。
**如果它报 `matching block_ar checkpoint(s): NONE`**，说明 `checkpoints/blockar_8800_vlm0.0338_ae0.0045/` 没有复制成功。

### 第 2 步：起 block_ar 服务器

```bash
nvidia-smi                                      # 先看哪张卡空闲、余量 > 10 GB
CUDA_VISIBLE_DEVICES=<空闲卡号> ./serve_blockar.sh   # 默认检查点 checkpoints/blockar_8800_vlm0.0338_ae0.0045、端口 8000
# 等价：CONFIG=configs/rokae_blockar_infer.yaml ./serve.sh checkpoints/blockar_8800_vlm0.0338_ae0.0045 8000
# token_ar 服务器照旧：./serve.sh（默认 8800、端口 8000）；两个同时起就换端口，例如 ./serve.sh checkpoints/8800_vlm0.0148_ae0.0040 8001
```

就绪标志：日志里出现 `checkpoint …: block planner tensors=yes, planner_mode=block_ar`、
`RokaeWaypointPolicy ready: planner_mode=block_ar decode=block …` 和 `server listening on 0.0.0.0:8000`。

如果日志里是 `ValueError: checkpoint … contains block_planner.* tensors … but the config says planner_mode='token_ar'`，
说明配置和检查点配错了——这是**故意**的拒绝：配错不会报错但输出是垃圾，服务器现在会主动拦下来。用 `serve_blockar.sh` 或
`CONFIG=configs/rokae_blockar_infer.yaml` 重新起。

### 第 3 步：对拍（不动真机）

另开一个终端（同一环境、同一目录）：

```bash
export PYTHONPATH=$PWD/src:$PWD/packages/openpi-client/src
export OPENPI_DATA_HOME=$PWD/.openpi_cache
python scripts/rokae_reference_client.py schema --host 127.0.0.1 --port 8000 --npz data/val_ep2_pepper_banana.npz
python scripts/rokae_reference_client.py run    --host 127.0.0.1 --port 8000 --npz data/val_ep2_pepper_banana.npz --out ref_blockar.json
python - <<'EOF'
import json
a = json.load(open("ref_blockar.json"))["summary"]
b = json.load(open("data/expected_val_ep2_blockar.json"))["summary"]
bad = 0
for k in b:
    if k in ("npz", "round_trip_ms_mean"): continue
    same = a[k] == b[k]; bad += not same
    print(f"{k:26s} got={a[k]!s:24s} expected={b[k]!s:24s} {'SAME' if same else 'DIFF'}")
print("\n对拍通过" if not bad else f"\n对拍失败：{bad} 项不一致")
EOF
```

期望值（block_ar 8800 步，同一局录像；以 `data/expected_val_ep2_blockar.json` 为准）：

| 指标 | 期望值 |
|---|---|
| frames（录像总帧数） | 592 |
| replans（重规划次数） | 38 |
| reached_t（录像推进到的帧号） | 606 |
| left_joint_mae_rad / right_joint_mae_rad（左/右臂关节平均绝对误差，弧度） | 0.0236 / 0.0258 |
| grip_acc（夹爪开合一致率） | 0.943 |
| max_first_row_jump_rad（首行跳变：动作第一帧相对当前关节的最大跳跃） | 0.110 |
| max_tick_jump_rad（逐拍跳变：相邻两帧之间的最大跳跃） | 0.025 |

`schema` 的输出里，响应会多一个键 `plan_ends_in`（整数或 `None`），`budget` 里多三个键；元数据里 `planner_mode` 应为 `block_ar`。
这一步过了，说明环境、协议、检查点都对。（可选：`--execute-waypoints 2` 对照 `data/expected_val_ep2_blockar_ew2.json`。）

### 第 4 步：真机对比实验

**目的**：回答"哪种规划器在真机上更好"——成功率、每局耗时、停顿时长、是否需要人工干预。

**设计（成对交替）**：
* 三个任务各做 N 对（建议 N ≥ 10，最少 5）。一"对" = 同一种物体摆放下，先跑 A 再跑 B（下一对换成先 B 再 A，即 ABBA 顺序），
  两局之间把物体放回**同一位置**（拍一张摆放照片作为记录）。这样两种规划器面对的初始条件是一样的，比较才公平。
* 服务器设置两边完全相同：`execute_waypoints=1`（默认）、`--terminal-stop-agree` 不开、预算默认。安全层参数完全相同。
* 任务句逐字用 `docs/17` §2.2 的三句原话；每局第一次请求带 `reset: true`。

**每一局记录这些字段**（写成一行 JSON 或表格一行）：

| 字段 | 怎么取 |
|---|---|
| `task` / `model` / `checkpoint` | 任务名；`token_ar` 或 `block_ar`；服务器元数据里的 `checkpoint` |
| `pair_id` / `order` | 第几对、这一局是该对里的第 1 局还是第 2 局 |
| `success` | 人工判定，0/1，判据每个任务事先写死（例如 shelf：两个方块都在指定层且手已离开） |
| `partial` | 可选：完成到哪一步（例如 pepper_banana：辣椒放对 / 香蕉放对 各记 0/1） |
| `time_s` | 从第一次请求到 `done` 或人工终止的时间 |
| `replans` / `steps_executed` | 响应 `budget` 里的 `replans`、`steps_executed`（最后一次响应的值） |
| `done_reason` | 最后一次响应的 `done_reason`；人工终止写 `manual` 并注明原因 |
| `planner_ms_mean` / `ae_ms_mean` | 各次响应 `planner_ms` / `ae_ms` 的平均（这就是停顿时长的来源） |
| `plan_ends_in_seq` | 各次响应的 `plan_ends_in` 按顺序记下来（`None` 记 `-`），例如 `- - - 3 2 1` |
| `safety_rejects` | 机器人端安全层拒绝执行的次数及原因 |
| `interventions` | 人工干预（急停、扶正物体等）次数及说明 |
| `notes` | 任何异常：抖动、撞到东西、夹爪误开合、卡住不动等 |

**怎么跑更省事**：两张卡分别常驻两个服务器（token_ar 在 8001、block_ar 在 8000），客户端只改端口；一张卡就每换模型重启一次服务器（约 40 s）。
先各跑 1 局 shelf 热身（低速、急停就绪），确认两边都能正常运动，再开始计数。

### 第 5 步：收尾

停服务器并确认显存释放（旧包 `SETUP.md` §6 的四步：用 `serve.pid` 里的真实 PID，杀之前先核对命令行与工作目录，
杀后轮询到进程消失，`nvidia-smi` 看显存还回来，最后用锚定行首的 `ps` 确认没有残留）。

---

## 4. 已知的坑

| 现象 | 原因 / 处理 |
|---|---|
| `apply_delta.sh` 报 `does not look like the 2026-08-26 package` | 路径给错了：要给旧包的根目录（里面有 `SETUP.md`、`models/pi05_base/`） |
| `sha256sum` 有 FAILED 行 | 对应文件传输不完整；重新拷贝那个文件，不要绕过 |
| 服务器启动即报 `ValueError: … block_planner.* … planner_mode='token_ar'` | 配置与检查点架构不配。用 `serve_blockar.sh`，或 `CONFIG=configs/rokae_blockar_infer.yaml` |
| 服务器报 `planner_mode='block_ar' but no block_planner.* tensor` | 反过来配错了：给 block_ar 配置指定了 token_ar 检查点 |
| `check_env.py` 报 `rokae_policy is the 2026-08-27 version` FAIL | `PYTHONPATH` 里排在前面的是旧代码（比如另一个 openpi 环境）；确认 `<包>/src` 在最前面 |
| 对拍 `plan_ends_in` 出现非 `None` 但 `done` 一直 False | 正常：默认只报告不停止 |
| `done_reason="terminal_plan"` 出现在 block_ar 上 | 只有两种情况：你开了 `--terminal-stop-agree`，或者计划的第一个路径点就是结束标记（验证集上从未发生；发生了请把那次响应的 `waypoints`、`plan_ends_in`、`budget` 记下来发我们） |
| block_ar 的往返耗时远大于 1 s | 卡被别的进程共用（看 `planner_ms` 与 `ae_ms`）；或网络。空闲 5090 上 `planner_ms` 应在 200–320 ms、`ae_ms` 约 400 ms，往返约 0.7 s |
| 默认 block_ar 检查点在真机上**系统性**异常（每局都乱动或根本不动，而不是偶发失败） | 先排除配置 / 覆盖层 / 校验问题（对拍通过就不是这些）。确认是模型本身后，可换备选 `checkpoints/blockar_4000_vlm0.1391_ae0.0070`（起法见 `SETUP.md` §3b；它没有对拍期望数字）。**换了检查点要从头重跑对照局**，并在汇报里写明用的是哪个检查点目录 |
| 其它（环境、覆盖层、分词器缓存、`reset` 漏带） | 与旧包相同，见旧包 `docs/PROMPT_FOR_SERVER_TEST.md` 第 4 节 |

---

## 5. 关于停止的说明（读一遍，实验时不用操作）

* 预算制停止没变：每个任务的 `max_steps` / `max_replans` 见 `docs/17` §6，到了就 `done=True`。
* block_ar 多了一个信号 `plan_ends_in`：模型认为再走几个路径点任务就结束。离线数据显示它**方向对、位置不准**
  （越临近越准：还剩 1 个路径点时 91% 报对，还剩 6 个时只有 27%）。所以默认只报告。
* 若以后想让它参与停止，用 `--terminal-stop-agree 2`（连续两次规划对"绝对结束位置"意见一致才停）。离线模拟：
  45 条轨迹里 11% 恰好停在结束处、2% 提前 1 个路径点、其余交给预算；`1` 会让 40% 的轨迹提前停，**不要用**。
  本次对比实验**不开**。

---

## 6. 汇报格式

1. **叠加与自检**：`apply_delta.sh` 的校验结果；`check_env.py` 最后一行。
2. **对拍**：block_ar 8 项 got vs expected（逐项）；`schema` 输出里 `planner_mode`、`checkpoint`、新键是否出现。
3. **真机实验汇总表**：按任务 × 规划器给成功局数 / 总局数、平均耗时、平均 `planner_ms`、`done_reason` 分布、干预次数；
   表头注明两边各用的检查点目录名（默认应是 token_ar `8800_vlm0.0148_ae0.0040` 与 block_ar `blockar_8800_vlm0.0338_ae0.0045`）。
4. **逐局记录**：第 4 步表格的全部行（附摆放照片编号）。
5. **定性观察**：两种规划器动作风格的差别（快慢、抖动、抓取时机、什么情况下失败）。
6. **问题与修改**：原始报错；对包/文档的任何修改；文档里说不清楚或跑不通的地方——这条对我们最有价值。
