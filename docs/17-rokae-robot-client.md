<!-- owns: rokae-robot-client -->
# 17 · Rokae 真机客户端对接指南（给机器人端的工程师 / AI 读）

本文自成一体：**只读这一份**就能把机器人端客户端写出来并完成首次联调。它描述的是我们这边已经实现好的
"策略服务器"（模型 + 网络接口），你要做的是"机器人端客户端"：抓观测 → 发给服务器 → 收到关节目标 → 按 30 Hz 执行 → 循环。
术语首次出现都展开；数字都给出处。

> **本版（2026-08-27）覆盖两种规划器**：旧包的 token_ar 与增量包新增的 block_ar。协议**只增不改**——响应多了一个键
> `plan_ends_in`、`budget` 多了三个键，其余字段与 2026-08-26 版完全相同；按旧版写好的客户端不用改一行就能对接两种规划器。

---

## 0. 一分钟概览

* **模型**：两阶段"路径点"策略。**规划器**看三路相机 + 当前关节状态 + 任务句，给出接下来 7 个**路径点**
  （waypoint：目标关节角 + 两个夹爪开合 + 到达所需帧数）；**动作专家**把"当前状态 → 第 1 个路径点"这一段
  展开成逐帧（30 Hz）的关节目标。机器人执行完这一段，再拍一次观测，重来。**每段重规划**是被验证过的协议。
* **两种规划器**（同一批数据、同一个随机种子训练，只差规划器结构）：**token_ar** 一次写一个"词"，规划一次约 1.6 s；
  **block_ar** 一次写完一整个路径点，规划一次约 0.2–0.3 s，下一个路径点更准，但对远处路径点的预测更粗——所以两种都
  **每段重规划**。服务器起哪一种由我们在启动时用配置文件决定，客户端从 `metadata["planner_mode"]` 能看到。
* **服务器**：`python -m openpi.waypoint.rokae_policy serve --config <cfg> --checkpoint <ckpt> --port 8000`
  （在我们的 GPU 机器上），WebSocket + msgpack 协议，一个请求 = 一帧观测，一个响应 = 接下来 d 帧的动作。
* **客户端**（你写的）：用 `openpi_client.websocket_client_policy.WebsocketClientPolicy` 连接；每次
  `infer(obs)` 后按行执行 `actions`；`done` 为真则停止。参考实现：`scripts/rokae_reference_client.py`。
* **先对拍再上电**：用我们提供的录像片段（`.npz`）跑一遍参考客户端和你的客户端，数字一致再动真机。

---

## 1. 通信协议

依赖：`packages/openpi-client`（纯 Python：`websockets`、`msgpack`、`numpy<2`、`pillow`、`dm-tree`），
在机器人端 `pip install -e packages/openpi-client` 即可；不需要 torch。

```python
from openpi_client import websocket_client_policy
client = websocket_client_policy.WebsocketClientPolicy(host="<服务器地址>", port=8000)
meta = client.get_server_metadata()   # 字典：动作布局、控制频率、预算、字段清单、planner_mode 等（见 §4）
obs["reset"] = True                   # 每一局的第一次请求带上：服务器清零步数 / 重规划计数并重置随机种子
out = client.infer(obs)               # obs / out 都是普通 dict，numpy 数组原样传输（msgpack 扩展）
```

* 连接是长连接；服务器同一时刻只服务一个客户端请求（同步：发一次、等一次）。
* **`client.reset()` 不会通知服务器**（上游客户端里它是空函数），所以每局开始的第一次请求必须带 `reset: true`；
  漏掉的后果是预算跨局累计，第二局会提前收到 `done_reason="step_budget"`（我们对拍时实际踩到过）。
* 跨机器访问：服务器监听 `0.0.0.0:8000`；若不在同一内网，用 SSH 隧道
  `ssh -L 8000:127.0.0.1:8000 <gpu-host>` 后连 `127.0.0.1:8000`。协议本身**没有鉴权**，不要暴露到公网。

---

## 2. 请求：观测字典 `obs`

| 键 | 类型 / 形状 | 必需 | 含义 |
|---|---|---|---|
| `external` | `uint8[H, W, 3]`，RGB | 是 | 外部（头部）相机。训练数据原始尺寸 480×640；**任意尺寸都可**，服务器用与训练完全相同的方式（PIL 双线性、不裁剪、直接压扁）缩到 224×224。请给**未裁剪**的整幅画面 |
| `left_wrist` | `uint8[H, W, 3]`，RGB | 是 | 左腕相机 |
| `right_wrist` | `uint8[H, W, 3]`，RGB | 是 | 右腕相机 |
| `state` | `float32[30]` | 是 | 机器人状态，布局见下表（与采集时的 `observation.state` 相同） |
| `prompt` | `str` | 是 | 任务句，**必须是训练用的三句原话之一**（§2.2，逐字，含标点） |
| `reset` | `bool` | 每局首次请求必带 | `true` = 开始新一局：清零预算计数、重置随机种子。其它请求不要带 |
| `execute_waypoints` | `int`，1 或 2 | 否 | 每次返回几段动作（默认 1，见 §5） |
| `max_steps` / `max_replans` | `int` | 否 | 本局预算覆盖（默认按任务，见 §6）；只在带 `reset: true` 的那次请求生效 |

**通道顺序必须是 RGB**（OpenCV 默认给 BGR，要 `[..., ::-1]`）。三张图必须是**同一时刻**抓的，与 `state` 同步。

### 2.1 `state` 的 30 维布局

| 下标 | 含义 | 策略是否使用 |
|---|---|---|
| 0–6 | 左臂 7 个关节角（弧度） | **是** |
| 7–12 | 左臂末端笛卡尔位姿 | 否（但要填真实值或 0，不能缺） |
| 13 | 左臂 psi | 否 |
| 14 | 左夹爪，0 = 闭合，1 = 张开 | **是** |
| 15–21 | 右臂 7 个关节角（弧度） | **是** |
| 22–27 | 右臂末端笛卡尔位姿 | 否 |
| 28 | 右臂 psi | 否 |
| 29 | 右夹爪，0 = 闭合，1 = 张开 | **是** |

夹爪用二值：如果驱动给的是开度，按"是否闭合"转成 0/1（训练数据里严格是 {0, 1}，判据阈值 0.5）。

### 2.2 三句任务原话（逐字复制）

```
Use the left hand to place the red block on the higher shelf, and use the right hand to place the orange block on the lower shelf.
First, pick up the red chili pepper with the left hand, then pick up the banana with the right hand. At the same time, place the chili pepper into the purple box and the banana into the yellow box.
Pick up the red pepper from the left plate first. Then pick up the green pepper from the right plate and place it on the left plate. Finally, place the red pepper on the right plate.
```

---

## 3. 响应：字典 `out`

| 键 | 类型 / 形状 | 含义 |
|---|---|---|
| `actions` | `float32[d, 16]` | 接下来 d 个控制拍（30 Hz）的目标，每行布局见下表。单位：关节弧度；夹爪列已经是 0/1 指令 |
| `duration` | `int` | d，等于 `segment_durations` 之和；`0` 表示没有动作（只会伴随 `done=True`） |
| `segment_durations` | `list[int]` | 每段的帧数（`execute_waypoints=1` 时只有一段） |
| `done` | `bool` | **执行完 `actions` 之后停止**（不是"现在立刻停"） |
| `done_reason` | `str` 或 `None` | `terminal_plan`（模型判断任务结束，见 §6）/ `step_budget` / `replan_budget` / `stalled`（见 §6） |
| `plan_ends_in` | `int` 或 `None` | **2026-08-27 新增。** 这次计划里，模型自己给出的"结束标记"之前还有几个真实路径点；`None` = 这次计划没有结束标记。token_ar 永远是 `None`；block_ar 大约 9% 的规划会给出一个整数。**默认只是信息，不影响 `done`**（§6） |
| `waypoints` | `list[[16 个 float], int]` | 整段计划：每个路径点 = 14 个关节角（弧度）+ 2 个夹爪（0/1）+ 帧数。**调试用**，不要直接执行 |
| `planner_ms` / `ae_ms` | `float` | 服务器上规划器 / 动作专家的耗时（毫秒） |
| `budget` | `dict` | `max_steps, max_replans, steps_executed, replans, stalled_replans`，以及 2026-08-27 新增的 `terminal_stop_agree`（服务器的设置，默认 0）、`executed_waypoints`（本局已执行的路径点数）、`terminal_agree_run`（连续几次规划对结束位置意见一致） |

### 3.1 `actions` 每行 16 列

| 列 | 含义 |
|---|---|
| 0–6 | 左臂关节 1–7 目标角（弧度），对应 `state[0:7]` |
| 7 | 左夹爪指令：0 = 闭合，1 = 张开 |
| 8–14 | 右臂关节 1–7 目标角（弧度），对应 `state[15:22]` |
| 15 | 右夹爪指令：0 = 闭合，1 = 张开 |

Rokae 的夹爪**不翻号**：训练数据、状态、指令三处都是 0 = 闭 / 1 = 开。

---

## 4. 控制循环（伪代码）

```python
client = WebsocketClientPolicy(host, port); meta = client.get_server_metadata()
assert meta["control_hz"] == 30                # meta["planner_mode"] 是 "token_ar" 或 "block_ar"
first = True
while True:
    obs = grab_synchronized_observation()          # 三张 RGB + state[30] + prompt
    if first:
        obs["reset"] = True; first = False         # 新一局：服务器清零预算（client.reset() 不发网络请求）
    out = client.infer(obs)                        # 阻塞：block_ar 约 0.3–0.6 s，token_ar 约 1.6–2.5 s（§5）；期间机器人保持上一目标不动
    safety_check(obs["state"], out["actions"])     # §7，不通过就停下并报警
    for row in out["actions"]:                     # 每行一个 30 Hz 控制拍（33.3 ms）
        command_joints(left=row[0:7], right=row[8:15])
        command_grippers(left=int(row[7]), right=int(row[15]))
        wait_for_next_tick()
    log(out["done_reason"], out["plan_ends_in"], out["budget"])   # 记下来，实验后要看
    if out["done"]:
        break                                      # 记录 out["done_reason"]
```

要点：
* **一整段执行完再拍下一次观测**。中途不要重新请求（请求是有状态的：服务器按每次响应的帧数累计预算）。
* 关节目标按你们控制器的方式跟踪即可（位置伺服 / 插值）；如果控制周期不是 30 Hz，对 `actions` 的行做线性插值到你的周期。
* 请求期间机器人**保持上一段的最后一个目标**，不要回零、不要松爪。
* `reset: true` 每局一次（第一次请求）；换任务句也算新一局。

---

## 5. 时延与 `execute_waypoints`

* 规划一次的耗时（`planner_ms`），同一台**空闲** RTX 5090、同一套评测连着测（2026-08-26，2172 次规划）：
  **token_ar 1653 ms、block_ar 171 ms**。换一台机器数字会变：2026-08-27 在另一台单卡 5090 机器（CPU 不同）上 block_ar 空闲实测 290–305 ms，
  是机器差异不是负载。动作专家（`ae_ms`）实测约 0.4 s（`execute_waypoints=1`）、0.8 s（=2，两段）——**比 block_ar 的规划本身还长**，
  两种规划器都要付这一份，所以 block_ar 一次往返约 0.7 s，token_ar 约 2 s。
  一段动作通常 10–30 帧（0.3–1.0 s）。所以 token_ar 是"走一段、停两秒想、再走一段"，block_ar 的停顿短得多但不是零。
* 如果 token_ar 的停顿不可接受，可在请求里带 `execute_waypoints: 2`：服务器一次返回**两段**（第二段由预测的第 1 个路径点
  开环生成，用同一帧图像），规划次数减半、停顿减半，代价是第二段没有用到新的观测（精度略降）。默认 1 是被完整验证的协议，
  **首次联调用 1；两种规划器做对比实验时也只用 1**（block_ar 对远处路径点的预测更粗，开环执行第二段会放大这一点）。
* 对拍实测（验证集 pepper_banana 第 2 局 592 帧；token_ar 行是 2026-08-26 在与训练共置的卡上测的，block_ar 行是 2026-08-27 在另一台**空闲**的单卡 5090 上测的——机器不同、负载不同，两行的往返耗时不能互相比，只能各自作为量级参考；关节误差等数字与机器无关）：

  | 规划器 | `execute_waypoints` | 重规划次数 | 往返均值 | 左 / 右臂关节平均误差（rad） | 夹爪一致率 | 首行跳变最大 | 逐拍跳变最大 |
  |---|---|---|---|---|---|---|---|
  | token_ar 8800 | 1 | 42 | 2.40 s | 0.0329 / 0.0440 | 0.939 | 0.230 rad | 0.057 rad |
  | token_ar 8800 | 2 | 21 | 2.74 s | 0.0415 / 0.0508 | 0.880 | 0.244 rad | **0.161 rad** |
  | block_ar 8800 | 1 | 38 | 0.73 s | 0.0236 / 0.0258 | 0.943 | 0.110 rad | 0.025 rad |
  | block_ar 8800 | 2 | 19 | 1.11 s | 0.0344 / 0.0381 | 0.908 | 0.063 rad | 0.074 rad |

  读法：`2` 把停顿次数减半，代价是关节误差上升、夹爪一致率下降，而且 token_ar 的**逐拍跳变最大值 0.161 rad 超过
  §7 的 0.1 rad 阈值**——它出现在两段的拼接处（第二段从"预测的第 1 个路径点"起算，而不是第一段实际到达的位置）。
  用 `2` 时机器人端必须对两段拼接处做限速插值，否则会被 §7 第 2 条拒绝。数字来自随包 `data/expected_val_ep2*.json`。

---

## 6. 停止条件（预算制是主停止；block_ar 多了一个"快结束了"的信号，默认只报告）

**预算制停止（两种规划器相同，2026-08-26 起没变）**——机器人端还必须有自己的硬停止：

| 任务 | `max_steps`（帧） | `max_replans`（次） | 来源 |
|---|---|---|---|
| shelf | 2373 | 95 | 训练示教 502–1582 帧、37–63 个路径点，取最长的 1.5 倍 |
| pepper_banana | 2013 | 150 | 示教 404–1342 帧、40–100 个路径点 |
| swap | 1749 | 138 | 示教 433–1166 帧、40–92 个路径点 |

* 达到预算时响应带 `done=True, done_reason="step_budget"/"replan_budget"`（这一段照常执行完再停）。
* `budget.stalled_replans`：连续多少次计划"原地不动"（第 1 个路径点与当前关节差 < 0.01 rad 且夹爪不变）。
  默认只报告不停止；服务器启动加 `--stall-stop-replans N` 可让它在连续 N 次原地不动后返回 `done_reason="stalled"`。
  **建议机器人端也把它当作"任务可能已完成"的信号**，配合人工判断。
* 机器人端硬停止：急停按钮、总时长上限（建议 `max_steps / 30` 秒 + 规划等待时间）、关节限位、碰撞检测。

**模型自己的"结束标记"（`plan_ends_in`，2026-08-27 新增）**：

* 规划器可以在计划的某个位置写"到此结束"。`plan_ends_in` = 结束标记前还有几个真实路径点；`None` = 没有标记。
* **token_ar 在 2172 次验证规划里一次都没写过**，所以对 token_ar 这个键永远是 `None`，停止只靠预算——与旧版行为完全相同。
* **block_ar 会写**：验证集上 194 次（约 9%），其中 164 次确实临近结束（精确率 84%）、误报 30 次；但**位置常常不准**
  （164 次里位置完全对的只有 46 次），越临近越准（还剩 1 个路径点时 91% 报对，还剩 6 个时 27%）。
  一次误报如果直接当停止用，就是任务半途停下——所以**默认只报告，不据此停止**。
* 想让它参与停止：服务器启动加 `--terminal-stop-agree N`（默认 0 = 关闭）。规则：连续 N 次规划都带结束标记、它们推算的
  "绝对结束位置"（已执行路径点数 + `plan_ends_in`）一致、且这次返回的动作段已经走到那个位置，才返回
  `done=True, done_reason="terminal_plan"`（这一段照常执行完再停）；预算制照常并联生效。
  离线模拟（15 局验证演示 × 3 个起点 = 45 条轨迹，遇到第一次触发就停）：**N=1 有 40% 的轨迹提前停（13% 提前 3 个路径点以上），不要用；
  N=2 在 11% 的轨迹恰好停在结束处、2% 提前 1 个路径点、其余交给预算——要开就用 2。**
* 另一种情况从未在验证集出现：计划的**第一个**路径点就是结束标记（`plan_ends_in = 0`）。此时没有任何动作可执行，服务器无论开关如何都返回
  `done=True, done_reason="terminal_plan", duration=0`（这是 2026-08-26 版就有的行为）。真机上如果遇到，请把那次响应记下来告诉我们。

---

## 7. 安全检查（客户端必须做，服务器不做）

服务器输出**不做**限位、限速、碰撞检查。数据依据（15 局验证集示教）：相邻两拍关节变化最大 0.051 rad
（p99.9 = 0.040 rad；折合 1.5 rad/s）；同一拍"指令 − 当前状态"最大 0.19 rad（p99 = 0.10）。建议：

1. **首行跳变**：`|actions[0, 关节列] − state[对应关节]|` 任一 > 0.3 rad → 拒绝执行本段、保持不动、报警。
2. **逐拍跳变**：相邻两行任一关节差 > 0.1 rad → 拒绝本段（或限速插值后执行，首次联调建议拒绝）。
3. **关节限位**：按机器人手册裁剪，越界即停。
4. **夹爪**：只接受 0/1；夹爪状态翻转的那一拍，关节目标照常执行。
5. **首次上电**：把控制器的速度/加速度上限调低（例如正常值的 30%），手放在急停上，先跑 shelf 任务。

两种规划器的安全检查**必须完全一样**，否则对比实验没有意义。

---

## 8. 首次联调流程（"对拍"：不动真机就能验证协议与数值）

我们随包提供：`scripts/rokae_reference_client.py`（参考客户端）、一局验证集录像 `val_ep2_pepper_banana.npz`
（592 帧：三路 224×224 图像、`state[592,30]`、`action[592,30]`、`prompt`）、以及我们这边跑出的期望数字
（token_ar：`data/expected_val_ep2.json`；block_ar：`data/expected_val_ep2_blockar.json`）。

1. 我们启动服务器（指定规划器与检查点），告诉你地址端口；`metadata["planner_mode"]` 与 `metadata["checkpoint"]` 会回显。
2. 你运行 `python scripts/rokae_reference_client.py schema --host … --port 8000 --npz val_ep2_pepper_banana.npz`：
   打印 `metadata`、请求与响应的每个键的形状/类型、一次往返耗时。**对照 §2/§3 表格确认无误。**
3. 你运行 `… run --host … --port 8000 --npz val_ep2_pepper_banana.npz --out ref.json`：把录像当作机器人，
   按响应的帧数推进录像、比较模型动作与录像动作。期望值：
   * token_ar 检查点 8800（`data/expected_val_ep2.json`）：42 次重规划、左臂关节平均误差 0.0329 rad、右臂 0.0440 rad、
     夹爪一致率 0.939、首行跳变最大 0.230 rad（§7 第 1 条阈值 0.3 以内）、逐拍跳变最大 0.057 rad（§7 第 2 条阈值 0.1 以内）；
     录像跑完前预算未触发，`done` 一直为 False。
   * block_ar 检查点 8800 步（`data/expected_val_ep2_blockar.json`）：38 次重规划、左臂关节平均误差 0.0236 rad、右臂 0.0258 rad、夹爪一致率 0.943、首行跳变最大 0.110 rad、逐拍跳变最大 0.025 rad。
4. 用**你的客户端**读同一个 `.npz`（`np.load`，键：`external/left_wrist/right_wrist/state/action/prompt`），
   走同样的循环，得到同样的数字。服务器在 `reset: true` 后重置随机种子，同一服务器上两次参考客户端回放**逐位相同**
   （我们实测），所以你的数字与期望值应在 ±0.005 rad 内（不同 GPU/驱动的浮点差）。不一致 = 你的打包/字段/通道顺序
   有问题（最常见：漏了 `reset: true`、BGR 没转 RGB、state 关节顺序），先修再上真机。
5. 真机：低速、急停就绪、shelf 任务、观察第一段是否朝示教方向运动；把 `ref.json` 风格的日志（每段的 t、d、
   耗时、跳变、`done_reason`、`plan_ends_in`）发给我们。两种规划器的真机对比怎么设计、记什么，见 `docs/PROMPT_FOR_BLOCKAR_EXPERIMENT.md` 第 4 节。

---

## 9. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `ValueError: missing camera images […]` | 三个相机键缺一不可，键名必须是 `external / left_wrist / right_wrist` |
| `state must have 30 dims` | `state` 必须是 30 维 float32，即使笛卡尔位姿你不用也要占位 |
| 动作和示教方向相反 / 关节乱跳 | 多半是通道顺序（BGR）、相机左右接反、或 `state` 关节顺序不对；先跑 §8 对拍 |
| 夹爪不动 / 反了 | 夹爪列是 0/1 指令，0 = 闭合；检查 `state[14]/[29]` 的语义是否也是 0 = 闭合 |
| `done` 一直是 False | 正常：默认靠预算（§6）和机器人端判断停止；block_ar 的 `plan_ends_in` 只是信息 |
| block_ar 返回了 `done_reason="terminal_plan"` | 只有两种情况：服务器开了 `--terminal-stop-agree`；或计划第一个路径点就是结束标记（验证集从未出现，请把该响应记下来） |
| 服务器启动即报 `block_planner.* … planner_mode` 不匹配 | 服务器端把检查点和配置配错了（这是故意的拒绝：配错不报错但输出是垃圾）；换用匹配的配置 |
| 第二局刚开始就 `done=True, done_reason="step_budget"` | 第一次请求没带 `reset: true`，预算从上一局累计过来了 |
| 往返 > 3 s（token_ar）/ > 1 s（block_ar） | 服务器 GPU 正被别的任务占用；或网络；看响应里的 `planner_ms`/`ae_ms` 区分 |
| `prompt` 换了句式效果变差 | 训练只见过三句原话；不要改写 |

---

## 10. 版本

* 代码：`src/openpi/waypoint/rokae_policy.py`（服务器；2026-08-27 版新增 `plan_ends_in`、`--terminal-stop-agree`、检查点架构守卫，
  其余与 2026-08-26 版相同）、`scripts/rokae_reference_client.py`（参考客户端；2026-08-27 版只多记录了 `plan_ends_in`、
  `planner_ms`、`ae_ms`，summary 多了 `plan_ends_in_hist` 等信息键，比较用的 8 项不变）、
  `packages/openpi-client`（协议库，上游 openpi 的客户端，未改动）。
* 检查点与配置由我们在启动服务器时指定；`metadata["checkpoint"]` / `metadata["planner_mode"]` 会回显。
  token_ar：`checkpoints/8800_vlm0.0148_ae0.0040` + `configs/rokae_tokenar_infer.yaml`；
  block_ar：`checkpoints/blockar_8800_vlm0.0338_ae0.0045` + `configs/rokae_blockar_infer.yaml`
  （备选 `checkpoints/blockar_4000_vlm0.1391_ae0.0070`，同一配置；规划器离线更准、动作专家更差、端到端打平，没有对拍期望数字）。
* 相关内部文档（不需要读也能对接）：`04-evaluation.md` 第八、九节（离线验证与封装）、`GOTCHAS.md` 第 8.x 条。
