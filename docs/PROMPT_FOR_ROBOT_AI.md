# 给机器人端 AI 的任务 prompt（随对接包一起发）

你要为 Rokae AR5-5 双臂机器人写一个"策略客户端"，让它按我们提供的策略服务器的指令运动，并先在不动真机的情况下完成协议对拍。你拿到的包里有四样东西，**先完整读 `17-rokae-robot-client.md`**（自成一体，含每个字段的表格、安全检查和联调流程），再动手：

1. `17-rokae-robot-client.md` —— 对接指南（唯一需要读完的文档）。服务器可能运行两种规划器之一（token_ar / block_ar），**客户端不用区分**：协议相同，只是响应多一个信息键 `plan_ends_in`，以及规划一次的耗时不同（约 1.6 s 对约 0.3 s）。
2. `rokae_reference_client.py` —— 参考客户端，展示准确的请求/响应格式；有三个子命令：`schema`（打印字段形状）、`run`（用录像对拍）、`export`（我们用，你不用）。
3. `openpi-client/` —— 通信库（WebSocket + msgpack），`pip install -e openpi-client` 后 `from openpi_client import websocket_client_policy`；只依赖 websockets、msgpack、numpy<2、pillow、dm-tree，不需要 torch/GPU。
4. `val_ep2_pepper_banana.npz` —— 一局示教录像（592 帧；键 external/left_wrist/right_wrist/state/action/prompt）和期望数字：`expected_val_ep2.json`（token_ar 服务器）、`expected_val_ep2_blockar.json`（block_ar 服务器）——对拍时按服务器元数据里的 `planner_mode` 选对应的一份；带 `_ew2` 的是 `execute_waypoints=2` 的对应结果，首次联调不用。

你的交付物：

A. **机器人端客户端程序**：每局的第一次请求带 `reset: true`（`client.reset()` 只是本地空函数，不会通知服务器）；循环"同一时刻抓三路 RGB 图像（注意 OpenCV 是 BGR，要转 RGB；给未裁剪整幅画面）+ 30 维状态 + 任务原话 → `infer(obs)` → 安全检查 → 按 30 Hz 逐行执行 `actions`（每行 16 列：左臂 7 关节弧度、左夹爪 0/1、右臂 7 关节弧度、右夹爪 0/1）→ `done` 为真则停止"。等待服务器响应期间机器人保持上一目标不动。关节顺序、夹爪语义（0=闭 1=开，不翻号）、state 的 30 维布局都以指南第 2、3 节为准，不要猜。每次响应把 `done_reason`、`plan_ends_in`、`budget`、`planner_ms` 记进日志。

B. **安全层**（服务器不做，必须你做）：首行与当前关节差 > 0.3 rad 拒绝；相邻两拍任一关节差 > 0.1 rad 拒绝；关节限位；夹爪只接受 0/1；总时长上限与急停。首次上电把速度/加速度上限调到正常的 30%。两种规划器用**完全相同**的安全参数。

C. **对拍报告**：
   1. `python rokae_reference_client.py schema --host <我们给的地址> --port 8000 --npz val_ep2_pepper_banana.npz` 的完整输出；
   2. `python rokae_reference_client.py run … --out ref.json` 的 `ref.json`，与对应规划器的期望文件对照（重规划次数应完全相同；关节平均误差、夹爪一致率、首行/逐拍最大跳变应一致，浮点差 ±0.005 rad 以内）；
   3. **你自己的客户端**读同一个 `.npz`、走同一循环得到的同样指标（这一步证明你的打包、通道顺序、字段顺序都对）；
   4. 每次请求的往返时延（在你的网络上）。

D. 三项都过了再上真机：低速、急停就绪、先做 shelf 任务；把每段的 t、d、耗时、跳变、`done_reason`、`plan_ends_in` 记成日志发回我们。两种规划器的对比实验按 `PROMPT_FOR_BLOCKAR_EXPERIMENT.md` 第 4 节的设计与记录表来做。

规则：不要修改协议或服务器；字段含义不清就问我们，不要自行假设；任务句必须逐字用指南 2.2 节的三句之一；`waypoints` 字段只用于调试，不要直接执行；停止靠预算（指南第 6 节）和你的判断——`plan_ends_in` 是信息，不是停止指令。
