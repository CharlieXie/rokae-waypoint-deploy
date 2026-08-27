# Rokae AR5-5 双臂 · waypoint 策略服务器部署包（token_ar + block_ar，2026-08-27 版）

给机器人端同学的自包含部署包：代码 + 模型 + 配置 + 对拍数据 + 文档。不含任何训练代码路径之外的实验记录。
本版 = 2026-08-26 的 token_ar 包 + 2026-08-27 的 block_ar 增量包（`CHANGELOG.md` 列出了每一处改动；旧包的用法全部照旧）。

## 先读什么
0. **收到包先跑 `sha256sum -c SHA256SUMS --quiet`**（约 1 分钟，清单覆盖包内全部文件，没有输出就是全部通过）。
   单文件最大 6.8 GB，传输截断的表现是"能起但数字全错"，先排除掉最省事。
   （如果你收到的是**增量包**：`bash apply_delta.sh <旧包目录>` 会自动做这一步。）
1. `SETUP.md` — 环境自检 / 从零安装 / 起服务器 / 对拍（**先做这个**；§3b、§4b 是 block_ar 部分）。
2. `docs/17-rokae-robot-client.md` — 机器人端客户端对接指南（协议、字段、安全检查、联调流程；两种规划器通用）。
3. `docs/PROMPT_FOR_BLOCKAR_EXPERIMENT.md` — **这次实验的任务说明**（叠加增量包 → 对拍 → 真机上两种规划器的对比实验怎么跑、记什么、怎么汇报）；如果用 AI 助手，把它整份作为任务说明给它。
4. `docs/PROMPT_FOR_ROBOT_AI.md` — 如果用 AI 助手写客户端，把这份作为任务说明给它。
5. `docs/PROMPT_FOR_SERVER_TEST.md` — 如果用 AI 助手部署/验证**服务器端**（本包），把这份作为任务说明给它。

## 目录
| 路径 | 内容 |
|---|---|
| `src/openpi/` | 推理代码（`openpi.waypoint.rokae_policy` 是服务器入口）；`models_pytorch/transformers_replace/` 是必须覆盖进 transformers 的补丁 |
| `packages/openpi-client/` | WebSocket 客户端库（机器人端 `pip install -e packages/openpi-client`，纯 Python，不需要 torch） |
| `scripts/check_env.py` | 环境自检（`--full` 会额外校验 `SHA256SUMS`）；自动识别每个检查点的架构并与配置配对 |
| `scripts/rokae_reference_client.py` | 参考客户端：`schema` / `run` 两个子命令 |
| `serve.sh` | 一键起 **token_ar** 服务器（默认不变；`CONFIG=` 环境变量可换配置） |
| `serve_blockar.sh` | 一键起 **block_ar** 服务器 |
| `SHA256SUMS` | 全部文件的校验和，用 `sha256sum -c SHA256SUMS` 验（收到包先做） |
| `configs/rokae_tokenar_infer.yaml` | token_ar 推理配置（路径相对包根） |
| `configs/rokae_blockar_infer.yaml` | block_ar 推理配置（只差 `planner_mode` 与解码器；配置与检查点必须同架构，服务器启动时会核对） |
| `checkpoints/8800_vlm0.0148_ae0.0040/` | token_ar 检查点（9000 步训练的第 8800 步；LoRA 增量 97.5 MiB） |
| `checkpoints/4000_vlm0.0385_ae0.0071/` | token_ar 备选检查点（第 4000 步）。离线验证集上两者下一路径点关节误差 0.154 vs 0.155 rad，差异在噪声内 |
| `checkpoints/blockar_8800_vlm0.0338_ae0.0045/` | **block_ar 检查点**（第 8800 步；`serve_blockar.sh` 默认用它；来源与选择依据见目录内 `ORIGIN.txt`） |
| `checkpoints/blockar_4000_vlm0.1391_ae0.0070/` | block_ar **备选**检查点（第 4000 步）。离线验证集（15 局、2172 次规划）上它的**规划器**比 8800 准：下一路径点关节误差两臂平均 0.1068 vs 0.1121 rad，15 局中 13 局更好；但它的**动作专家**（把路径点展开成逐帧动作的那部分）差 30%（开环误差 0.0128 vs 0.0093 rad），三局录像的端到端重放两者打平（0.0227 vs 0.0232 rad）。过拟合最轻（验证交叉熵 5.9 vs 7.0），夹爪一致率最高。**没有对拍期望数字**，换用方法见 `SETUP.md` §3b |
| `models/pi05_base/` | π0.5 底座权重（PyTorch safetensors，6.8 GB），任一检查点都必须叠在它上面 |
| `data/dataset_statistics.json` | 训练集归一化统计（部署必须用这一份；两种规划器相同） |
| `data/val_ep2_pepper_banana.npz` | 对拍用的一局示教录像（592 帧） |
| `data/expected_val_ep2.json` / `expected_val_ep2_ew2.json` | token_ar 的对拍期望数字（`execute_waypoints` = 1 / 2） |
| `data/expected_val_ep2_blockar.json` / `expected_val_ep2_blockar_ew2.json` | block_ar 的对拍期望数字（同一局录像） |
| `assets/big_vision/paligemma_tokenizer.model` | PaliGemma 分词器（免联网） |
| `requirements-infer.txt` / `requirements-infer-frozen.txt` | 直接依赖 / 逐版本冻结清单（两种规划器相同） |
| `pyproject.toml` / `uv.lock` | 原仓库的依赖声明，仅供参考（含训练依赖，**不要**照它整套安装） |
| `CHANGELOG.md` | 2026-08-27 增量包改了什么、兼容性承诺、验证记录 |

## 两种规划器，一句话
同一批数据、同一个随机种子训练。**token_ar** 逐词写计划，规划一次约 1.6 s；**block_ar** 整个路径点一起写，规划一次约 0.2–0.3 s，
下一个路径点更准（两臂平均关节误差低 28%），会报告"任务快结束了"（`plan_ends_in`，默认只报告不停止），
但对远处路径点的预测更粗——**两种都必须每到一个路径点就重新规划**（`execute_waypoints=1`）。

## 三句任务原话、动作布局、停止条件
全部在 `docs/17-rokae-robot-client.md`。要点：夹爪 0=闭 1=开不翻号；每局第一次请求带 `reset: true`；
停止靠预算与机器人端判断（block_ar 的结束信号默认只报告，见 `docs/17` §6）；`execute_waypoints=2` 时两段拼接处要限速插值。

## 运行产物（转发前请排除）
跑过一遍之后包目录里会多出：`.venv/`（约 7 GB，且绑死在你的机器路径上）、`.openpi_cache/`、
`__pycache__/`、`ref*.json`、`serve.log`、`serve.pid`。**把这个包再转给别人之前请排除它们**，
具体命令见 `SETUP.md` §5 末尾的"转发前清理"。

## 验证状态
* 2026-08-26：token_ar 包在一台全新的最小虚拟环境里按 `SETUP.md` §2 逐条安装并跑通对拍；随后由另一位同事独立复验，零修改、8 项指标逐位相同。
* 2026-08-27：block_ar 增量包在 RTX 5090 上验证——新代码起 token_ar 服务器，`execute_waypoints` 1 / 2 的 8 项指标与 2026-08-26 的期望**逐位相同**（兼容性）；
  block_ar 服务器对拍数字见 `SETUP.md` §4b；合并后的包 `sha256sum -c SHA256SUMS` 与 `check_env.py` 全部通过；服务器单元测试 21 项通过。

## 来源
仓库 `openpi-rokae`（openpi 私有 fork，分支 `rokae/dual-arm`），训练数据为 Rokae 三任务 120 局示教（42 局训练 / 15 局验证 / 15 局测试各按任务分层）。
