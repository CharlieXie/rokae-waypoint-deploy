# 变更说明 · block_ar 增量包（2026-08-27）

这是叠加在 **`rokae_tokenar_deploy_20260826`**（2026-08-26 的 token_ar 部署包）之上的**增量包**：
只包含新增和改动的文件，底座权重（6.8 GB）、Python 环境、token_ar 检查点与对拍数据全部复用旧包。
用 `bash apply_delta.sh <旧包目录>` 叠加，脚本会在叠加后用本包附带的完整 `SHA256SUMS` 校验整个合并后的目录。

## 一句话

同一批数据、同一个随机种子训练的第二种规划器 **block_ar**（一次写出一整个路径点，而不是逐词写）：
规划一次约 0.2–0.3 s（token_ar 约 1.6 s），下一个路径点更准（两臂平均关节误差低 28%），
并且能报告「任务快结束了」。代价是对更远处路径点的预测更粗，所以**必须保持"每到一个路径点就重新规划"的用法**。
离线对比报告见 `docs/PROMPT_FOR_BLOCKAR_EXPERIMENT.md` 第 1 节的摘要；完整数字在我们这边的报告页。

## 兼容性承诺（已实测）

* **旧包的一切照常工作**：`./serve.sh` 默认仍是 token_ar 8800；用本包的新代码起 token_ar 服务器，
  跑旧包的对拍，`execute_waypoints` = 1 和 2 两种模式下 8 项指标与旧包的 `data/expected_val_ep2*.json`
  **逐位相同**（2026-08-27 在 RTX 5090 上验证）。
* **协议只增不改**：响应多了一个键 `plan_ends_in`，`budget` 字典多了三个键；已有键的含义、取值、`done_reason`
  的四种值都没变。旧客户端不读新键也完全正常。
* **默认行为不变**：`done` 仍只由预算触发（以及"计划第一个路径点就是终止"这种从未在验证集出现过的情况）。
  新的「终止信号一致才停」规则默认关闭，要用必须显式加 `--terminal-stop-agree N`。

## 文件清单

| 文件 | 新增 / 更新 | 说明 |
|---|---|---|
| `checkpoints/blockar_8800_vlm0.0338_ae0.0045/` | 新增 | block_ar 检查点（LoRA 增量 + metadata；`serve_blockar.sh` 默认用它），来源与选择依据见目录内 `ORIGIN.txt` |
| `checkpoints/blockar_4000_vlm0.1391_ae0.0070/` | 新增 | block_ar **备选**检查点（第 4000 步；规划器离线更准但动作专家更差、端到端打平，**没有对拍期望数字**；取舍见 README 与目录内 `ORIGIN.txt`） |
| `configs/rokae_blockar_infer.yaml` | 新增 | block_ar 推理配置（`planner_mode: block_ar`、`impl: block`），其余与 token_ar 配置相同 |
| `serve_blockar.sh` | 新增 | 一键起 block_ar 服务器（默认上面的检查点，端口 8000） |
| `serve.sh` | 更新 | 多了 `CONFIG` 环境变量；默认值不变（token_ar） |
| `src/openpi/waypoint/rokae_policy.py` | 更新 | ① 响应新增 `plan_ends_in`；② 可选的 `--terminal-stop-agree N`；③ 启动时核对检查点架构与配置是否匹配，不匹配直接拒绝启动（详见下文） |
| `scripts/check_env.py` | 更新 | 不再写死检查点文件名：自动识别每个检查点是 token_ar 还是 block_ar，并与每个配置配对检查；旧包文件全部仍通过 |
| `scripts/rokae_reference_client.py` | 更新 | 逐段日志多记 `plan_ends_in`、`planner_ms`、`ae_ms`；summary 多了 `plan_ends_in_hist`、`planner_ms_mean`、`ae_ms_mean`、`done_reason`（信息键，不参与 8 项比较；比较脚本只看期望文件里有的键，所以旧期望文件照常可用） |
| `data/expected_val_ep2_blockar.json` / `_ew2.json` | 新增 | block_ar 的对拍期望数字（同一局录像、同一台 5090） |
| `docs/17-rokae-robot-client.md` | 更新 | §3 新字段、§5 时延、§6 停止条件重写、§8 对拍数字（两种规划器） |
| `docs/PROMPT_FOR_BLOCKAR_EXPERIMENT.md` | 新增 | 给对方 AI 助手的任务说明：叠加增量包 → 自检 → 对拍 → 真机 A/B 实验怎么跑、记什么、怎么汇报 |
| `docs/PROMPT_FOR_SERVER_TEST.md` / `docs/PROMPT_FOR_ROBOT_AI.md` | 更新 | 覆盖两种规划器 |
| `README.md` / `SETUP.md` | 更新 | 目录说明与启动/对拍手册加入 block_ar 部分 |
| `SHA256SUMS` | 更新 | 覆盖合并后的**全部**文件（旧文件 + 本包文件；只有清单自身与 `DELTA_MANIFEST.txt` 不在其中） |
| `apply_delta.sh` / `DELTA_MANIFEST.txt` / `CHANGELOG.md` | 新增 | 叠加脚本、本包文件清单（含每个文件的 sha256）、本文件；叠加时三者也会一并复制进合并后的目录 |

## 代码改动的细节（`rokae_policy.py`）

1. **`plan_ends_in`（新响应键，总是存在）**：模型这次规划里，在它自己给出的"结束标记"之前还有几个真实路径点；
   计划里没有结束标记时为 `None`。token_ar 检查点在验证集上从没给过结束标记，所以它永远是 `None`；
   block_ar 会在约 9% 的规划里给出（验证集上 2172 次规划中 194 次，其中 164 次确实临近结束）。
   **默认只报告，不据此停止**——因为它的位置常常不准（164 次里位置完全对的只有 46 次）。
2. **`--terminal-stop-agree N`（新启动参数，默认 0 = 关闭）**：连续 N 次规划都带结束标记、且它们推算出的"绝对结束位置"一致、
   且这次返回的动作段已经走到那个位置时，才返回 `done=True, done_reason="terminal_plan"`。
   离线模拟（15 局验证演示 × 3 个起点 = 45 条轨迹，遇到第一次触发就停）：
   N=1 有 40% 的轨迹会提前停（13% 提前 3 个路径点以上）——**不要用 1**；
   N=2 在 11% 的轨迹上恰好停在结束处、2% 提前 1 个路径点、其余交给预算——**要开就用 2**。
   预算制停止始终生效，与本规则并联。
3. **启动守卫**：读取检查点文件头，判断它是不是 block_ar 检查点（有没有 `block_planner.*` 张量），
   与配置里的 `planner_mode` 不一致就报错退出。原因：把 block_ar 检查点配上 token_ar 配置**不会报错**，
   加载器只是静默跳过那 3 个张量，然后用逐词方式解码一个按整块训练的模型——输出看起来像动作，其实是垃圾。
4. `budget` 字典新增 `terminal_stop_agree`（当前设置）、`executed_waypoints`（本局已执行的路径点数）、
   `terminal_agree_run`（连续多少次规划对结束位置意见一致）。服务器元数据新增 `terminal_stop_agree`，
   `response_keys` 列表加入 `plan_ends_in`。
5. 服务器端 `replay` 自检的 summary 新增 `plan_ends_in_hist`（各取值出现次数）。

## 验证记录

* 单元测试：`tests/waypoint/test_rokae_policy_budget.py` 21 项通过（含结束标记计数、一致性规则、架构守卫）。
* 兼容对拍：新代码 + token_ar 8800，`execute_waypoints` 1 / 2 两种模式 8 项指标与旧期望逐位相同。
* block_ar 对拍：见 `SETUP.md` §4b 的期望表与 `data/expected_val_ep2_blockar*.json`。
* 合并后的包：`sha256sum -c SHA256SUMS` 全部通过；`scripts/check_env.py` 全部 OK。
