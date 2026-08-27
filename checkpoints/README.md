# checkpoints — 四个 LoRA 检查点（都在 git 里，每个 lora.safetensors 约 98 MiB）

| 目录 | 规划器 | 角色 | 说明 |
|---|---|---|---|
| `8800_vlm0.0148_ae0.0040/` | token_ar | 默认（`./serve.sh`） | 9000 步训练的第 8800 步 |
| `4000_vlm0.0385_ae0.0071/` | token_ar | 备选 | 离线指标与 8800 在噪声内 |
| `blockar_8800_vlm0.0338_ae0.0045/` | block_ar | 默认（`./serve_blockar.sh`） | 与 token_ar 8800 同数据、同种子、同步数；来源与选择依据见目录内 `ORIGIN.txt` |
| `blockar_4000_vlm0.1391_ae0.0070/` | block_ar | 备选 | 规划器离线更准、动作专家更差、端到端打平；见目录内 `ORIGIN.txt` |

每个目录：`lora.safetensors`（LoRA 增量 + 规划器专用张量）+ `metadata.pt`（训练元数据）。
它们必须叠在 `models/pi05_base/` 的底座权重上才能用（见那个目录的 README）。
配置与检查点必须同一种架构：token_ar 检查点配 `configs/rokae_tokenar_infer.yaml`，block_ar 配 `configs/rokae_blockar_infer.yaml`；
配错时服务器启动会主动报错退出（读检查点文件头核对）。
目录名里的 `vlm`/`ae` 是训练时的损失值，不是评估指标。校验：仓库根目录 `sha256sum -c SHA256SUMS --quiet`。
