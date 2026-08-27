# 权重与录像文件（不在 git 里）

这个仓库只放代码、配置、文档、脚本和期望数字。下面这些大文件另行传输，放到仓库根目录的对应位置后，
用 `sha256sum -c SHA256SUMS --quiet`（或 `python scripts/check_env.py --full`）校验——`SHA256SUMS` 覆盖它们。

| 放到 | 内容 | 大小 |
|---|---|---|
| `models/pi05_base/model.safetensors` + `config.json` | π0.5 底座权重（两种规划器共用） | 6.8 GB |
| `checkpoints/8800_vlm0.0148_ae0.0040/` | token_ar 检查点（`lora.safetensors` + `metadata.pt`） | 98 MB |
| `checkpoints/4000_vlm0.0385_ae0.0071/` | token_ar 备选检查点 | 98 MB |
| `checkpoints/blockar_8800_vlm0.0338_ae0.0045/` | block_ar 检查点（默认；`serve_blockar.sh` 用它） | 98 MB |
| `checkpoints/blockar_4000_vlm0.1391_ae0.0070/` | block_ar 备选检查点（规划器离线更准、动作专家更差、端到端打平，见 `README.md`） | 98 MB |
| `data/val_ep2_pepper_banana.npz` | 对拍用的一局录像 | 150 MB |

拿到过 2026-08-26 部署包的人：这些文件（除 block_ar 检查点外）已经在那个包里，直接复制或做符号链接即可。
以后代码 / 文档 / 期望数字有更新，`git pull` 就行；只有新增检查点才需要再传文件。
