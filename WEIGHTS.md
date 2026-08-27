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

## 从 GitHub 克隆后的完整流程（等价于"旧包 + 2026-08-27 增量包"）

```bash
git clone https://github.com/CharlieXie/rokae-waypoint-deploy.git
cd rokae-waypoint-deploy
OLD=<2026-08-26 旧包目录>                       # 里面有 models/pi05_base/、checkpoints/8800_…、data/val_ep2_pepper_banana.npz
mkdir -p models checkpoints
cp -r "$OLD/models/pi05_base" models/            # 6.8 GB；也可以 ln -s 做符号链接省空间
cp -r "$OLD"/checkpoints/8800_vlm0.0148_ae0.0040 "$OLD"/checkpoints/4000_vlm0.0385_ae0.0071 checkpoints/
cp "$OLD/data/val_ep2_pepper_banana.npz" data/
# block_ar 的两个检查点在 rokae_blockar_delta_20260827.tar.gz 里（增量包，另行传给你）：
tar xzf rokae_blockar_delta_20260827.tar.gz && cp -r rokae_blockar_delta_20260827/checkpoints/blockar_* checkpoints/
sha256sum -c SHA256SUMS --quiet                  # 没有输出 = 全部文件（代码 + 权重 + 录像）都对
```

然后按 `SETUP.md` §1–§2 准备 Python 环境（旧包的 `.venv` 可以直接复用：`cp -r "$OLD/.venv" .` 或者软链，依赖没有变化），
再按 `docs/PROMPT_FOR_BLOCKAR_EXPERIMENT.md` 做实验——它的"第 1 步：叠加增量包"在这条路径下已经等价完成，
从该步骤里的 `python scripts/check_env.py` 自检开始往下做即可。
