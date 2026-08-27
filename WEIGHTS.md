# 权重与录像文件：哪些在 git 里，哪些不在

这个仓库放代码、配置、文档、脚本、期望数字，**以及四个 LoRA 检查点**（`checkpoints/`，每个约 98 MiB，
刚好在 GitHub 单文件 100 MiB 的硬限制之内）和底座的 `models/pi05_base/config.json`。
只有两个文件太大、不在 git 里，需要另行放到对应位置，然后用 `sha256sum -c SHA256SUMS --quiet`
（或 `python scripts/check_env.py --full`）校验——`SHA256SUMS` 覆盖它们：

| 放到 | 内容 | 大小 | 从哪来 |
|---|---|---|---|
| `models/pi05_base/model.safetensors` | π0.5 底座权重（两种规划器共用；sha256 见 `models/pi05_base/README.md`） | 6.8 GB | 2026-08-26 部署包里的同一个文件 |
| `data/val_ep2_pepper_banana.npz` | 对拍用的一局录像 | 150 MB | 同上 |

在 git 里的（`git clone` / `git pull` 直接拿到）：`checkpoints/8800_vlm0.0148_ae0.0040/`、`checkpoints/4000_vlm0.0385_ae0.0071/`（token_ar 默认 / 备选）、
`checkpoints/blockar_8800_vlm0.0338_ae0.0045/`、`checkpoints/blockar_4000_vlm0.1391_ae0.0070/`（block_ar 默认 / 备选，各含 `ORIGIN.txt`），见 `checkpoints/README.md`。
以后代码 / 文档 / 期望数字 / **新检查点**有更新，`git pull` 就行。

## 从 GitHub 克隆后的完整流程（等价于"旧包 + 2026-08-27 增量包"）

```bash
git clone https://github.com/CharlieXie/rokae-waypoint-deploy.git   # 约 400 MB（含四个检查点）
cd rokae-waypoint-deploy
OLD=<2026-08-26 旧包目录>                       # 里面有 models/pi05_base/model.safetensors 和 data/val_ep2_pepper_banana.npz
cp "$OLD/models/pi05_base/model.safetensors" models/pi05_base/    # 6.8 GB；也可以 ln -s 做符号链接省空间
cp "$OLD/data/val_ep2_pepper_banana.npz" data/
sha256sum -c SHA256SUMS --quiet                  # 没有输出 = 全部文件（代码 + 权重 + 录像）都对
```

然后按 `SETUP.md` §1–§2 准备 Python 环境（旧包的 `.venv` 可以直接复用：`cp -r "$OLD/.venv" .` 或者软链，依赖没有变化），
再按 `docs/PROMPT_FOR_BLOCKAR_EXPERIMENT.md` 做实验——它的"第 1 步：叠加增量包"在这条路径下已经等价完成，
从该步骤里的 `python scripts/check_env.py` 自检开始往下做即可。
