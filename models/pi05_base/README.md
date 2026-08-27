# models/pi05_base — π0.5 底座权重（model.safetensors 不在 git 里）

这个目录在仓库里只有 `config.json`。真正的权重文件 `model.safetensors`（7,233,650,408 字节 ≈ 6.8 GB）
超过 GitHub 单文件 100 MiB 的硬限制，必须另行放到这里：

| 文件 | sha256 | 来源 |
|---|---|---|
| `model.safetensors` | `9067d4cd90d9f858ce016598c9420da6af1e89745167fc9974b2d24ee10a341b` | 2026-08-26 部署包 `rokae_tokenar_deploy_20260826/models/pi05_base/` 里的同一个文件（复制或做符号链接）；它是 Physical Intelligence 公开发布的 π0.5 底座（pi05_base）转成 PyTorch safetensors 的版本 |

放好后在仓库根目录跑 `sha256sum -c SHA256SUMS --quiet` 校验（没有输出 = 全部通过）。
两种规划器（token_ar / block_ar）的每个检查点都只是 LoRA 增量，加载时都叠在这份底座上；配置文件里
`pretrained_weight_path: models/pi05_base` 指向的就是这里。
