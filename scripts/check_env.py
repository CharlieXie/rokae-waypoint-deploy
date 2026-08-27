#!/usr/bin/env python
"""Environment self-check for the Rokae policy server.  Run it BEFORE installing anything:
if every line says OK you already have a usable environment (e.g. an existing openpi / pi0.5 env)
and can skip SETUP.md §2.  Run from the package root:

    python scripts/check_env.py            fast checks only (a few seconds)
    python scripts/check_env.py --full     ALSO verify SHA256SUMS (~7 GB, about a minute)

Run --full once after you receive/copy the package: it is the only check that catches a
truncated or corrupted transfer.  Everything else only checks that files *exist*.

2026-08-27 (block_ar delta): the checkpoint list is no longer hard-coded.  Every directory
under checkpoints/ is inspected (token_ar or block_ar, read from the safetensors header) and
matched against every config under configs/, so a config without a usable checkpoint -- or a
checkpoint that no config can load -- is reported.  The 2026-08-26 token_ar files still pass.
"""
import argparse, importlib, json, os, pathlib, struct, sys, hashlib

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--full", action="store_true", help="also verify every entry of SHA256SUMS (slow)")
args = ap.parse_args()

ROOT = pathlib.Path(__file__).resolve().parents[1]
ok = True
def report(flag, msg):
    global ok
    ok &= bool(flag)
    print(("OK   " if flag else "FAIL ") + msg)

# 1. python
v = sys.version_info
report(v[:2] == (3, 11), f"Python {v.major}.{v.minor}.{v.micro} (need 3.11.x; the frozen list was built on 3.11.16)")

# 2. packages + versions
want = {"torch": "2.7.1", "transformers": "4.53.2", "numpy": None, "safetensors": None, "PIL": None, "yaml": None,
        "websockets": None, "msgpack": None, "tree": None, "einops": None, "beartype": "0.19.0", "jaxtyping": "0.2.36",
        "imageio": None, "jax": "0.5.3", "flax": "0.10.2", "augmax": None, "sentencepiece": None,
        "tqdm_loggable": None, "fsspec": None, "filelock": None}
for mod, ver in want.items():
    try:
        m = importlib.import_module(mod)
        got = getattr(m, "__version__", "?")
        if ver is None:
            report(True, f"{mod} {got}")
        else:
            report(str(got).startswith(ver), f"{mod} {got} (need {ver})")
    except Exception as e:  # noqa: BLE001
        report(False, f"{mod}: import failed ({e.__class__.__name__}: {e})")
try:
    import numpy as np
    report(int(np.__version__.split(".")[0]) < 2, f"numpy < 2 (got {np.__version__})")
except Exception:
    pass

# 3. CUDA
try:
    import torch
    report(torch.cuda.is_available(), f"torch.cuda.is_available() (torch {torch.__version__}, built for CUDA {torch.version.cuda})")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        # torch's device 0 is the FIRST VISIBLE device, which is not physical GPU 0 when
        # CUDA_VISIBLE_DEVICES is set -- say so, or this line misleads on a shared machine.
        vis = os.environ.get("CUDA_VISIBLE_DEVICES")
        which = f"torch device 0 (= physical GPU {vis.split(',')[0]}, CUDA_VISIBLE_DEVICES={vis})" if vis else "torch device 0 (CUDA_VISIBLE_DEVICES unset -> physical GPU 0)"
        report(p.total_memory / 2**30 >= 10, f"{which}: {p.name}, {p.total_memory / 2**30:.1f} GB (server needs ~8 GB)")
        report("cu12" in torch.__version__ or torch.version.cuda.startswith("12."), "torch is a CUDA 12.x build")
except Exception as e:  # noqa: BLE001
    report(False, f"torch/cuda check failed: {e}")

# 4. transformers overlay (silent-failure trap: wrong attention semantics if missing)
try:
    import transformers
    sp = pathlib.Path(transformers.__file__).parent
    overlay = ROOT / "src/openpi/models_pytorch/transformers_replace"
    same = True; missing = []
    for f in overlay.rglob("*.py"):
        rel = f.relative_to(overlay); tgt = sp / rel
        if not tgt.exists() or hashlib.sha256(tgt.read_bytes()).digest() != hashlib.sha256(f.read_bytes()).digest():
            same = False; missing.append(str(rel))
    report(same, "transformers overlay applied (src/openpi/models_pytorch/transformers_replace -> site-packages/transformers)"
           + ("" if same else f"; differs/missing: {missing}"))
except Exception as e:  # noqa: BLE001
    report(False, f"overlay check failed: {e}")

# 5. package files that every configuration needs
for rel, desc in [("models/pi05_base/model.safetensors", "PaliGemma base weights (pi0.5 base, ~6.8 GB)"),
                  ("models/pi05_base/config.json", "base config"),
                  ("data/dataset_statistics.json", "TRAIN-split normalization statistics"),
                  ("data/val_ep2_pepper_banana.npz", "recorded episode for the 对拍 comparison (592 frames)"),
                  ("assets/big_vision/paligemma_tokenizer.model", "PaliGemma SentencePiece tokenizer"),
                  ("serve.sh", "server launcher (token_ar defaults)")]:
    report((ROOT / rel).exists(), f"{rel}  ({desc})")

# 5b. checkpoints x configs: architecture is read from the safetensors header, not from file names.
#     A block_ar checkpoint has block_planner.* tensors; loading it with a token_ar config would
#     silently drop them and decode garbage, so the server refuses the mismatch -- check it here first.
def ckpt_arch(d: pathlib.Path):
    for name in ("lora.safetensors", "model.safetensors"):
        f = d / name
        if f.exists():
            with open(f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                keys = json.loads(fh.read(n))
            return "block_ar" if any(k.startswith("block_planner.") for k in keys if k != "__metadata__") else "token_ar"
    return None

ckpts = {}
for d in sorted((ROOT / "checkpoints").glob("*")) if (ROOT / "checkpoints").exists() else []:
    if d.is_dir():
        arch = ckpt_arch(d)
        if arch:
            ckpts[d.name] = arch
            report((d / "metadata.pt").exists(), f"checkpoints/{d.name}  ({arch} checkpoint, LoRA delta" + ("" if (d / "metadata.pt").exists() else "; metadata.pt MISSING") + ")")
report(bool(ckpts), f"at least one checkpoint under checkpoints/ (found {len(ckpts)})")

EXPECTED = {"token_ar": ["data/expected_val_ep2.json", "data/expected_val_ep2_ew2.json"],
            "block_ar": ["data/expected_val_ep2_blockar.json", "data/expected_val_ep2_blockar_ew2.json"]}
configs = sorted((ROOT / "configs").glob("*.yaml")) if (ROOT / "configs").exists() else []
report(bool(configs), f"at least one inference config under configs/ (found {len(configs)})")
try:
    import yaml
    for c in configs:
        cfg = yaml.safe_load(c.read_text()) or {}
        mode = cfg.get("planner_mode", "token_ar")
        impl = ((cfg.get("eval") or {}).get("waypoint_decode") or {}).get("impl", "compact")
        report((mode == "block_ar") == (impl == "block"),
               f"configs/{c.name}: planner_mode={mode}, decode impl={impl} (block_ar <-> block, token_ar <-> compact)")
        usable = [n for n, a in ckpts.items() if a == mode]
        report(bool(usable), f"configs/{c.name}: matching {mode} checkpoint(s): {usable or 'NONE'}")
        for rel in EXPECTED.get(mode, []):
            report((ROOT / rel).exists(), f"{rel}  (expected 对拍 numbers for {mode}; the comparison in SETUP.md needs it)")
        for key in ("dataset_statistics_path", "pretrained_weight_path"):
            p = cfg.get(key)
            report(bool(p) and (ROOT / p).exists(), f"configs/{c.name}: {key}={p} exists (relative to the package root)")
except Exception as e:  # noqa: BLE001
    report(False, f"config inspection failed: {e}")

# 6. tokenizer cache.  The model loader looks in $OPENPI_DATA_HOME/big_vision/, defaulting to
#    ~/.cache/openpi when the variable is unset.  serve.sh exports OPENPI_DATA_HOME=<root>/.openpi_cache
#    when it is unset, so which location counts depends on whether the variable is set RIGHT NOW:
#      * set   -> the loader will look ONLY there, so only that path may count (checking the others
#                 would be a false OK: the server would still try to download the tokenizer).
#      * unset -> serve.sh will use <root>/.openpi_cache; a hand-rolled `python -m openpi...` uses
#                 ~/.cache/openpi.  Either one is fine, so accept both.
#    Without this, a fresh terminal FAILs here even though the environment is correct, because
#    SETUP.md's `export OPENPI_DATA_HOME=...` only lives in the shell that ran it.
_rel = "big_vision/paligemma_tokenizer.model"
_env = os.environ.get("OPENPI_DATA_HOME")
if _env:
    _cands = [(f"$OPENPI_DATA_HOME={_env}", pathlib.Path(_env).expanduser() / _rel)]
else:
    _cands = [("<root>/.openpi_cache  (what serve.sh uses)", ROOT / ".openpi_cache" / _rel),
              ("~/.cache/openpi  (loader default)", pathlib.Path("~/.cache/openpi").expanduser() / _rel)]
_hit = next((c for c in _cands if c[1].exists()), None)
report(_hit is not None,
       f"tokenizer in cache: {_hit[0]} -> {_hit[1]}" if _hit else
       "tokenizer in cache -- not found in: " + " | ".join(str(p) for _, p in _cands) +
       "  (run SETUP.md §2 step 5; else the server tries to download it from GCS)")

# 7. PYTHONPATH
pp = os.environ.get("PYTHONPATH", "")
report(str(ROOT / "src") in pp and str(ROOT / "packages/openpi-client/src") in pp,
       "PYTHONPATH contains <root>/src and <root>/packages/openpi-client/src")
try:
    import openpi.waypoint.rokae_policy as _rp, openpi_client  # noqa: F401
    report(True, "import openpi.waypoint.rokae_policy / openpi_client")
    report(hasattr(_rp, "TerminalAgreement") and hasattr(_rp, "check_checkpoint_matches_planner_mode"),
           "rokae_policy is the 2026-08-27 version (plan_ends_in / terminal_stop_agree / architecture guard)"
           + ("" if hasattr(_rp, "TerminalAgreement") else " -- an OLDER copy of src/ is being imported first; check PYTHONPATH order"))
except Exception as e:  # noqa: BLE001
    report(False, f"import failed: {e.__class__.__name__}: {e}")

# 8. file integrity (only with --full: hashing ~7 GB takes about a minute)
sums = ROOT / "SHA256SUMS"
if args.full:
    if not sums.exists():
        report(False, "SHA256SUMS missing -- cannot verify file integrity")
    else:
        for line in sums.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            want_hex, _, rel = line.partition("  ")
            f = ROOT / rel
            if not f.exists():
                report(False, f"sha256 {rel}: MISSING"); continue
            h = hashlib.sha256()
            with open(f, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 22), b""):
                    h.update(chunk)
            report(h.hexdigest() == want_hex, f"sha256 {rel}")
elif sums.exists():
    print("INFO file integrity NOT verified.  Run `python scripts/check_env.py --full`"
          " (or `sha256sum -c SHA256SUMS --quiet`) once after copying the package -- it is the"
          " only check that catches a truncated or corrupted file, e.g. the 6.8 GB base weights.")

print("\nALL OK" if ok else "\nSOME CHECKS FAILED - see SETUP.md")
sys.exit(0 if ok else 1)
