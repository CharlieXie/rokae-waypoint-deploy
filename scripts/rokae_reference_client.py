#!/usr/bin/env python
"""Reference client for the Rokae waypoint policy server (docs/17-rokae-robot-client.md).

It shows the exact wire format the robot side must speak and lets both sides "对拍" (compare)
against a recorded episode before any real motion:

  # 1. export one recorded validation episode to a portable .npz (needs tensorflow_datasets)
  python scripts/rokae_reference_client.py export --rlds <val>/1.0.0 --episode 2 --out ep2.npz
  # 2. print the request/response schema against a running server (one call, no loop)
  python scripts/rokae_reference_client.py schema --host <server> --port 8000 --npz ep2.npz
  # 3. open-loop replay through the network, same protocol the robot will use
  python scripts/rokae_reference_client.py run --host <server> --port 8000 --npz ep2.npz [--execute-waypoints 2]

Only ``numpy`` and ``openpi_client`` (packages/openpi-client) are needed for ``schema`` / ``run``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

STATE_LAYOUT = ("left joints 0..6 (rad)", "left cartesian 7..12", "left psi 13", "left gripper 14 (0=closed,1=open)",
                "right joints 15..21 (rad)", "right cartesian 22..27", "right psi 28", "right gripper 29")
JOINT_COLS = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14]  # the 14 joint columns of a 16-col action row
ACTION16_IDX = [0, 1, 2, 3, 4, 5, 6, 14, 15, 16, 17, 18, 19, 20, 21, 29]   # 30-dim recorded action -> 16-dim policy action


def export_episode(rlds_dir: str, episode: int, out: str) -> None:
    import tensorflow as tf
    import tensorflow_datasets as tfds
    tf.config.set_visible_devices([], "GPU")
    builder = tfds.builder_from_directory(rlds_dir)
    ds = builder.as_dataset(split="train", read_config=tfds.ReadConfig(interleave_cycle_length=16, interleave_block_length=16))
    ep = None
    for i, e in enumerate(ds):
        if i == episode:
            ep = e
            break
    if ep is None:
        raise SystemExit(f"episode {episode} not found in {rlds_dir}")
    steps = list(ep["steps"])
    get = lambda key: np.stack([s["observation"][key].numpy() for s in steps])
    np.savez_compressed(out, external=get("external"), left_wrist=get("left_wrist"), right_wrist=get("right_wrist"),
                        state=np.stack([s["observation"]["state"].numpy() for s in steps]).astype(np.float32),
                        action=np.stack([s["action"].numpy() for s in steps]).astype(np.float32),
                        prompt=np.array(steps[0]["language_instruction"].numpy().decode("utf-8")), episode=np.array(episode))
    print(f"wrote {out}: {len(steps)} frames, prompt={steps[0]['language_instruction'].numpy().decode('utf-8')!r}")


def load_npz(path: str) -> dict:
    d = np.load(path)
    return {k: d[k] for k in d.files}


def obs_at(ep: dict, t: int, execute_waypoints: int | None, reset: bool = False) -> dict:
    obs = {"external": ep["external"][t], "left_wrist": ep["left_wrist"][t], "right_wrist": ep["right_wrist"][t],
           "state": ep["state"][t].astype(np.float32), "prompt": str(ep["prompt"])}
    if execute_waypoints:
        obs["execute_waypoints"] = int(execute_waypoints)
    if reset:
        obs["reset"] = True   # first request of an episode; client.reset() alone does not reach the server
    return obs


def describe(x):
    if isinstance(x, np.ndarray):
        return f"ndarray shape={x.shape} dtype={x.dtype}"
    if isinstance(x, list):
        return f"list len={len(x)}" + (f" first={describe(x[0])}" if x else "")
    return f"{type(x).__name__}={x!r}" if not isinstance(x, dict) else "{" + ", ".join(f"{k}: {describe(v)}" for k, v in x.items()) + "}"


def connect(host: str, port: int):
    from openpi_client import websocket_client_policy
    client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
    meta = client.get_server_metadata()
    return client, meta


def cmd_schema(args) -> None:
    ep = load_npz(args.npz)
    client, meta = connect(args.host, args.port)
    print("server metadata:", json.dumps(meta, indent=1, ensure_ascii=False))
    obs = obs_at(ep, 0, args.execute_waypoints, reset=True)
    print("request:", describe(obs))
    t0 = time.time(); out = client.infer(obs); rt = (time.time() - t0) * 1000
    print("response:", describe(out))
    print(f"round trip {rt:.0f} ms (planner {out.get('planner_ms', 0):.0f} + action expert {out.get('ae_ms', 0):.0f} on the server)")
    print("first action row (16):", np.round(np.asarray(out["actions"])[0], 4).tolist() if len(out["actions"]) else "(empty)")
    print("current state joints+grippers (16):", np.round(ep["state"][0][ACTION16_IDX], 4).tolist())


def cmd_run(args) -> None:
    ep = load_npz(args.npz)
    client, meta = connect(args.host, args.port)
    client.reset()
    T = len(ep["state"]); t = 0; rows = []; rts = []
    while t < T - 1 and len(rows) < args.max_replans:
        obs = obs_at(ep, t, args.execute_waypoints, reset=not rows)
        t0 = time.time(); out = client.infer(obs); rts.append((time.time() - t0) * 1000)
        d = int(out["duration"])
        if d <= 0:
            print(f"t={t}: done={out['done']} reason={out.get('done_reason')} (no actions)"); break
        acts = np.asarray(out["actions"], dtype=np.float32)
        gt = ep["action"][t:t + d][:, ACTION16_IDX]; n = min(len(gt), d)
        err = np.abs(acts[:n] - gt[:n])
        rows.append({"t": t, "d": d, "segments": list(out.get("segment_durations", [d])),
                     "left_joint_mae": float(err[:, 0:7].mean()), "right_joint_mae": float(err[:, 8:15].mean()),
                     "grip_acc": float((acts[:n][:, [7, 15]] == np.round(gt[:n][:, [7, 15]])).mean()),
                     # The two safety-check quantities of docs/17 §7, kept separate because their thresholds differ:
                     # first row vs the current joints (limit 0.3 rad) and consecutive rows (limit 0.1 rad).
                     "first_row_jump_rad": float(np.abs(acts[0][JOINT_COLS] - ep["state"][t][ACTION16_IDX][JOINT_COLS]).max()),
                     "max_tick_jump_rad": float(np.abs(np.diff(acts[:n][:, JOINT_COLS], axis=0)).max()) if n > 1 else 0.0,
                     # 2026-08-27, informational: the planner's own end marker (None = no marker; token_ar never
                     # emits one, block_ar does in ~9% of plans) and the server-side timings.  Not compared.
                     "plan_ends_in": out.get("plan_ends_in"),
                     "planner_ms": float(out.get("planner_ms", 0.0)), "ae_ms": float(out.get("ae_ms", 0.0))})
        print(f"t={t:4d} d={d:2d} seg={rows[-1]['segments']} L={rows[-1]['left_joint_mae']:.4f} R={rows[-1]['right_joint_mae']:.4f} "
              f"grip={rows[-1]['grip_acc']:.2f} jump0={rows[-1]['first_row_jump_rad']:.3f} tick={rows[-1]['max_tick_jump_rad']:.3f} "
              f"rt={rts[-1]:.0f}ms ends_in={out.get('plan_ends_in')} done={out['done']} {out.get('done_reason') or ''}")
        t += d
        if out["done"]:
            break
    summary = {"npz": args.npz, "frames": T, "replans": len(rows), "reached_t": t,
               "round_trip_ms_mean": float(np.mean(rts)) if rts else None,
               "left_joint_mae_rad": float(np.mean([r["left_joint_mae"] for r in rows])) if rows else None,
               "right_joint_mae_rad": float(np.mean([r["right_joint_mae"] for r in rows])) if rows else None,
               "grip_acc": float(np.mean([r["grip_acc"] for r in rows])) if rows else None,
               "max_first_row_jump_rad": float(max(r["first_row_jump_rad"] for r in rows)) if rows else None,
               "max_tick_jump_rad": float(max(r["max_tick_jump_rad"] for r in rows)) if rows else None,
               # informational (2026-08-27), not part of the 8-item comparison:
               "planner_ms_mean": float(np.mean([r["planner_ms"] for r in rows])) if rows else None,
               "ae_ms_mean": float(np.mean([r["ae_ms"] for r in rows])) if rows else None,
               "plan_ends_in_hist": {k: sum(1 for r in rows if ("none" if r["plan_ends_in"] is None else str(r["plan_ends_in"])) == k)
                                     for k in sorted({"none" if r["plan_ends_in"] is None else str(r["plan_ends_in"]) for r in rows})},
               "done_reason": (out.get("done_reason") if rows else None)}
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "rows": rows, "server_metadata": meta}, f, indent=1, ensure_ascii=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export"); e.add_argument("--rlds", required=True); e.add_argument("--episode", type=int, default=0); e.add_argument("--out", required=True)
    for name in ("schema", "run"):
        p = sub.add_parser(name)
        p.add_argument("--host", default="127.0.0.1"); p.add_argument("--port", type=int, default=8000)
        p.add_argument("--npz", required=True); p.add_argument("--execute-waypoints", type=int, default=None, choices=(1, 2))
    sub.choices["run"].add_argument("--max-replans", type=int, default=300)
    sub.choices["run"].add_argument("--out", default=None)
    args = ap.parse_args()
    if args.cmd == "export":
        export_episode(args.rlds, args.episode, args.out)
    elif args.cmd == "schema":
        cmd_schema(args)
    else:
        cmd_run(args)


if __name__ == "__main__":
    main()
