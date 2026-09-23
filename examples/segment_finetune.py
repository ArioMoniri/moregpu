#!/usr/bin/env python3
"""JEPA encoder → segmentation fine-tune on a pool (synthetic data; no download).

  1) JEPA-pretrain a micro ViT on synthetic slabs (DiLoCo across the pool's torch workers)
  2) export the encoder on a worker
  3) fine-tune a light decoder on top (segment task, Dice+CE, DiLoCo) and export the whole model

  python3 examples/segment_finetune.py --url http://localhost:8787 --token $MOREGPU_ADMIN_TOKEN --out /tmp/mgpu-seg
(`--out` is a directory ON THE WORKERS; on a single machine all workers share it.)"""
import argparse, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "clients", "python"))
from moregpu import MoreGPU  # noqa: E402

SYN = {"kind": "2p5d", "n": 32, "size": [32, 32], "channels": 3, "seed": 0}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=os.environ.get("MOREGPU_URL", "http://localhost:8787"))
    p.add_argument("--token", default=os.environ.get("MOREGPU_ADMIN_TOKEN", ""))
    p.add_argument("--out", required=True)
    p.add_argument("--pretrain-rounds", type=int, default=4)
    p.add_argument("--finetune-rounds", type=int, default=12)
    a = p.parse_args(argv)
    pool = MoreGPU(a.url, a.token)
    j = pool.train_jepa("jepa_2p5d", synthetic={**SYN, "classes": 3}, model="micro", patch=8, manifest_len=SYN["n"],
                        batch=8, inner_steps=3, lr=2e-3, amp="fp32",
                        jepa={"pred_dim": 16, "pred_depth": 1, "pred_heads": 2, "n_targets": 2, "ema": [0.99, 1.0],
                              "total_steps": a.pretrain_rounds * 3, "weight_decay": 0.0})
    jid = j["session"]
    for _ in range(a.pretrain_rounds):                       # one round per call: keeps each HTTP request short
        pool.train_session_round(jid, 1)
    enc_dir = os.path.join(a.out, "encoder")
    pool.train_session_export(jid, "safetensors", enc_dir)
    pool.train_session_delete(jid)
    cfg = {"kind": "2p5d", "num_classes": 3, "encoder": {"init": "export", "path": enc_dir}, "decoder": {"channels": [16, 8]},
           "synthetic": SYN, "mode": "full", "weight_decay": 0.0}
    s = pool.train_session_create("segment", cfg, manifest_len=SYN["n"], batch=4, inner_steps=3, lr=3e-3, amp="fp32",
                                  eval={"refs": list(range(16)), "kind": "dice", "every": 4})
    sid = s["session"]
    d0 = pool.train_session_eval(sid, list(range(16)), "dice")
    hist = [pool.train_session_round(sid, 1)["rounds"][-1] for _ in range(a.finetune_rounds)]
    d1 = pool.train_session_eval(sid, list(range(16)), "dice")
    model_dir = os.path.join(a.out, "segmenter")
    ex = pool.train_session_export(sid, "safetensors", model_dir)
    print(f"segment dice {d0['dice_mean']:.3f} → {d1['dice_mean']:.3f} after {len(hist)} rounds; exported {ex.get('weights')}")
    return {"dice_before": d0, "dice_after": d1, "model_dir": model_dir, "session": sid, "rounds": hist}


if __name__ == "__main__":
    main()
