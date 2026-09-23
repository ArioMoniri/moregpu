#!/usr/bin/env python3
"""JEPA self-supervised pretraining on synthetic structured volumes — no data download, CPU-friendly.

  # against a running pool with ≥1 torch worker (apps/worker/worker_torch.py):
  python3 examples/jepa_synthetic.py --url http://localhost:8787 --token $MOREGPU_ADMIN_TOKEN --rounds 10
  # fully in-process reference (no pool), same semantics as the coordinator path:
  python3 examples/jepa_synthetic.py --local --workers 2 --rounds 10

Prints per-round loss, collapse monitors (per-dim std, RankMe) and a k-NN probe on the synthetic labels, then exports
the encoder (safetensors + ONNX) on a worker.
"""
import argparse, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "clients", "python"))
sys.path.insert(0, os.path.join(HERE, "..", "apps", "worker"))

SYN = {"kind": "2p5d", "n": 96, "size": [32, 32], "channels": 3, "seed": 0, "classes": 3}
JEPA = {"model": "micro", "patch": 8, "pred_dim": 16, "pred_depth": 1, "pred_heads": 2, "n_targets": 2,
        "ema": [0.99, 1.0], "weight_decay": 0.0, "probe_batch": 32}


def run_pool(a):
    from moregpu import MoreGPU
    pool = MoreGPU(a.url, a.token or os.environ.get("MOREGPU_ADMIN_TOKEN", ""))
    target = a.rounds * a.batch * a.inner * max(1, a.workers)
    s = pool.train_jepa("jepa_2p5d", synthetic=SYN, jepa={**JEPA, "total_steps": a.rounds * a.inner},
                        manifest_len=SYN["n"], batch=a.batch, inner_steps=a.inner, lr=a.lr, amp="fp32", seed=0,
                        target_samples=target, eval={"refs": list(range(48)), "kind": "knn", "every": max(1, a.rounds // 2)},
                        lr_schedule={"kind": "cosine", "warmup_frac": 0.1, "min_lr": a.lr * 0.05})
    sid = s["session"]
    print(f"session {sid} on {s['workers']} · params={s['params']}")
    hist = []
    while True:
        r = pool.train_session_round(sid, 1)
        last = r["rounds"][-1]; hist.append(last["avg_last_loss"])
        mon = last.get("monitors") or {}
        print(f"round {last['round']:3d} loss {last['avg_last_loss']:.4f} lr {last['lr']:.2e} std {mon.get('std_mean', float('nan')):.3f} "
              f"rankme {mon.get('rankme', float('nan')):.1f} samples {last['samples_seen']} {'eval ' + str(last.get('eval')) if last.get('eval') else ''}")
        if last["alarms"]:
            print("ALARMS:", last["alarms"])
        if last["done"]:
            break
    knn = pool.train_session_eval(sid, list(range(48)), "knn")
    ex = pool.train_session_export(sid, "safetensors", a.export_dir) if a.export_dir else None
    return {"session": sid, "losses": hist, "knn": knn, "export": ex, "telemetry": pool.train_session_telemetry(sid)}


def run_local(a):
    from moregpu_worker.train.local import run_diloco
    r = run_diloco("jepa_2p5d", {**JEPA, "synthetic": SYN, "total_steps": a.rounds * a.inner}, n_workers=a.workers,
                   rounds=a.rounds, inner_steps=a.inner, batch=a.batch, lr=a.lr, outer_lr=0.7, outer_momentum=0.9,
                   seed=0, manifest_len=SYN["n"],
                   lr_schedule={"kind": "cosine", "warmup_frac": 0.1, "min_lr": a.lr * 0.05})
    for i, l in enumerate(r["losses"][::a.workers]):
        print(f"round {i + 1:3d} loss {l:.4f}")
    w = r["workers"][0]
    print("knn", w.evaluate(list(range(48)), "knn"))
    return r


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--url", default=os.environ.get("MOREGPU_URL", "http://localhost:8787"))
    p.add_argument("--token")
    p.add_argument("--local", action="store_true")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--inner", type=int, default=4)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--export-dir", default=None, help="export dir ON THE WORKER, inside its MOREGPU_OUTPUT_DIR (relative → inside it)")
    a = p.parse_args(argv)
    return run_local(a) if a.local else run_pool(a)


if __name__ == "__main__":
    main()
