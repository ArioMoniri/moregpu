"""Golden cases for the round LR schedule (Python moregpu_worker.train.schedule ⇔ TS apps/coordinator/lib/schedule.ts).
Values are computed here from the closed-form definition, independent of either implementation."""
import json, math
from pathlib import Path

def ref(cfg, seen, rs, rnd):
    if cfg.get("target_samples"): p = (seen + rs / 2) / cfg["target_samples"]
    elif cfg.get("max_rounds"): p = (rnd + 0.5) / cfg["max_rounds"]
    else: return cfg["lr"]
    s = cfg.get("lr_schedule") or {}
    if s.get("kind", "constant") == "constant": return cfg["lr"]
    w, lo = s.get("warmup_frac", 0.0), s.get("min_lr", 0.0); p = min(1, max(0, p))
    if w > 0 and p < w: return cfg["lr"] * p / w
    q = (p - w) / max(1e-12, 1 - w)
    return lo + (cfg["lr"] - lo) * 0.5 * (1 + math.cos(math.pi * q))

cases = []
for cfg in [{"lr": 1e-3}, {"lr": 1e-3, "target_samples": 1000, "lr_schedule": {"kind": "cosine", "warmup_frac": 0.1, "min_lr": 1e-6}},
            {"lr": 5e-4, "max_rounds": 20, "lr_schedule": {"kind": "cosine", "warmup_frac": 0.05, "min_lr": 0}}]:
    for seen, rs, rnd in [(0, 64, 0), (64, 64, 1), (500, 64, 7), (990, 10, 15), (1000, 0, 19)]:
        cases.append({"cfg": cfg, "seen": seen, "round_samples": rs, "round": rnd, "lr": ref(cfg, seen, rs, rnd)})
Path(__file__).with_name("lr_schedule.json").write_text(json.dumps(cases))
print("wrote lr_schedule.json")
