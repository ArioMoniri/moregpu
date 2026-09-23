"""Golden for the SplitMix64 Fisher-Yates permutation shared by Python (moregpu_worker.train.sharding) and the
coordinator (apps/coordinator/lib/sharding.ts). Hand-rolled here independently of the module under test."""
import json
from pathlib import Path

M = (1 << 64) - 1


def splitmix(state):
    while True:
        state = (state + 0x9E3779B97F4A7C15) & M
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M
        yield z ^ (z >> 31)


def perm(n, seed, epoch):
    rng = splitmix((seed * 0x100000001B3 + epoch) & M)
    p = list(range(n))
    for i in range(n - 1, 0, -1):
        j = next(rng) % (i + 1)
        p[i], p[j] = p[j], p[i]
    return p


Path(__file__).with_name("sharding.json").write_text(json.dumps({"n": 37, "seed": 42, "epoch": 3, "perm": perm(37, 42, 3)}))
print("wrote sharding.json")
