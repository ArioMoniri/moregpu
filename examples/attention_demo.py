"""A real single-head self-attention forward pass, composed entirely on a MoreGPU pool.

    attention(Q,K,V) = softmax(Q·Kᵀ / √d) · V

Every heavy op (the two matmuls and the row-softmax) runs on the pool, sharded across workers, sealed,
and verified against the pool's CPU reference. This shows the honest "AI" story: MoreGPU gives you
fast, verified tensor primitives; your application composes them into a model — here, a transformer's
attention block. (It does not run a black-box model or use tensor cores.)

    export MOREGPU_URL=http://ADMIN:8787 MOREGPU_TOKEN=<admin-token>
    python examples/attention_demo.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from moregpu_client import MoreGPU


def flat(m):
    return [x for row in m for x in row]


def attention(pool, Q, K, V):
    T, d = len(Q), len(Q[0])
    Kt = [[K[t][j] for t in range(T)] for j in range(d)]              # transpose K → [d×T] (a reshape)
    scores = pool.matmul(flat(Q), flat(Kt), M=T, N=T, K=d)            # Q·Kᵀ            → [T×T]
    scaled = pool.run("scale", scores, scalar=1 / math.sqrt(d))["output_decoded"]
    weights = pool.run("softmax", scaled, N=T)["output_decoded"]      # row-softmax     → [T×T]
    return pool.matmul(weights, flat(V), M=T, N=d, K=T)              # weights·V        → [T×d]


if __name__ == "__main__":
    pool = MoreGPU(os.environ.get("MOREGPU_URL", "http://localhost:8787"),
                   os.environ.get("MOREGPU_TOKEN", ""))
    Q = [[0.1, 0.2, 0.3], [0.4, 0.1, 0.0], [0.2, 0.2, 0.2], [0.9, 0.1, 0.5]]
    K = [[0.3, 0.1, 0.2], [0.1, 0.4, 0.1], [0.0, 0.2, 0.3], [0.5, 0.5, 0.1]]
    V = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]
    out = attention(pool, Q, K, V)
    print("attention output (T×d):")
    d = len(Q[0])
    for i in range(0, len(out), d):
        print("  ", [round(x, 4) for x in out[i:i + d]])
