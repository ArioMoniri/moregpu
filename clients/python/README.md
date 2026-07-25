# moregpu-client

Dependency-free Python client + CLI for the [MoreGPU](https://github.com/ArioMoniri/moregpu) distributed
GPU compute pool.

```bash
pip install moregpu-client
```

```python
from moregpu import MoreGPU
pool = MoreGPU("http://ADMIN:8787", "<admin-token>")

pool.matmul([1,2,3, 4,5,6], [7,8, 9,10, 11,12], M=2, N=2, K=3)   # → [58, 64, 139, 154]
pool.run("relu", [-1, 2, -3, 4])["output_decoded"]               # → [0.0, 2.0, 0.0, 4.0]
pool.device()                                                     # the pool presented as a GPU slot
```

CLI:

```bash
export MOREGPU_URL=http://ADMIN:8787 MOREGPU_TOKEN=<admin-token>
moregpu-client device
moregpu-client submit matmul 1024
```

Standard library only. See [docs/AI_USAGE.md](https://github.com/ArioMoniri/moregpu/blob/main/docs/AI_USAGE.md).
Apache-2.0 © Ariorad Moniri.
