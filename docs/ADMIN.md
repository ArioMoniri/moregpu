# MoreGPU — Admin & Operations Guide

Operational reference for running a MoreGPU coordinator: the setup wizard, the dashboard, token rotation, isolated pools, TLS, throttle tuning, service management, and troubleshooting.

For getting a pool running end to end, see the root `README.md`. This document assumes the coordinator command from there.

---

## 1. The setup wizard

The first time you start the coordinator, it runs a one-time wizard before serving traffic. The wizard generates, for this pool only:

- an **admin token** — authenticates job submission and the dashboard;
- a **worker join token** — authenticates machines enrolling as workers;
- an **AES-GCM encryption key** — seals each work unit so only ciphertext crosses the wire.

It then prints the ready-to-copy worker one-liner (with `MOREGPU_SERVER` and `MOREGPU_TOKEN` filled in) and the dashboard URL.

These values are written to the coordinator's config file on disk. They persist across restarts, so subsequent server starts skip the wizard and reuse the same identity. Record the admin token somewhere safe: the pool is controlled by whoever holds it.

---

## 2. The dashboard

Open `http://HOST:8787` (or your configured `MOREGPU_HOST` / `PORT`, and `https://` when TLS is enabled).

- **Authenticate** by pasting the admin token.
- **Fleet view** shows connected workers: name (`MOREGPU_NAME`), GPU or CPU mode, and live status.
- **Submit jobs** from the dashboard instead of `curl` — choose a task type and size, submit, and watch shards distribute, complete, and pool back.

The dashboard uses the same admin-token authentication as the HTTP `/submit` endpoint. Anyone with the admin token has full control of the pool; treat it as a secret.

---

## 3. Job submission over HTTP

Equivalent to the dashboard, for scripting:

```sh
curl -X POST http://ADMIN:8787/submit \
  -H "authorization: Bearer <admin-token>" \
  -H "content-type: application/json" \
  -d '{"size":1024}'
```

Kernels today: `matmul`, `vector_add`, `vector_mul`, `saxpy`, `relu`, `scale`, `softmax`, `layernorm` (all fp32, sharded and pooled). `matmul` — the compute-bound one — runs on each worker's GPU via WGSL; the elementwise and row-wise kernels are memory-bound and run on the worker's CPU. To add a kernel, provide a compute path plus a matching CPU reference implementation — the CPU reference is what results are verified against. See [AI_USAGE.md](AI_USAGE.md) for the full kernel map.

---

## 4. Rotating tokens

There is no in-place rotation command. Tokens live in the coordinator's config file; to regenerate them, delete that config file and start the server again. The wizard runs again and mints a fresh admin token, join token, and encryption key.

**Warning:** regenerating changes the pool's identity. Every currently enrolled worker holds the old join token and encryption key, so after rotation:

- existing workers can no longer authenticate or decrypt work units and must be re-enrolled with the new join token;
- the old admin token stops working — update any scripts or saved dashboard sessions;
- re-run the printed worker one-liner on every machine.

Rotate deliberately, during a maintenance window, not casually.

---

## 5. Running multiple isolated pools

Each pool is defined by its own config file (its own admin token, join token, and encryption key). To run more than one isolated pool:

- run each coordinator with its **own config file** and its **own `PORT`** (and/or `MOREGPU_HOST`);
- give each pool's workers the join token and `MOREGPU_SERVER` for **that** pool only.

Tokens are never shared across pools. A worker enrolled in pool A cannot join or decrypt work from pool B. Keep each pool's admin token separate; there is no cross-pool admin.

---

## 6. TLS (`wss://`)

To serve the coordinator and worker WebSocket over TLS, set both variables before starting the server:

```sh
MOREGPU_TLS_CERT=/path/to/fullchain.pem \
MOREGPU_TLS_KEY=/path/to/privkey.pem \
deno run --allow-net --allow-env --allow-read --allow-write \
  https://raw.githubusercontent.com/ArioMoniri/moregpu/main/apps/coordinator/server.ts
```

With TLS enabled:

- workers connect using `MOREGPU_SERVER=wss://ADMIN:8787/ws` (note `wss`, not `ws`);
- the dashboard is served over `https://`.

Provide certificate and key for a hostname the workers can reach. Both variables must be set together; setting only one leaves the server on plaintext.

---

## 7. Throttle tuning

Two levers bound how hard workers run, keeping electricity low and the interactive user undisturbed.

- **`MOREGPU_THROTTLE`** (per worker) — duty cycle in the range `0.05`–`1`. `0.4` means the worker is active roughly 40% of the time. Lower it on shared desktops and laptops; raise it toward `1` on dedicated machines.
- **`MOREGPU_DUTY`** (on the coordinator) — pool-side duty-cycle control set as an environment variable when starting the server.

Guidance:

- Interactive workstations that someone is actively using: keep the throttle low (e.g. `0.2`–`0.4`).
- Dedicated or overnight-idle machines: raise toward `1` for maximum contribution.
- Combine with `MOREGPU_FORCE_CPU=1` on machines whose GPU must stay free for the local user.

---

## 8. Service management (reboot survival)

Enrol a worker with `MOREGPU_SERVICE=1` to install a reboot-surviving, self-healing service. The installer picks the right mechanism per OS. Inspect and control it as follows.

### Linux — systemd (user service)

```sh
systemctl --user status moregpu-worker
systemctl --user restart moregpu-worker
systemctl --user stop moregpu-worker
journalctl --user -u moregpu-worker -f      # live logs
```

If the worker should keep running after logout, enable lingering: `loginctl enable-linger $USER`.

### macOS — launchd (launchctl)

```sh
launchctl list | grep moregpu
launchctl kickstart -k gui/$(id -u)/dev.moregpu.worker   # restart
launchctl bootout   gui/$(id -u)/dev.moregpu.worker      # stop/unload
```

### Windows — scheduled task

```powershell
Get-ScheduledTask   -TaskName MoreGPUWorker
Start-ScheduledTask -TaskName MoreGPUWorker
Stop-ScheduledTask  -TaskName MoreGPUWorker
Unregister-ScheduledTask -TaskName MoreGPUWorker    # remove
```

The service runs the same supervised restart loop as a foreground worker: if the process exits or the Deno install is incomplete, it retries and restarts automatically.

---

## 9. Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| Worker never appears in the fleet | Wrong `MOREGPU_SERVER` or unreachable admin host/port | Confirm the URL and port; ensure the admin server is running and reachable outbound from the worker. |
| Worker connects then drops | Join token mismatch (e.g. after a rotation) | Re-run the worker one-liner with the current join token. |
| `wss://` connection fails | TLS not enabled, or `ws://` used against a TLS server (or vice versa) | Set both `MOREGPU_TLS_CERT` and `MOREGPU_TLS_KEY`; match the scheme (`ws`/`wss`) to the server. |
| `/submit` returns unauthorized | Wrong or rotated admin token | Use the admin token printed by the wizard; if the config was deleted, the token changed. |
| Results fail verification | Task type without a matching CPU reference, or a bad kernel | Ensure each task type has both a WGSL kernel and a CPU reference. |
| Machine runs hot or disturbs the user | Throttle too high | Lower `MOREGPU_THROTTLE`; consider `MOREGPU_FORCE_CPU=1`. |
| Worker didn't restart after reboot | Enrolled without service, or (Linux) lingering disabled | Re-enrol with `MOREGPU_SERVICE=1`; on Linux run `loginctl enable-linger $USER`. |
| Wizard didn't run on start | Config file already exists | Delete the config file to re-run the wizard (this rotates all tokens — see section 4). |

---

## 10. Authorized use

Run workers only on machines you own or are explicitly authorized to use. MoreGPU is neutral infrastructure: it ships a generic authorized-runtime worker joiner and no automation for evading any platform's Terms of Service, quotas, or bot-detection — using it on third-party platforms is very often prohibited by their Terms of Service. All responsibility, legal risk, and compliance rest entirely with the operator. The software is provided AS IS, without warranty, and the authors accept no liability. Apache-2.0 © 2026 Ariorad Moniri.
