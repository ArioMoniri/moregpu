# MoreGPU observability (Prometheus + Grafana)

Optional, self-hostable monitoring for a MoreGPU pool. Prometheus scrapes the admin server's
`/metrics` endpoint; Grafana loads a pre-provisioned **MoreGPU Pool** dashboard.

```bash
cd config/observability
export MOREGPU_ADMIN_TOKEN=<your admin token>          # from the server wizard
export MOREGPU_TARGET=host.docker.internal:8787        # or your admin host:port
docker compose up -d
```

- Grafana → http://localhost:3000 (login `admin` / `admin`) — the **MoreGPU Pool** dashboard is pre-loaded.
- Prometheus → http://localhost:9090.

The dashboard visualizes fleet size, queue depth, average user-util and pool-duty, total units/throughput,
job success/failure, and **per-worker contribution** (`moregpu_worker_units` / `moregpu_worker_share`,
legend `{{worker}}`). The dashboard JSON is `grafana/dashboards/moregpu.json` (also at
`config/grafana/moregpu-dashboard.json`) — import it into any existing Grafana.

Nothing here is required to run MoreGPU; the pool works without it.
