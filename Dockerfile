# MoreGPU coordinator (the admin server). Workers run on hosts with real GPUs and join over the network;
# this image is the GPU-free coordinator — run it anywhere, then point workers at it.
#
#   docker run -p 8787:8787 -v moregpu-data:/data ghcr.io/ariomoniri/moregpu:latest
#   # first-run tokens are printed to the logs and persisted in the /data volume
#
FROM denoland/deno:alpine

# link this package to the repo (so it shows in the repo's Packages sidebar) + metadata
LABEL org.opencontainers.image.source="https://github.com/ArioMoniri/moregpu"
LABEL org.opencontainers.image.description="MoreGPU coordinator — the admin server for a native GPU compute pool"
LABEL org.opencontainers.image.licenses="Apache-2.0"

WORKDIR /app
COPY apps/coordinator/server.ts ./apps/coordinator/server.ts
RUN deno cache apps/coordinator/server.ts

# persist each pool's own tokens/key across restarts in a mounted volume
USER root
RUN mkdir -p /data && chmod 777 /data
ENV PORT=8787 MOREGPU_BIND=0.0.0.0 MOREGPU_CONFIG=/data/.moregpu-server.json
VOLUME ["/data"]
EXPOSE 8787

# no --allow-run: this is coordinator-only (no built-in worker); workers join from other machines
ENTRYPOINT ["deno", "run", "--allow-net", "--allow-read", "--allow-write", "--allow-env", "apps/coordinator/server.ts"]
