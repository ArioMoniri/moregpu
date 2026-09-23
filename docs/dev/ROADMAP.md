# Roadmap — vision, training tasks, JEPA

(`docs/ROADMAP.md` is git-ignored upstream, so this roadmap lives here.) Items needed by the study come first.

| Milestone | Scope | Exit criterion | Status |
|---|---|---|---|
| M1 | Training framework, `llm_lora` migration, session isolation, heterogeneous DiLoCo, AMP | legacy LLM e2e unchanged; DiLoCo equivalence tests | ✅ `tests/e2e/train_sessions.py`, `tests/py/test_task_framework.py` |
| M2 | JEPA 2D/2.5D, telemetry and bench, data plane and loader | CPU CI example; 2-worker run | ✅ `tests/e2e/jepa_sessions.py`, `tests/e2e/data_plane_jepa.py`. The 2-GPU run is pending Lambda. |
| M3 | `segment` fine-tuning and encoder export | segment example green | ✅ `tests/e2e/vision_pipeline.py` |
| M4 | Model adapters and automatic lowering | parity and security tests | ✅ `tests/py/test_vision_*`, `tests/e2e/published_model.py` |
| M5 | JEPA 3D, compression, churn/resume hardening, `/vision/batch`, tile sharding, `finetune_model` | resume and compression tests | ✅ |
| M6 | WebGPU, browser and Deno vision; mixed fleet | mixed-fleet parity | kernels ✅ on software adapters; mixed-fleet dispatch in progress |

## Next

1. **Real-GPU CI evidence.** Run the `cuda`-marked tests and the 2-GPU JEPA example on Lambda during the study pilot,
   and attach the logs to the PR.
2. **Release.** Cut a 0.7.0 tag after the owner's OK. Signing covers `worker.ts`, `vision_wgsl.ts` and the
   `moregpu_worker` manifest.
3. **Stretch (ADR + prototype only):**
   - vision-language serving (a vision encoder feeding the LLM path);
   - limited WGSL backward kernels, so WebGPU workers can train small heads;
   - asynchronous DiLoCo;
   - secure aggregation;
   - worker-to-worker all-reduce to remove the coordinator's 2·N·|θ| bandwidth bottleneck.
