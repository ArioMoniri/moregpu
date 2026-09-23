"""Vision data plane (ADR-0110): policy-checked refs, readers, content cache, pushed blobs, batches, fast shards.

Import submodules directly (``from moregpu_worker.data.plane import DataPlane``); this package init stays empty of
heavy imports so ``refs``/``cache``/``blobs`` load without torch.
"""
