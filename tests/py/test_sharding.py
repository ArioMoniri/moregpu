import json
import math
from pathlib import Path

import pytest

from moregpu_worker.train import sharding as S


def test_splitmix_golden_matches_ts():
    g = json.loads((Path(__file__).parents[1] / "goldens" / "sharding.json").read_text())
    assert S.permutation(g["n"], g["seed"], g["epoch"]) == g["perm"]


def test_permutation_is_a_permutation_and_seed_dependent():
    p = S.permutation(100, 7, 0)
    assert sorted(p) == list(range(100))
    assert p != S.permutation(100, 8, 0) and p != S.permutation(100, 7, 1)
    assert p == S.permutation(100, 7, 0)


def test_stream_crosses_epochs_and_is_resumable():
    s = S.SampleStream(10, seed=3)
    a = s.take(7) + s.take(7)
    assert a[:10] == S.permutation(10, 3, 0) and a[10:] == S.permutation(10, 3, 1)[:4]
    s2 = S.SampleStream.from_state(S.SampleStream(10, 3).state())
    s3 = S.SampleStream(10, 3); s3.take(7); s4 = S.SampleStream.from_state(s3.state())
    assert s4.take(7) == a[7:]
    assert s2.cursor == 0


def test_split_contiguous_in_worker_order():
    assert S.split([0, 1, 2, 3, 4, 5], [1, 2, 3]) == [[0], [1, 2], [3, 4, 5]]
    with pytest.raises(ValueError):
        S.split([0, 1], [1, 2])


@pytest.mark.parametrize("mode", ["fixed", "proportional"])
def test_allocate_sums_to_budget(mode):
    alloc = S.allocate(per_worker=8, n=3, speeds=[1.0, 2.0, 1.0], mode=mode)
    assert sum(alloc) == 24 and all(a >= 1 for a in alloc)
    if mode == "fixed":
        assert alloc == [8, 8, 8]
    else:
        assert alloc[1] > alloc[0]


def test_allocate_truncates_to_remaining_target():
    assert sum(S.allocate(8, 3, [1, 1, 1], "fixed", remaining=5)) == 5


def test_lr_schedule_matches_golden():
    from moregpu_worker.train.schedule import lr_at, progress
    import json
    from pathlib import Path
    g = json.loads((Path(__file__).parents[1] / "goldens" / "lr_schedule.json").read_text())
    for c in g:
        assert abs(lr_at(c["cfg"], progress(c["cfg"], c["seen"], c["round_samples"], c["round"])) - c["lr"]) < 1e-12


def test_lr_schedule_shapes():
    from moregpu_worker.train.schedule import lr_at, progress
    cfg = {"lr": 1.0, "target_samples": 100, "lr_schedule": {"kind": "cosine", "warmup_frac": 0.1, "min_lr": 0.0}}
    assert lr_at(cfg, progress(cfg, 0, 10, 0)) == pytest.approx(0.5)      # midpoint of round 0 = 5% → half warmup
    assert lr_at(cfg, 0.1) == pytest.approx(1.0)
    assert lr_at(cfg, 1.0) == pytest.approx(0.0, abs=1e-12)
    assert lr_at({"lr": 0.3}, 0.5) == 0.3 and lr_at({"lr": 0.3, "lr_schedule": {"kind": "cosine"}}, None) == 0.3
    cfg_r = {"lr": 1.0, "max_rounds": 10, "lr_schedule": {"kind": "cosine", "min_lr": 0.1}}
    assert lr_at(cfg_r, progress(cfg_r, 0, 0, 4)) == pytest.approx(0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * 0.45)))
    assert progress({}, 0, 0, 0) is None
    with pytest.raises(ValueError):
        lr_at({"lr": 1, "lr_schedule": {"kind": "step"}}, 0.2)
