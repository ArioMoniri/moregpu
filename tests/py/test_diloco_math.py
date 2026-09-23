import torch
import pytest

from moregpu_worker.train import diloco as D


def _t(*v):
    return torch.tensor(v, dtype=torch.float32)


def test_weighted_average_by_samples_and_equal_weights_is_plain_mean():
    a, b = {"w": _t(1, 2)}, {"w": _t(3, 6)}
    assert torch.allclose(D.weighted_average([(a, 1), (b, 1)])["w"], _t(2, 4))
    assert torch.allclose(D.weighted_average([(a, 1), (b, 3)])["w"], _t(2.5, 5))


def test_weighted_average_rejects_nonpositive_total_and_shape_mismatch():
    with pytest.raises(ValueError):
        D.weighted_average([({"w": _t(1)}, 0)])
    with pytest.raises(ValueError):
        D.weighted_average([({"w": _t(1)}, 1), ({"w": _t(1, 2)}, 1)])


def test_outer_step_matches_torch_sgd_nesterov():
    g0 = {"w": _t(1.0, -2.0, 3.0)}
    avgs = [_t(0.5, -1.0, 2.0), _t(0.2, -0.5, 1.0), _t(0.1, 0.0, 0.5)]
    st = D.OuterState.init(g0)
    p = torch.nn.Parameter(g0["w"].clone())
    opt = torch.optim.SGD([p], lr=0.7, momentum=0.9, nesterov=True)
    for a in avgs:
        opt.zero_grad(); p.grad = (p.detach() - a).clone(); opt.step()
        D.outer_step(st, {"w": a}, lr=0.7, momentum=0.9)
        assert torch.allclose(st.global_["w"], p.detach(), atol=1e-6)


def test_outer_lr1_mom0_is_plain_averaging():
    st = D.OuterState.init({"w": _t(5, 5)})
    D.outer_step(st, {"w": _t(1, 2)}, lr=1.0, momentum=0.0)
    assert torch.equal(st.global_["w"], _t(1, 2))


def test_nonfinite_filter():
    good, bad = ({"w": _t(1)}, 1), ({"w": _t(float("nan"))}, 1)
    kept, dropped = D.drop_nonfinite([("a", *good), ("b", *bad)])
    assert [k for k, *_ in kept] == ["a"] and dropped == ["b"]
