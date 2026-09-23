import pytest
import torch

transformers = pytest.importorskip("transformers")
from moregpu_worker.train.tasks.llm_lora import LlmLoraTask, attach_lora, LoRAWrap  # noqa: E402
from moregpu_worker.train.task import TaskContext  # noqa: E402


def tiny(seed=0):
    torch.manual_seed(seed)
    return transformers.GPT2LMHeadModel(transformers.GPT2Config(vocab_size=64, n_positions=32, n_embd=16, n_layer=2, n_head=2))


def test_attach_lora_freezes_base_and_starts_as_noop():
    m = tiny().eval(); ids = torch.randint(0, 64, (1, 8))
    with torch.no_grad():
        before = m(ids).logits
    n = attach_lora(m, ["c_attn"], 4, 8)
    assert n == 2 * (4 * 16 + 48 * 4)
    with torch.no_grad():
        assert torch.allclose(m(ids).logits, before, atol=1e-6)
    assert all(isinstance(m.transformer.h[i].attn.c_attn, LoRAWrap) for i in range(2))


def test_task_contract_and_learning(tmp_path):
    m = tiny(); m.save_pretrained(tmp_path / "m", safe_serialization=True)
    t = LlmLoraTask()
    info = t.init({"model_dir": str(tmp_path / "m"), "rank": 4, "alpha": 8, "no_dropout": True, "seed": 0},
                  TaskContext(device="cpu"))
    assert info["trainable_params"] > 0 and info["targets"] == ["c_attn", "q_proj", "v_proj"]
    ids = torch.randint(0, 64, (1, 12))
    losses = [t.train_step(ids, ids, lr=5e-3) for _ in range(15)]
    assert losses[-1] < losses[0] and t.step == 15
    rep = t.inner_steps([ids[0].tolist()] * 2, 3, 1e-3)
    assert len(rep.losses) == 3 and t.step == 18
    st = t.state_for_sync()
    assert all(k.endswith((".A", ".B")) for k in st)
    t.load_sync_state({k: torch.zeros_like(v) for k, v in st.items()})
    assert all((v == 0).all() for v in t.state_for_sync().values())
    t.close(); assert t.model is None


def test_task_accepts_model_object():
    t = LlmLoraTask()
    t.init({"model_obj": tiny(), "targets": ["c_attn"]}, TaskContext(device="cpu"))
    assert len(t.trainable) == 4
