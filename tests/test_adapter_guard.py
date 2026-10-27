"""`assert_adapter_resolves` in `train/adapters.py`, and
`eval/local_model.py::merge_chain_into`, the merge loop that applies it.

PEFT's `from_pretrained` wraps the modules an adapter's tensor names resolve to
and silently drops the tensors that name modules the model does not have (for
example unfused in_proj_q/k/v against a fused in_proj_qkv). The guard refuses
such an adapter instead of merging part of it.

Needs torch and peft: the file is skipped without them, so run it where
`.[train]` is installed.
"""
import pytest

torch = pytest.importorskip(
    "torch",
    reason="torch absent, so the adapter-load guard is not tested in this environment; "
           "run it where .[train] is installed (see the module docstring).")
peft = pytest.importorskip("peft", reason="peft absent")

from graph_reasoner.train.adapters import assert_adapter_resolves  # noqa: E402


def _tiny_base():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(3)
    cfg = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
                      num_hidden_layers=2, num_attention_heads=2,
                      num_key_value_heads=2, max_position_embeddings=32)
    return LlamaForCausalLM(cfg)


def _tiny_lora_model():
    return peft.get_peft_model(_tiny_base(), peft.LoraConfig(
        r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM"))


def _trained_adapter(path, seed):
    """A saved adapter whose B matrices are nonzero, so merging it moves the
    weights (a fresh LoRA's B is zero and its merge is a no-op)."""
    pm = _tiny_lora_model()
    torch.manual_seed(seed)
    with torch.no_grad():
        for name, param in pm.named_parameters():
            if "lora_B" in name:
                param.normal_()
    pm.save_pretrained(str(path))
    return str(path)


def test_clean_adapter_passes(tmp_path):
    pm = _tiny_lora_model()
    pm.save_pretrained(str(tmp_path / "a"))
    fresh = _tiny_lora_model()  # any model wrapping the same module names
    assert_adapter_resolves(fresh, str(tmp_path / "a"))  # must not raise


def test_orphan_tensor_is_refused(tmp_path):
    from safetensors.torch import load_file, save_file

    pm = _tiny_lora_model()
    pm.save_pretrained(str(tmp_path / "a"))
    f = tmp_path / "a" / "adapter_model.safetensors"
    tensors = load_file(str(f))
    key = next(k for k in tensors if "q_proj.lora_A" in k)
    # A trained tensor whose module name the model does not have.
    tensors[key.replace("q_proj", "q_proj_unfused_alias")] = tensors.pop(key)
    save_file(tensors, str(f))
    with pytest.raises(SystemExit, match="q_proj_unfused_alias"):
        assert_adapter_resolves(pm, str(tmp_path / "a"))


def test_missing_checkpoint_is_not_this_guards_problem(tmp_path):
    pm = _tiny_lora_model()
    assert_adapter_resolves(pm, str(tmp_path))  # no safetensors file: no-op


def test_merge_chain_into_merges_each_adapter_in_order(tmp_path):
    from graph_reasoner.eval.local_model import merge_chain_into

    chain = [_trained_adapter(tmp_path / "a0", 10), _trained_adapter(tmp_path / "a1", 11)]
    want = _tiny_base()
    for a in chain:
        want = peft.PeftModel.from_pretrained(want, a).merge_and_unload()
    got = merge_chain_into(_tiny_base(), chain)
    assert not isinstance(got, peft.PeftModel)
    want_sd, got_sd = want.state_dict(), got.state_dict()
    assert want_sd.keys() == got_sd.keys()
    for k in want_sd:
        assert torch.equal(want_sd[k], got_sd[k]), k
    q = "model.layers.0.self_attn.q_proj.weight"
    assert not torch.equal(_tiny_base().state_dict()[q], got_sd[q])
    assert merge_chain_into(_tiny_base(), []).state_dict()[q].equal(
        _tiny_base().state_dict()[q])


# peft warns about the key this test renames away; the warning is expected, and
# the guard's refusal of the orphaned tensor is what is asserted.
@pytest.mark.filterwarnings("ignore:Found missing adapter keys:UserWarning")
def test_merge_chain_into_refuses_an_orphan_tensor(tmp_path):
    from safetensors.torch import load_file, save_file

    from graph_reasoner.eval.local_model import merge_chain_into

    a = _trained_adapter(tmp_path / "a", 12)
    f = tmp_path / "a" / "adapter_model.safetensors"
    tensors = load_file(str(f))
    key = next(k for k in tensors if "v_proj.lora_B" in k)
    tensors[key.replace("v_proj", "v_proj_unfused_alias")] = tensors.pop(key)
    save_file(tensors, str(f))
    with pytest.raises(SystemExit, match="v_proj_unfused_alias"):
        merge_chain_into(_tiny_base(), [a])
