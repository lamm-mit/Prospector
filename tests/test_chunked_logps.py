"""`_chunked_logps_entropies` in `train/grpo.py` (the `logps_chunk` > 0 path)
matches TRL's `GRPOTrainer._get_per_token_logps_and_entropies` in values and
in gradients.

The chunked path calls the decoder and the unembedding separately and projects
one slice of positions at a time under `torch.utils.checkpoint`, so its window
arithmetic, its temperature scaling and the hidden state it projects must each
agree with TRL's. The reference is TRL's own method body on a duck-typed self.

Needs torch, trl and peft; without them the whole file is skipped. Run it
where `.[train]` is installed:

    python -m pytest tests/test_chunked_logps.py
"""
import types

import pytest

torch = pytest.importorskip(
    "torch",
    reason="torch absent, so the logps_chunk path is not tested in this environment; "
           "run it where .[train] is installed (see the module docstring).")
trl = pytest.importorskip(
    "trl", reason="trl absent: cannot compare against the original method.")
peft = pytest.importorskip(
    "peft", reason="peft absent: adapter-grad comparison needs a LoRA wrap.")

from trl import GRPOTrainer  # noqa: E402

from graph_reasoner.train.grpo import _chunked_logps_entropies  # noqa: E402

B, T, K, VOCAB = 2, 13, 9, 251
TEMPERATURE = 0.9  # not 1.0, so a missing /temperature is caught


def _tiny_lora_model():
    """A tiny random Llama causal LM with LoRA on the attention projections,
    fp32 on CPU. It has the decoder and unembedding structure the chunked path
    relies on."""
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(11)
    cfg = LlamaConfig(vocab_size=VOCAB, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=64)
    model = LlamaForCausalLM(cfg)
    lora = peft.LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                           task_type="CAUSAL_LM")
    model = peft.get_peft_model(model, lora)
    # PEFT zero-inits lora_B, which makes the adapter a no-op; perturb it so
    # the adapter path and disable_adapter() change the numbers.
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.add_(torch.randn_like(p) * 0.05)
    model.train()
    return model


def _batch():
    g = torch.Generator().manual_seed(7)
    input_ids = torch.randint(0, VOCAB, (B, T), generator=g)
    attention_mask = torch.ones(B, T, dtype=torch.long)
    attention_mask[1, -2:] = 0  # ragged row, as real padded batches are
    return input_ids, attention_mask


def _trl_reference(model, input_ids, attention_mask, logits_to_keep):
    """TRL's own method body on a duck-typed self."""
    dummy = types.SimpleNamespace(
        temperature=TEMPERATURE,
        model_kwarg_keys={"logits_to_keep"},
        _entropy_bonus_enabled=False,
    )
    return GRPOTrainer._get_per_token_logps_and_entropies(
        dummy, model, input_ids, attention_mask, logits_to_keep,
        compute_entropy=True)


def _grad_vec(model):
    return torch.cat([p.grad.flatten() for n, p in model.named_parameters()
                      if "lora_" in n and p.grad is not None])


@pytest.mark.parametrize("chunk", [3, 7, K, 1000])
def test_chunked_matches_trl_values_and_grads(chunk):
    """Every chunk size agrees with the unchunked original in values and in
    gradients: 3 is several chunks, 7 does not divide K, K is one exact chunk
    and 1000 is one oversized chunk."""
    model = _tiny_lora_model()
    input_ids, attention_mask = _batch()

    ref_lp, ref_ent, _ = _trl_reference(model, input_ids, attention_mask, K)
    ref_lp.sum().backward()
    ref_grads = _grad_vec(model)
    model.zero_grad()

    lp, ent = _chunked_logps_entropies(
        model, input_ids, attention_mask, K, chunk=chunk,
        temperature=TEMPERATURE, compute_entropy=True)
    assert lp.shape == (B, K) and ent.shape == (B, K)
    torch.testing.assert_close(lp, ref_lp, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(ent, ref_ent, rtol=1e-5, atol=1e-6)
    assert not ent.requires_grad, "diagnostic entropies must not retain graph"

    lp.sum().backward()
    torch.testing.assert_close(_grad_vec(model), ref_grads,
                               rtol=1e-5, atol=1e-6)


def test_no_grad_context_still_matches():
    """Under `torch.no_grad` and inside `disable_adapter()`, the context the
    reference log-probs are computed in, the values still match TRL's."""
    model = _tiny_lora_model()
    input_ids, attention_mask = _batch()
    with torch.no_grad():
        ref_lp, _, _ = _trl_reference(model, input_ids, attention_mask, K)
        with model.disable_adapter():
            ref_dis, _, _ = _trl_reference(model, input_ids, attention_mask, K)
            lp_dis, _ = _chunked_logps_entropies(
                model, input_ids, attention_mask, K, chunk=4,
                temperature=TEMPERATURE, compute_entropy=False)
        lp, _ = _chunked_logps_entropies(
            model, input_ids, attention_mask, K, chunk=4,
            temperature=TEMPERATURE, compute_entropy=False)
    torch.testing.assert_close(lp, ref_lp, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(lp_dis, ref_dis, rtol=1e-5, atol=1e-6)
    # the adapter must actually change the answer, or disable_adapter proved nothing
    assert not torch.allclose(lp, lp_dis), "LoRA had no effect; test is vacuous"


def test_empty_completion_returns_empty():
    """`logits_to_keep == 0` (no completion tokens) returns empty tensors."""
    model = _tiny_lora_model()
    input_ids, attention_mask = _batch()
    lp, ent = _chunked_logps_entropies(
        model, input_ids, attention_mask, 0, chunk=4,
        temperature=TEMPERATURE, compute_entropy=True)
    assert lp.shape == (B, 0) and ent.shape == (B, 0)

