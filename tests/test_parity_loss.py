"""`_parity_loss` (train/grpo.py), the GRPO trainer's only loss, against the
reference arithmetic of `token_advantages` (train/reward.py).

The loss weights each token's advantage by the importance ratio between the
trainer's and the sampler's logprobs, and subtracts a k1 KL term from the
advantage, so the penalty is linear in the gap to the reference.

This module needs torch. Without it the whole module is skipped, so a passing
suite in a venv without torch says nothing about the loss. Run it where
`.[train]` is installed:

    python -m pytest tests/test_parity_loss.py tests/test_chunked_logps.py tests/test_adapter_guard.py
"""
import pytest

torch = pytest.importorskip(
    "torch",
    reason="torch absent, so the parity loss is not tested in this environment; "
           "run it where .[train] is installed (see the module docstring).")

from graph_reasoner.train.grpo import _parity_loss
from graph_reasoner.train.reward import token_advantages


def _hand_built(per_token_logps, sampling, ref, advantages, mask, kl_beta,
                denom, bessel):
    """The reference: per-sequence token_advantages + explicit ratio sum."""
    total = 0.0
    B, T = mask.shape
    for i in range(B):
        ntok = max(1.0, float(mask[i].sum()))
        kl = ((per_token_logps[i] - ref[i]).tolist() if ref is not None
              else [0.0] * T)
        av = token_advantages(
            float(advantages[i]) * bessel,
            mask=[int(x) for x in mask[i].tolist()],
            kl_tok=kl, kl_beta=kl_beta, norm=1.0 / ntok)
        ratio = torch.exp(per_token_logps[i] - sampling[i])
        total += float((ratio * torch.tensor(av)).sum())
    return -total / denom


def _batch():
    g = torch.Generator().manual_seed(7)
    B, T = 3, 11
    lp = -torch.rand((B, T), generator=g) * 3
    slp = lp + torch.randn((B, T), generator=g) * 0.05   # small sampler drift
    ref = lp + torch.randn((B, T), generator=g) * 0.30   # displaced reference
    adv = torch.tensor([1.2, -0.7, 0.1])
    mask = torch.tensor([[1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0],   # 5 tokens
                         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],   # 11 tokens
                         [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]],  # 2 tokens
                        dtype=torch.float32)
    return lp, slp, ref, adv, mask


def test_parity_loss_equals_token_advantages_arithmetic():
    lp, slp, ref, adv, mask = _batch()
    for kl_beta, bessel in [(0.0, 1.0), (0.015, 1.0), (0.015, (16 / 15) ** 0.5)]:
        got = float(_parity_loss(lp, slp, ref, adv, mask, kl_beta,
                                 denom=6.0, bessel=bessel))
        want = _hand_built(lp, slp, ref, adv, mask, kl_beta, 6.0, bessel)
        assert got == pytest.approx(want, rel=1e-5), (kl_beta, bessel)


def test_masked_tokens_carry_no_gradient_mass():
    """Perturbing logprobs at masked (observation) positions does not move
    the loss."""
    lp, slp, ref, adv, mask = _batch()
    base = float(_parity_loss(lp, slp, ref, adv, mask, 0.015, 6.0))
    lp2 = lp.clone(); lp2[mask == 0] -= 5.0
    ref2 = ref.clone(); ref2[mask == 0] += 3.0
    again = float(_parity_loss(lp2, slp, ref2, adv, mask, 0.015, 6.0))
    assert again == pytest.approx(base, rel=1e-6)


def test_sequences_contribute_equally_regardless_of_length():
    """The per-sequence norm: a 5-token and an 11-token sequence with the
    same advantage and ratio 1 contribute the same gradient mass."""
    T = 11
    lp = torch.zeros((2, T)); slp = torch.zeros((2, T))
    mask = torch.zeros((2, T)); mask[0, :5] = 1; mask[1, :] = 1
    adv = torch.tensor([1.0, 1.0])
    loss = float(_parity_loss(lp, slp, None, adv, mask, 0.0, denom=2.0))
    # ratio==1, adv 1: each sequence contributes mean-over-its-tokens = 1;
    # loss = -(1 + 1)/2
    assert loss == pytest.approx(-1.0)


def test_ratio_reweights_drifted_tokens():
    """Where the trainer assigns less probability than the sampler did
    (lp < slp), the token's weight falls below 1."""
    T = 4
    lp = torch.full((1, T), -2.0); slp = torch.full((1, T), -1.0)
    mask = torch.ones((1, T)); adv = torch.tensor([1.0])
    loss = float(_parity_loss(lp, slp, None, adv, mask, 0.0, denom=1.0))
    assert loss == pytest.approx(-torch.e ** -1, rel=1e-4)  # ratio e^-1 < 1


def test_k1_kl_is_linear_never_exponential():
    """The KL adjustment to the advantage is linear in the gap to the
    reference, at any size of gap."""
    T = 2
    lp = torch.zeros((1, T)); slp = torch.zeros((1, T))
    mask = torch.ones((1, T)); adv = torch.tensor([0.0])
    for gap in (1.0, 10.0, 50.0):
        ref = torch.full((1, T), -gap)   # policy displaced `gap` nats above ref
        loss = float(_parity_loss(lp, slp, ref, adv, mask, 0.015, denom=1.0))
        # adv 0 -> adv_tok = -kl_beta*gap/ntok per token; ratio 1;
        # loss = +kl_beta*gap, linear in gap.
        assert loss == pytest.approx(0.015 * gap, rel=1e-5)


def test_accumulation_is_batch_size_invariant():
    """With denom = rows x accum, the summed loss and its accumulated gradient
    are the same however the microbatches slice the sequences. Padding a row
    to a common width changes nothing: padded positions carry mask 0."""
    g = torch.Generator().manual_seed(3)
    N, T = 8, 6
    lp = torch.randn(N, T, generator=g, requires_grad=True)
    sampling = (torch.randn(N, T, generator=g) * 0.1).detach()
    ref = torch.randn(N, T, generator=g)
    adv = torch.randn(N, generator=g)
    mask = torch.ones(N, T)
    mask[1, -3:] = 0  # ragged row: its pad positions must contribute nothing

    def total(bs):
        accum = N // bs
        losses = []
        for i in range(0, N, bs):
            losses.append(_parity_loss(
                lp[i:i + bs], sampling[i:i + bs], ref[i:i + bs],
                adv[i:i + bs], mask[i:i + bs], kl_beta=0.015,
                denom=bs * accum, bessel=1.06))
        return torch.stack(losses).sum()

    ref_loss = total(1)
    for bs in (2, 4, 8):
        torch.testing.assert_close(total(bs), ref_loss, rtol=1e-6, atol=1e-7)

    ref_loss.backward()
    grad_bs1 = lp.grad.clone()
    lp.grad = None
    total(4).backward()
    torch.testing.assert_close(lp.grad, grad_bs1, rtol=1e-6, atol=1e-7)


def test_ratio_clip_bounds_drifted_tokens_and_is_off_by_default():
    """`is_ratio_clip`: a token whose sampler and trainer logprobs differ by
    more than log(c) contributes at most c times its advantage; with the clip
    unset the ratio is unclipped."""
    lp = torch.tensor([[0.0, -1.0, -1.0]])
    slp = torch.tensor([[-3.0, -1.0, -1.0]])      # first token: ratio e^3 ≈ 20
    adv = torch.tensor([1.0]); mask = torch.ones(1, 3)
    raw = float(_parity_loss(lp, slp, None, adv, mask, 0.0, denom=1.0))
    same = float(_parity_loss(lp, slp, None, adv, mask, 0.0, denom=1.0, ratio_clip=None))
    clipped = float(_parity_loss(lp, slp, None, adv, mask, 0.0, denom=1.0, ratio_clip=2.0))
    assert raw == same
    # unclipped: -(e^3 + 1 + 1)/3 ; clipped: -(2 + 1 + 1)/3
    assert abs(raw + (2.718281828 ** 3 + 2) / 3) < 1e-4
    assert abs(clipped + 4.0 / 3) < 1e-6
