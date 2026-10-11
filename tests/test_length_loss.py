"""Unit tests 7, 8: differentiable TLC length loss (brief section 20)."""
import torch

from src.models.length_loss import LengthAwareLoss, expected_length_from_logits


def _toy(batch=2, T=6, V=20, eos=3, seed=0):
    torch.manual_seed(seed)
    logits = torch.randn(batch, T, V, requires_grad=True)
    labels = torch.randint(1, V, (batch, T))
    labels[:, -1] = eos
    labels[0, 4:] = -100  # some padding on the first example
    return logits, labels, eos


def test_expected_length_is_finite_and_masked():
    logits, labels, eos = _toy()
    el = expected_length_from_logits(logits, labels, eos)
    assert el.shape == (2,)
    assert torch.isfinite(el).all()
    # example 0 has 4 real target positions -> expected length <= 4
    assert el[0].item() <= 4.0 + 1e-4


def test_length_loss_finite_and_has_gradient():
    logits, labels, eos = _toy()
    loss = LengthAwareLoss(eos_id=eos, length_loss_lambda=0.5)(logits, labels)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_gradient_propagates_to_eos_logits_specifically():
    logits, labels, eos = _toy()
    # isolate the length term
    ll = LengthAwareLoss(eos_id=eos, length_loss_lambda=1.0)
    term = ll.length_term(logits, labels)
    term.backward()
    # the EOS column must receive gradient at non-pad positions
    g = logits.grad
    mask = (labels != -100)
    assert g[..., eos][mask].abs().sum() > 0


def test_length_loss_changes_when_eos_prob_changes():
    logits, labels, eos = _toy()
    ll = LengthAwareLoss(eos_id=eos, length_loss_lambda=1.0)
    base = ll.length_term(logits.detach().clone(), labels).item()

    boosted = logits.detach().clone()
    boosted[..., eos] += 5.0  # push lots of EOS mass -> expected length drops
    changed = ll.length_term(boosted, labels).item()
    assert abs(changed - base) > 1e-4


def test_lambda_zero_is_plain_nll():
    logits, labels, eos = _toy()
    a = LengthAwareLoss(eos_id=eos, length_loss_lambda=0.0)(logits, labels)
    b = LengthAwareLoss(eos_id=eos, length_loss_lambda=0.0, use_length_loss=False)(logits, labels)
    nll = LengthAwareLoss(eos_id=eos, length_loss_lambda=0.3).nll(logits, labels)
    assert torch.allclose(a, nll) and torch.allclose(b, nll)
