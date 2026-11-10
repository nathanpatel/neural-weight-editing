import torch

from direct_weight_edit import apply_analytical_edit
from model import PolicyNetwork


def test_edit_changes_only_target_and_competing_final_rows() -> None:
    model = PolicyNetwork()
    final_layer = model.network[-1]
    with torch.no_grad():
        final_layer.weight.zero_()
        final_layer.bias.zero_()
        final_layer.bias[1] = 2.0
    before_weights = final_layer.weight.detach().clone()
    before_bias = final_layer.bias.detach().clone()

    report = apply_analytical_edit(
        model,
        torch.zeros(3 * 7 * 7),
        target_action=3,
        competing_action=1,
    )

    assert report["target_is_highest"] is True
    assert report["changed_parameters"] == 66
    assert torch.equal(final_layer.weight[[0, 2]], before_weights[[0, 2]])
    assert torch.equal(final_layer.bias[[0, 2]], before_bias[[0, 2]])
    assert not torch.equal(final_layer.weight[[1, 3]], before_weights[[1, 3]])
    assert final_layer.bias[3] > before_bias[3]
    assert final_layer.bias[1] < before_bias[1]


def test_edit_reaches_requested_margin() -> None:
    model = PolicyNetwork()
    report = apply_analytical_edit(
        model,
        torch.ones(3 * 7 * 7),
        target_action=3,
        competing_action=1,
        margin=0.25,
    )

    assert report["target_margin_after_edit"] >= 0.25 - 1e-5
