import itertools

import pytest
import torch

from same_budget_analytical_edit import allocate_magnitudes, optimal_patch
from model import PolicyNetwork


def objective_for_support(coefficients: list[float], support: tuple[int, ...]) -> float:
    magnitudes = allocate_magnitudes([abs(coefficients[index]) for index in support])
    return sum(abs(coefficients[index]) * magnitude for index, magnitude in zip(support, magnitudes))


def test_allocation_respects_limits_and_uses_l2_budget() -> None:
    magnitudes = allocate_magnitudes([1.0 - index * 0.01 for index in range(16)])

    assert all(0 < magnitude <= 0.25 for magnitude in magnitudes)
    assert sum(magnitude**2 for magnitude in magnitudes) == pytest.approx(0.61**2)


def test_selected_support_matches_exhaustive_small_problem() -> None:
    coefficients = [0.99, 0.81, 0.52, 0.31, 0.12]
    selected = tuple(index for index, _ in sorted(enumerate(coefficients), key=lambda pair: -pair[1])[:3])
    selected_objective = objective_for_support(coefficients, selected)
    exhaustive_objective = max(
        objective_for_support(coefficients, support)
        for support in itertools.combinations(range(len(coefficients)), 3)
    )

    assert selected_objective == pytest.approx(exhaustive_objective)


def test_optimal_patch_is_valid_and_does_not_change_original() -> None:
    model = PolicyNetwork()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    result = optimal_patch(model)

    changes = result["patch"]["changes"]
    assert len(changes) == 16
    assert max(abs(change["delta"]) for change in changes) <= 0.25
    assert sum(change["delta"] ** 2 for change in changes) ** 0.5 <= 0.61 + 1e-9
    assert all(change["parameter"] in {"network.2.weight", "network.2.bias"} for change in changes)
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())
