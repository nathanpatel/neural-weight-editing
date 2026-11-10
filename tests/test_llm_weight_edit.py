import pytest
import torch

from llm_weight_edit import apply_patch, parse_response, validate_patch
from model import PolicyNetwork


def valid_patch() -> dict[str, object]:
    return {
        "changes": [
            {"parameter": "network.2.weight", "index": [3, 4], "delta": 0.1},
            {"parameter": "network.2.bias", "index": [1], "delta": -0.1},
        ],
        "reason": "A test patch.",
    }


def test_valid_patch_changes_only_permitted_entries() -> None:
    model = PolicyNetwork()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    patch = validate_patch(valid_patch(), model)
    edited, metrics = apply_patch(model, patch)

    assert metrics["changed_parameters"] == 2
    assert torch.equal(model.network[-1].weight, before["network.2.weight"])
    assert torch.equal(model.network[-1].bias, before["network.2.bias"])
    assert edited.network[-1].weight[3, 4].item() == pytest.approx(
        before["network.2.weight"][3, 4].item() + 0.1
    )


@pytest.mark.parametrize(
    "change",
    [
        {"parameter": "network.0.weight", "index": [0, 0], "delta": 0.1},
        {"parameter": "network.2.weight", "index": [4, 32], "delta": 0.1},
        {"parameter": "network.2.weight", "index": [3, 4], "delta": 0.1},
    ],
)
def test_invalid_parameter_index_or_duplicate_is_rejected(change: dict[str, object]) -> None:
    patch = {"changes": [change, change], "reason": "invalid"}
    with pytest.raises(ValueError):
        validate_patch(patch, PolicyNetwork())


def test_absolute_limit_is_rejected() -> None:
    patch = {"changes": [{"parameter": "network.2.bias", "index": [1], "delta": 0.251}], "reason": "large"}
    with pytest.raises(ValueError):
        validate_patch(patch, PolicyNetwork())


def test_change_count_and_l2_limits_are_rejected() -> None:
    too_many = [
        {"parameter": "network.2.weight", "index": [row, column], "delta": 0.01}
        for row in range(4)
        for column in range(5)
    ]
    with pytest.raises(ValueError):
        validate_patch({"changes": too_many, "reason": "too many"}, PolicyNetwork())

    too_large = [
        {"parameter": "network.2.weight", "index": [row, column], "delta": 0.25}
        for row, column in [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)]
    ]
    with pytest.raises(ValueError):
        validate_patch({"changes": too_large, "reason": "large"}, PolicyNetwork())


def test_empty_response_is_recorded_as_empty() -> None:
    response = type("Response", (), {"output_text": ""})()
    assert parse_response(response) == (None, "")


def test_non_object_response_is_rejected() -> None:
    response = type("Response", (), {"output_text": "[]"})()
    with pytest.raises(ValueError, match="not an object"):
        parse_response(response)


def test_invalid_json_response_is_rejected() -> None:
    response = type("Response", (), {"output_text": "not json"})()
    with pytest.raises(ValueError, match="invalid JSON"):
        parse_response(response)
