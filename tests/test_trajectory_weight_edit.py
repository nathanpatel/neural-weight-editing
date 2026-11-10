import torch

from model import PolicyNetwork
from trajectory_weight_edit import (
    analytical_patch,
    build_prompt,
    collect_repair_states,
)


def test_repair_states_are_fixed_and_distinct() -> None:
    states = collect_repair_states(PolicyNetwork())

    assert len({state["index"] for state in states}) == len(states)
    assert len({tuple(state["position"]) for state in states}) == len(states)
    assert all(state["chosen_action"] != state["expert_action"] for state in states)
    assert all(
        state["chosen_invalid"] or state["chosen_remaining_distance"] > state["expert_remaining_distance"]
        for state in states
    )


def test_prompt_contains_repair_information_but_no_evaluation_results() -> None:
    prompt = build_prompt(PolicyNetwork())

    assert "REPAIR TRAJECTORY INFORMATION" in prompt
    assert "ordinary" not in prompt.lower()
    assert "held-out" not in prompt.lower()
    assert "analytical solution" not in prompt.lower()
    assert "An empty patch is valid" in prompt


def test_analytical_patch_is_bounded_and_preserves_original() -> None:
    model = PolicyNetwork()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    states = collect_repair_states(model)
    result = analytical_patch(model, states)
    changes = result["patch"]["changes"]

    assert set(result["active_states"]) == {state["index"] for state in states}
    assert len(changes) <= 16
    assert max(abs(change["delta"]) for change in changes) <= 0.25
    assert sum(change["delta"] ** 2 for change in changes) ** 0.5 <= 0.61 + 1e-9
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())
