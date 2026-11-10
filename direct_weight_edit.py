"""Apply and evaluate one analytical final-layer weight edit."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from evaluate_baseline import evaluate, summarize
from gridworld import GridWorld, rollout, scenario_from_seed
from model import PolicyNetwork


def apply_analytical_edit(
    model: PolicyNetwork,
    observation: torch.Tensor,
    *,
    target_action: int,
    competing_action: int,
    margin: float = 0.1,
) -> dict[str, object]:
    """Make the smallest symmetric final-layer edit reaching a target margin."""
    if observation.ndim != 1:
        raise ValueError("observation must be a one-dimensional vector")
    if target_action == competing_action:
        raise ValueError("target_action and competing_action must differ")
    if margin <= 0:
        raise ValueError("margin must be positive")

    final_layer = model.network[-1]
    if not isinstance(final_layer, torch.nn.Linear):
        raise TypeError("the final network layer must be linear")

    with torch.no_grad():
        hidden = model.network[:-1](observation)
        before_scores = final_layer(hidden).clone()
        current_margin = float(before_scores[target_action] - before_scores[competing_action])
        required_change = max(0.0, margin - current_margin)

        # Append 1 so the same vector updates both the final weights and bias.
        feature = torch.cat((hidden, hidden.new_ones(1)))
        feature_squared_norm = float(torch.dot(feature, feature))
        update = (required_change / (2.0 * feature_squared_norm)) * feature

        final_layer.weight[target_action].add_(update[:-1])
        final_layer.bias[target_action].add_(update[-1])
        final_layer.weight[competing_action].sub_(update[:-1])
        final_layer.bias[competing_action].sub_(update[-1])

        after_scores = final_layer(hidden).clone()
        changed_parameters = int(torch.count_nonzero(update).item() * 2)
        parameter_l2_change = float((2.0 * torch.dot(update, update)).sqrt())
        maximum_change = float(update.abs().max())
        preferred_action = int(after_scores.argmax().item())

    return {
        "target_action": target_action,
        "competing_action": competing_action,
        "requested_margin": margin,
        "before_scores": before_scores.tolist(),
        "after_scores": after_scores.tolist(),
        "target_margin_after_edit": float(
            after_scores[target_action] - after_scores[competing_action]
        ),
        "preferred_action_after_edit": preferred_action,
        "target_is_highest": preferred_action == target_action,
        "changed_parameters": changed_parameters,
        "maximum_parameter_change": maximum_change,
        "parameter_l2_change": parameter_l2_change,
    }


def load_model(checkpoint_path: Path) -> PolicyNetwork:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = PolicyNetwork()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def evaluate_edit(original: PolicyNetwork, edited: PolicyNetwork) -> dict[str, object]:
    target_scenario = scenario_from_seed(30_000, difficulty="difficult")
    target_observation_environment = GridWorld(target_scenario)
    target_observation_environment.position = (5, 0)
    target_observation = torch.from_numpy(target_observation_environment.observation())
    edit = apply_analytical_edit(
        edited,
        target_observation,
        target_action=3,
        competing_action=1,
    )

    def choose_action(model: PolicyNetwork):
        return lambda observation: int(model(torch.from_numpy(observation)).argmax().item())

    original_target = rollout(target_scenario, choose_action(original))
    edited_target = rollout(target_scenario, choose_action(edited))
    original_standard = evaluate(original, "standard", 20_000, 50)
    edited_standard = evaluate(edited, "standard", 20_000, 50)
    original_difficult = evaluate(original, "difficult", 30_000, 30)
    edited_difficult = evaluate(edited, "difficult", 30_000, 30)

    return {
        "edit": edit,
        "target_scenario": {
            "original": original_target,
            "edited": edited_target,
        },
        "ordinary_held_out": {
            "original": summarize(original_standard),
            "edited": summarize(edited_standard),
        },
        "difficult_held_out_including_target": {
            "original": summarize(original_difficult),
            "edited": summarize(edited_difficult),
        },
        "difficult_held_out_excluding_target": {
            "original": summarize([result for result in original_difficult if result["seed"] != 30_000]),
            "edited": summarize([result for result in edited_difficult if result["seed"] != 30_000]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/baseline.pt"))
    parser.add_argument("--edited-checkpoint", type=Path, default=Path("outputs/direct_edit.pt"))
    parser.add_argument("--report", type=Path, default=Path("outputs/direct_edit_evaluation.json"))
    args = parser.parse_args()

    original = load_model(args.checkpoint)
    edited = copy.deepcopy(original)
    original_state = {
        name: parameter.detach().clone()
        for name, parameter in original.state_dict().items()
    }
    report = evaluate_edit(original, edited)
    report["original_model_unchanged"] = all(
        torch.equal(parameter, original_state[name])
        for name, parameter in original.state_dict().items()
    )
    args.edited_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": edited.state_dict(), "edit": report["edit"]}, args.edited_checkpoint)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"Saved edited checkpoint to {args.edited_checkpoint}")
    print(f"Saved evaluation report to {args.report}")


if __name__ == "__main__":
    main()
