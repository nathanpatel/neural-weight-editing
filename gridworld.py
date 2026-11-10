"""A small deterministic gridworld and shortest-path expert."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from random import Random

import numpy as np

Action = int
Position = tuple[int, int]

UP: Action = 0
DOWN: Action = 1
LEFT: Action = 2
RIGHT: Action = 3
ACTIONS: tuple[tuple[Action, Position], ...] = (
    (UP, (-1, 0)),
    (DOWN, (1, 0)),
    (LEFT, (0, -1)),
    (RIGHT, (0, 1)),
)


@dataclass(frozen=True)
class Scenario:
    """A fixed layout used for training or evaluation."""

    size: int
    start: Position
    goal: Position
    obstacles: frozenset[Position]
    seed: int
    difficulty: str


class GridWorld:
    """A deterministic environment for one scenario."""

    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario
        self.position = scenario.start

    def reset(self) -> np.ndarray:
        self.position = self.scenario.start
        return self.observation()

    def observation(self) -> np.ndarray:
        """Encode obstacles, robot position and goal as three grid channels."""
        size = self.scenario.size
        encoded = np.zeros((3, size, size), dtype=np.float32)
        for row, column in self.scenario.obstacles:
            encoded[0, row, column] = 1.0
        encoded[1, self.position[0], self.position[1]] = 1.0
        encoded[2, self.scenario.goal[0], self.scenario.goal[1]] = 1.0
        return encoded.reshape(-1)

    def step(self, action: Action) -> tuple[np.ndarray, bool, bool]:
        """Apply an action and return observation, success, and whether it was invalid."""
        if action not in range(len(ACTIONS)):
            raise ValueError(f"Unknown action: {action}")

        _, (row_delta, column_delta) = ACTIONS[action]
        proposed = (
            self.position[0] + row_delta,
            self.position[1] + column_delta,
        )
        invalid = (
            not (0 <= proposed[0] < self.scenario.size)
            or not (0 <= proposed[1] < self.scenario.size)
            or proposed in self.scenario.obstacles
        )
        if not invalid:
            self.position = proposed
        return self.observation(), self.position == self.scenario.goal, invalid


def shortest_path_actions(scenario: Scenario) -> list[Action]:
    """Return one shortest valid action sequence, or raise if none exists."""
    queue: deque[Position] = deque([scenario.start])
    previous: dict[Position, tuple[Position, Action] | None] = {scenario.start: None}

    while queue:
        position = queue.popleft()
        if position == scenario.goal:
            break
        for action, (row_delta, column_delta) in ACTIONS:
            next_position = (position[0] + row_delta, position[1] + column_delta)
            if (
                0 <= next_position[0] < scenario.size
                and 0 <= next_position[1] < scenario.size
                and next_position not in scenario.obstacles
                and next_position not in previous
            ):
                previous[next_position] = (position, action)
                queue.append(next_position)

    if scenario.goal not in previous:
        raise ValueError("Scenario has no path from start to goal")

    actions: list[Action] = []
    position = scenario.goal
    while previous[position] is not None:
        prior_position, action = previous[position]
        actions.append(action)
        position = prior_position
    actions.reverse()
    return actions


def scenario_from_seed(
    seed: int,
    *,
    size: int = 7,
    difficulty: str = "standard",
) -> Scenario:
    """Create a reproducible layout, retaining only layouts with a valid path."""
    if difficulty not in {"standard", "difficult"}:
        raise ValueError("difficulty must be 'standard' or 'difficult'")

    obstacle_count = 5 if difficulty == "standard" else 14
    random = Random(seed)
    start = (0, 0)
    goal = (size - 1, size - 1)
    candidates = [
        (row, column)
        for row in range(size)
        for column in range(size)
        if (row, column) not in {start, goal}
    ]

    # Retry with a derived seed so every requested seed produces a valid layout.
    for attempt in range(1000):
        random.seed(seed + attempt * 1_000_003)
        obstacles = frozenset(random.sample(candidates, obstacle_count))
        scenario = Scenario(size, start, goal, obstacles, seed, difficulty)
        try:
            shortest_path_actions(scenario)
        except ValueError:
            continue
        return scenario
    raise RuntimeError(f"Could not create a valid {difficulty} scenario for seed {seed}")


def collect_expert_examples(scenarios: list[Scenario]) -> tuple[np.ndarray, np.ndarray]:
    """Collect observations and shortest-path actions for supervised training."""
    observations: list[np.ndarray] = []
    actions: list[Action] = []
    for scenario in scenarios:
        environment = GridWorld(scenario)
        for action in shortest_path_actions(scenario):
            observations.append(environment.observation())
            actions.append(action)
            _, success, invalid = environment.step(action)
            assert not invalid
            if success:
                break
    return np.stack(observations), np.asarray(actions, dtype=np.int64)


def rollout(
    scenario: Scenario,
    choose_action,
    *,
    max_steps: int | None = None,
) -> dict[str, object]:
    """Run a policy and return metrics plus its visited positions."""
    environment = GridWorld(scenario)
    environment.reset()
    limit = max_steps or scenario.size * scenario.size * 2
    positions = [environment.position]
    invalid_moves = 0
    success = False
    for step in range(1, limit + 1):
        action = int(choose_action(environment.observation()))
        _, success, invalid = environment.step(action)
        invalid_moves += int(invalid)
        positions.append(environment.position)
        if success:
            break
    return {
        "seed": scenario.seed,
        "difficulty": scenario.difficulty,
        "success": success,
        "steps": step,
        "invalid_moves": invalid_moves,
        "positions": positions,
    }
