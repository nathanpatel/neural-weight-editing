from gridworld import GridWorld, RIGHT, Scenario, collect_expert_examples, scenario_from_seed, shortest_path_actions


def test_scenario_generation_is_reproducible() -> None:
    assert scenario_from_seed(42) == scenario_from_seed(42)
    assert scenario_from_seed(42, difficulty="difficult") == scenario_from_seed(42, difficulty="difficult")


def test_obstacle_blocks_movement() -> None:
    scenario = Scenario(3, (0, 0), (2, 2), frozenset({(0, 1)}), 1, "standard")
    environment = GridWorld(scenario)
    _, success, invalid = environment.step(RIGHT)
    assert invalid
    assert not success
    assert environment.position == (0, 0)


def test_expert_path_reaches_goal() -> None:
    scenario = scenario_from_seed(11)
    environment = GridWorld(scenario)
    for action in shortest_path_actions(scenario):
        _, success, invalid = environment.step(action)
        assert not invalid
    assert success


def test_expert_examples_have_matching_lengths() -> None:
    observations, actions = collect_expert_examples([scenario_from_seed(1), scenario_from_seed(2)])
    assert observations.shape[0] == actions.shape[0]
    assert observations.shape[1] == 3 * 7 * 7
