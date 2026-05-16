"""
Quick smoke test for the VISTA MDP environment.

This does not use the real viewshed yet.
It uses a fake viewshed so we can check that the MDP loop runs.

Note: this is a primitive version that always commits to the first valid action 
without scouting or considering future rewards.
"""

from __future__ import annotations

import numpy as np

from environment import CandidateViewpoint, VistaEnvironment


def fake_viewshed(viewpoint: CandidateViewpoint) -> np.ndarray:
    """
    Fake visibility function for testing.

    It returns a boolean visibility mask where:
        True  = visible
        False = not visible

    Each viewpoint reveals a different row and column.
    """

    map_shape = (10, 10)
    mask = np.zeros(map_shape, dtype=bool)

    row = viewpoint.viewpoint_id % map_shape[0]
    col = viewpoint.viewpoint_id % map_shape[1]

    mask[row, :] = True
    mask[:, col] = True

    return mask


def main() -> None:
    candidate_viewpoints = [
        CandidateViewpoint(viewpoint_id=0, x=0.0, y=0.0, travel_cost=1.0, computation_cost=1.0),
        CandidateViewpoint(viewpoint_id=1, x=1.0, y=1.0, travel_cost=1.0, computation_cost=1.0),
        CandidateViewpoint(viewpoint_id=2, x=2.0, y=2.0, travel_cost=1.0, computation_cost=1.0),
        CandidateViewpoint(viewpoint_id=3, x=3.0, y=3.0, travel_cost=1.0, computation_cost=1.0),
    ]

    env = VistaEnvironment(
        candidate_viewpoints=candidate_viewpoints,
        viewshed_function=fake_viewshed,
        map_shape=(10, 10),
        initial_budget=10.0,
        target_coverage_percentage=50.0,
        max_steps=10,
    )

    state = env.reset()

    done = False
    total_reward = 0.0

    print("Starting smoke test")
    print()

    while not done:
        valid_actions = env.get_valid_action_numbers()

        print("Valid actions:", valid_actions)

        # For now, choose the first valid non-stop action if possible.
        # STOP is usually the final action in our action list.
        action_number = valid_actions[0] #look at the list of legal actions and pick the first one.

        state, reward, done, info = env.step(action_number) #commit to the action.
        total_reward += reward

        print(f"Step: {state.steps_taken}")
        print(f"Chosen action number: {action_number}")
        print(f"Reward: {reward}")
        print(f"Total reward: {total_reward}")
        print(f"Coverage: {state.coverage_percentage:.2f}%")
        print(f"Remaining budget: {state.remaining_budget}")
        print(f"Selected viewpoints: {state.selected_viewpoints}")
        print(f"Reason: {info.get('reason')}")
        print()

    print("Smoke test finished")


if __name__ == "__main__":
    main()