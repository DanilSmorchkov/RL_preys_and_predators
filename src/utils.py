"""
Functions that use multiple times
"""

from collections import defaultdict

import torch
import numpy as np
from world.utils import RenderedEnvWrapper

my_hash = defaultdict(int)


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3):
        super(SharedAdam, self).__init__(params, lr=lr)
        # State initialization
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p.data)
                state["exp_avg_sq"] = torch.zeros_like(p.data)

                # share in memory
                state["exp_avg"].share_memory_()
                state["exp_avg_sq"].share_memory_()


def calculate_reward(old_info, new_info) -> np.ndarray:
    # eaten словарь из пойманных существ.
    # Ключи - номер команды и индекс пойманного существа,
    # значение - номер команды и индекс существа, которое его поймало
    team_indices = [predator["id"] for predator in new_info["predators"]]
    team_coordinates = np.array([(predator["x"], predator["y"]) for predator in new_info["predators"]])
    old_team_coordinates = np.array([(predator["x"], predator["y"]) for predator in old_info["predators"]])
    preys_coordinates = np.array([(prey["x"], prey["y"]) for prey in new_info["preys"] if prey["alive"]])

    eaten = np.array(list(new_info["eaten"].values()))
    # if not len(eaten):
    #     return np.zeros(5)
    rewards = []

    for hunter_index, hunter_position, old_hunter_position in zip(team_indices, team_coordinates, old_team_coordinates):
        distance_to_closest_prey = np.linalg.norm(hunter_position[None, :] - preys_coordinates, axis=1).min()
        # todo: add eaten hunter handling
        if len(eaten):
            eaten_by_current_hunter = eaten[(eaten == (0, hunter_index)).all(axis=1)].shape[0]
        else:
            eaten_by_current_hunter = 0
        my_hash[tuple(hunter_position)] += 1
        rewards.append(
            10 * eaten_by_current_hunter
            + 5 / (distance_to_closest_prey + 1e-9)
            + 3 * (my_hash[tuple(hunter_position)]) ** (-1 / 2)
        )
    return np.array(rewards)


def record(global_ep, global_ep_r, episode_score, res_queue, name):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        global_ep_r.value = episode_score
    res_queue.put(global_ep_r.value)
    print(
        name,
        "Ep:",
        global_ep.value,
        "| Ep_r: %.0f" % global_ep_r.value,
        # "| Ep_score: %.0f" % episode_score,
    )


def evaluate_policy(agent, env, device, episodes=5):
    env = RenderedEnvWrapper(env)
    returns = []
    for i in range(episodes):
        done = False
        state, info = env.reset()
        total_reward = np.zeros(5)

        while not done:
            state = torch.tensor(state).unsqueeze(0).permute(0, 3, 1, 2).float()
            additional_info = (
                torch.tensor(np.array([(agent["x"], agent["y"]) for agent in info["predators"]])).unsqueeze(0).float()
            )
            state, done, new_info = env.step(agent.act(state.to(device), additional_info.to(device)))

            reward = calculate_reward(info, new_info)

            total_reward += reward

            info = new_info.copy()
        env.render(f"../Episode_{i+1}")
        returns.append(total_reward.sum())
    return returns
