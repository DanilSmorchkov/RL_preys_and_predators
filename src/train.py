import numpy as np
import torch

from DQN import DQN, evaluate_policy
from DQN import calculate_reward

from options import INITIAL_STEPS, TRANSITIONS

from world.envs import OnePlayerEnv
from world.map_loaders.single_team import SingleTeamLabyrinthMapLoader
from world.realm import Realm

np.random.seed(1337)
torch.manual_seed(1337)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = OnePlayerEnv(Realm(SingleTeamLabyrinthMapLoader(), 1))
dqn = DQN(state_dim=256, action_dim=5)
eps = 0.1
state, info = env.reset()

for _ in range(INITIAL_STEPS):
    action = np.random.randint(0, 5, size=(5,))

    next_state, done, next_info = env.step(action)
    reward = calculate_reward(next_info)
    dqn.consume_transition(
        (
            state,
            [(item["x"], item["y"]) for item in info["predators"]],
            action,
            next_state,
            [(item["x"], item["y"]) for item in next_info["predators"]],
            reward,
            done,
        )
    )

    if done:
        state, info = env.reset()
    else:
        state = next_state.copy()
        info = next_info.copy()

state, info = env.reset()

for i in range(TRANSITIONS):
    # Epsilon-greedy policy
    if np.random.rand() < eps:
        action = np.random.randint(0, 5, size=(5,))
    else:
        action = dqn.act(state, info)

    next_state, done, next_info = env.step(action)
    reward = calculate_reward(next_info)
    dqn.update(
        (
            state,
            [(item["x"], item["y"]) for item in info["predators"]],
            action,
            next_state,
            [(item["x"], item["y"]) for item in next_info["predators"]],
            reward,
            done,
        )
    )

    if done:
        state, info = env.reset()
    else:
        state = next_state.copy()
        info = next_info.copy()

    if (i + 1) % (TRANSITIONS // 100) == 0:
        rewards = evaluate_policy(dqn, env, episodes=5)
        print(f"Step: {i + 1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}")
