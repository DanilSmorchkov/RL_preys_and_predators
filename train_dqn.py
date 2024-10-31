import numpy as np
import torch
from tqdm import tqdm

from src.DQN import DQN, evaluate_policy
from src.utils import calculate_reward
from src.preprocess import preprocess_data
from src.options import INITIAL_STEPS, TRANSITIONS

from world.envs import OnePlayerEnv
from world.map_loaders.single_team import SingleTeamLabyrinthMapLoader, SingleTeamRocksMapLoader
from world.realm import Realm

MIN_REWARD = 70

np.random.seed(1337)
torch.manual_seed(1337)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = OnePlayerEnv(Realm(SingleTeamLabyrinthMapLoader(), 1))
dqn = DQN(state_dim=256, 
          action_dim=5, 
          save_path="/home/RL_course_Predators_and_Preys/best_rocks/", 
          load_path="/home/RL_course_Predators_and_Preys/best_all/")
eps = 0.1
state, info = env.reset()
dqn.reset(state, info)
processed_state = preprocess_data(state, info)
for _ in tqdm(range(INITIAL_STEPS)):
    action = np.random.randint(0, 5, size=(5,))
    next_state, done, next_info = env.step(action)
    next_processed_state = preprocess_data(next_state, next_info)
    reward = calculate_reward(processed_state, next_processed_state, info, next_info, action)
    dqn.consume_transition(
        (
            processed_state,
            action,
            next_processed_state,
            reward,
            done,
        )
    )

    if done:
        state, info = env.reset()
        dqn.reset(state, info)
        processed_state = preprocess_data(state, info)
    else:
        processed_state = next_processed_state.copy()
        info = next_info.copy()

state, info = env.reset()
dqn.reset(state, info)
processed_state = preprocess_data(state, info)

for i in tqdm(range(TRANSITIONS)):
    # Epsilon-greedy policy
    if np.random.rand() < eps:
        action = np.random.randint(0, 5, size=(5,))
    else:
        action = dqn.act_preprocessed(processed_state)

    next_state, done, next_info = env.step(action)
    next_processed_state = preprocess_data(next_state, next_info)

    reward = calculate_reward(processed_state, next_processed_state, info, next_info, action)

    dqn.update(
        (
            processed_state,
            action,
            next_processed_state,
            reward,
            done,
        )
    )

    if done:
        state, info = env.reset()
        dqn.reset(state, info)
        processed_state = preprocess_data(state, info)
    else:
        processed_state = next_processed_state.copy()
        info = next_info.copy()

    if (i + 1) % (TRANSITIONS // 100) == 0:
        rewards = evaluate_policy(dqn, env, episodes=3)
        print(f"Step: {i + 1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}")
        if np.mean(rewards) > MIN_REWARD:
            dqn.save()
            MIN_REWARD = np.mean(rewards)