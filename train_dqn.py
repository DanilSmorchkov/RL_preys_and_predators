import numpy as np
import torch
from tqdm import tqdm

from src.DQN import DQN, evaluate_policy
from src.utils import calculate_reward
from src.options import INITIAL_STEPS, TRANSITIONS

from world.envs import OnePlayerEnv, VersusBotEnv
from world.map_loaders.single_team import SingleTeamLabyrinthMapLoader, SingleTeamRocksMapLoader
from world.realm import Realm
from world.map_loaders.two_teams import TwoTeamLabyrinthMapLoader, TwoTeamRocksMapLoader
from world.scripted_agents import Dummy, ClosestTargetAgent
MIN_REWARD = 0

np.random.seed(1337)
torch.manual_seed(1337)
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
env = VersusBotEnv(Realm(TwoTeamRocksMapLoader(), 2, bots={1: ClosestTargetAgent()}))
# env = OnePlayerEnv(Realm(SingleTeamRocksMapLoader(), 1))
dqn = DQN(embedding_size=256, 
          num_input_channels=6, 
          save_path="/home/RL_course_Predators_and_Preys/best_bot_vs_normal/", 
          load_path="/home/RL_course_Predators_and_Preys/best_bot_start/"
          )
eps = 0.1
state, info = env.reset()
dqn.reset(state, info)
processed_state = dqn.preprocess_data(state, info)
for _ in tqdm(range(INITIAL_STEPS)):
    action = np.random.randint(0, 5, size=(5,))
    next_state, done, next_info = env.step(action)
    next_processed_state = dqn.preprocess_data(next_state, next_info)
    reward = calculate_reward(processed_state, 
                              next_processed_state, 
                              info, 
                              next_info,
                              dqn.distance_map)
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
        processed_state = dqn.preprocess_data(state, info)
    else:
        processed_state = next_processed_state.copy()
        info = next_info.copy()

state, info = env.reset()
dqn.reset(state, info)
processed_state = dqn.preprocess_data(state, info)

for i in tqdm(range(TRANSITIONS)):
    # Epsilon-greedy policy
    if np.random.rand() < eps:
        action = np.random.randint(0, 5, size=(5,))
    else:
        action = dqn.act_preprocessed(processed_state)

    next_state, done, next_info = env.step(action)
    next_processed_state = dqn.preprocess_data(next_state, next_info)

    reward = calculate_reward(processed_state, 
                              next_processed_state, 
                              info, 
                              next_info, 
                              dqn.distance_map)

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
        processed_state = dqn.preprocess_data(state, info)
    else:
        processed_state = next_processed_state.copy()
        info = next_info.copy()

    if (i + 1) % (TRANSITIONS // 100) == 0:
        rewards, enemy_rewards = evaluate_policy(dqn, env, episodes=3)
        print(f"Step: {i + 1}, Reward mean: {np.mean(rewards)}, std: {np.std(rewards)}")
        print(f"Step: {i + 1}, Enemy reward mean: {np.mean(enemy_rewards)}, std: {np.std(enemy_rewards)}")
        if np.mean(rewards) > MIN_REWARD:
            dqn.save()
            MIN_REWARD = np.mean(rewards)