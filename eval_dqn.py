import numpy as np
import torch

from pathlib import Path
from src.DQN import DQN, evaluate_policy

from world.envs import OnePlayerEnv
from world.map_loaders.single_team import SingleTeamLabyrinthMapLoader, SingleTeamRocksMapLoader
from world.realm import Realm

np.random.seed(1337)
torch.manual_seed(1337)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = OnePlayerEnv(Realm(SingleTeamLabyrinthMapLoader(), 1))
dqn = DQN(state_dim=256, 
          action_dim=5, 
          save_path="/home/RL_course_Predators_and_Preys/best_rocks/")

rewards = evaluate_policy(dqn, env, episodes=1, do_render=True)

import imageio
with imageio.get_writer('/home/Episode_1/0_movie.gif', mode='I', duration=0.5) as writer:
    for filename in sorted(list(Path("/home/Episode_1/").glob("*.png")), key=lambda x: int(x.stem)):
        print(filename)
        image = imageio.imread(filename)
        writer.append_data(image) # type: ignore

print(f"Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}")