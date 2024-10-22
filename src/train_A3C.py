import torch
import multiprocessing as mp
import numpy as np
import seaborn as sns


from src.A3C import A3C
from src.utils import SharedAdam, evaluate_policy
from src.parallel import Worker

from world.envs import OnePlayerEnv
from world.realm import Realm
from world.map_loaders.single_team import SingleTeamLabyrinthMapLoader

import os

os.environ["OMP_NUM_THREADS"] = "1"

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gnet = A3C().to(device)  # global network
    gnet.share_memory()  # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=1e-4)  # global optimizer
    global_episode_counter, global_episode_rewards, results_queue = mp.Value("i", 0), mp.Value("d", 0.0), mp.Queue()
    # parallel training
    workers = [
        Worker(gnet, opt, global_episode_counter, global_episode_rewards, results_queue, i, device) for i in range(4)
    ]
    [w.start() for w in workers]
    res = []  # record episode reward to plot
    while True:
        r = results_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]

    plot = sns.lineplot(x=np.arange(len(res)), y=res)
    plot.get_figure().savefig('training.jpg')
    evaluate_policy(agent=gnet, env=OnePlayerEnv(Realm(SingleTeamLabyrinthMapLoader(), 1)), device=device, episodes=1)
