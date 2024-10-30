import multiprocessing as mp
import torch

from world.envs import OnePlayerEnv
from world.realm import Realm
from src.A3C import A3C
from world.map_loaders.single_team import SingleTeamLabyrinthMapLoader
from src.options import TRANSITIONS
from src.utils import calculate_reward
from src.options import GAMMA, STEPS_PER_UPDATE
from src.utils import record
from src.preprocess import preprocess_data
import numpy as np


class Worker(mp.Process):
    def __init__(
        self,
        global_network: torch.nn.Module,
        opt: torch.optim.Optimizer,
        global_episode: mp.Value,
        global_episode_reward: mp.Value,
        res_queue: mp.Queue,
        name:int,
        device: torch.device,
    ):
        super(Worker, self).__init__()
        self.name = "w%02i" % name
        self.global_episode, self.global_episode_reward, self.res_queue = (
            global_episode,
            global_episode_reward,
            res_queue,
        )
        self.global_network, self.opt = global_network, opt
        self.local_network = A3C().to(device)  # local network
        self.local_network.load_state_dict(self.global_network.state_dict())
        self.local_env = OnePlayerEnv(Realm(SingleTeamLabyrinthMapLoader(), 1))
        self.device = device

    def run(self):
        total_step = 1
        while self.global_episode.value < TRANSITIONS:
            state, info = self.local_env.reset()
            states_buffer, actions_buffer, rewards_buffer = [], [], []
            total_reward = np.zeros(5)
            while True:
                processed_state = preprocess_data(state, info)
                action = self.local_network.act(processed_state.to(self.device))
                next_state, done, next_info = self.local_env.step(action)

                reward = calculate_reward(processed_state, 
                                          preprocess_data(next_state, next_info, count_distance=False), 
                                          info, 
                                          next_info, 
                                          action)

                total_reward+=reward

                states_buffer.append(processed_state)
                # info_buffer.append(additional_info)
                actions_buffer.append(torch.tensor(action).unsqueeze(0))
                rewards_buffer.append(reward)

                if total_step % STEPS_PER_UPDATE == 0 or done:  # update global and assign to local net
                    # sync
                    self.push_and_pull(
                        next_state,
                        next_info,
                        states_buffer,
                        actions_buffer,
                        rewards_buffer,
                    )
                    

                    if done:  # done and print information
                        record(
                            self.global_episode,
                            self.global_episode_reward,
                            next_info["scores"][0],
                            total_reward,
                            self.res_queue,
                            self.name,
                        )
                        break
                    states_buffer, actions_buffer, rewards_buffer = [], [], []
                state = next_state.copy()
                info = next_info.copy()
                total_step += 1

        self.res_queue.put(None)

    def push_and_pull(
        self,
        next_state,
        next_info,
        states_buffer,
        actions_buffer,
        rewards_buffer,
    ):
        next_state = preprocess_data(next_state, next_info).to(self.device)
            
        with torch.no_grad():
            R = self.local_network.forward(next_state)[-1].cpu().numpy()

        discounted_rewards = []
        for reward in rewards_buffer[::-1]:  # reverse buffer r
            R = reward + GAMMA * R
            discounted_rewards.append(torch.tensor(R))
        discounted_rewards.reverse()
        # todo: tensors
        loss = self.local_network.loss_func(
            torch.cat(states_buffer).to(self.device),
            torch.cat(actions_buffer).to(self.device),
            torch.cat(discounted_rewards).to(self.device),
        )

        # calculate local gradients and push local parameters to global
        self.opt.zero_grad()
        loss.backward()
        for lp, gp in zip(self.local_network.parameters(), self.global_network.parameters()):
            gp._grad = lp.grad
        self.opt.step()

        # pull global parameters
        self.local_network.load_state_dict(self.global_network.state_dict())
