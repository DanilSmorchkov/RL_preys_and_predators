import multiprocessing as mp
import torch

from world.envs import OnePlayerEnv
from world.realm import Realm
from src.A3C import A3C
from world.map_loaders.single_team import SingleTeamLabyrinthMapLoader
from src.options import TRANSITIONS
from utils import calculate_reward
from src.options import GAMMA, STEPS_PER_UPDATE
from src.utils import record
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
        self.local_env = OnePlayerEnv(Realm(SingleTeamLabyrinthMapLoader(), 1))
        self.device = device

    def run(self):
        total_step = 1
        while self.global_episode.value < TRANSITIONS:
            state, info = self.local_env.reset()
            states_buffer, info_buffer, actions_buffer, rewards_buffer = [], [], [], []
            while True:
                # if self.name == "w00":
                #     self.env.render()
                state = torch.tensor(state).unsqueeze(0).permute(0, 3, 1, 2).float()
                additional_info = (
                    torch.tensor(np.array([(agent["x"], agent["y"]) for agent in info["predators"]]))
                    .unsqueeze(0)
                    .float()
                )
                action = self.local_network.act(state.to(self.device), additional_info.to(self.device))
                next_state, done, next_info = self.local_env.step(action)

                reward = calculate_reward(info, next_info)

                states_buffer.append(state)
                info_buffer.append(additional_info)
                actions_buffer.append(torch.tensor(action).unsqueeze(0))
                rewards_buffer.append(reward)

                if total_step % STEPS_PER_UPDATE == 0 or done:  # update global and assign to local net
                    # sync
                    self.push_and_pull(
                        done,
                        next_state,
                        next_info,
                        states_buffer,
                        info_buffer,
                        actions_buffer,
                        rewards_buffer,
                    )
                    states_buffer, info_buffer, actions_buffer, rewards_buffer = [], [], [], []

                    if done:  # done and print information
                        record(
                            self.global_episode,
                            self.global_episode_reward,
                            next_info["scores"][0],
                            self.res_queue,
                            self.name,
                        )
                        break

                if done:
                    break
                else:
                    state = next_state.copy()
                    info = next_info.copy()

        self.res_queue.put(None)

    def push_and_pull(
        self,
        done,
        next_state,
        next_info,
        states_buffer,
        info_buffer,
        actions_buffer,
        rewards_buffer,
    ):
        if done:
            R = 0.0  # terminal
        else:
            next_state = torch.tensor(next_state).unsqueeze(0).permute(0, 3, 1, 2).float().to(self.device)
            next_info = (
                torch.tensor(np.array([(agent["x"], agent["y"]) for agent in next_info["predators"]]))
                .unsqueeze(0)
                .float()
                .to(self.device)
            )
            R = self.local_network.forward(next_state, next_info)[-1].numpy()

        discounted_rewards = []
        for reward in rewards_buffer[::-1]:  # reverse buffer r
            R = reward + GAMMA * R
            discounted_rewards.append(torch.tensor(R).unsqueeze(0))
        discounted_rewards.reverse()
        discounted_rewards = torch.cat(discounted_rewards)
        # todo: tensors
        loss = self.local_network.loss_func(
            torch.cat(states_buffer).to(self.device),
            torch.cat(info_buffer).to(self.device),
            torch.cat(actions_buffer).to(self.device),
            discounted_rewards.to(self.device),
        )

        # calculate local gradients and push local parameters to global
        self.opt.zero_grad()
        loss.backward()
        for lp, gp in zip(self.local_network.parameters(), self.global_network.parameters()):
            gp._grad = lp.grad
        self.opt.step()

        # pull global parameters
        self.local_network.load_state_dict(self.global_network.state_dict())
