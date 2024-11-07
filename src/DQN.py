import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam # type: ignore
from collections import deque, namedtuple
import random

from src.preprocess import RLPreprocessor, preprocess_data
from src.options import (
    BATCH_SIZE,
    INITIAL_STEPS,
    LEARNING_RATE,
    GAMMA,
    STEPS_PER_UPDATE,
)

from world.utils import RenderedEnvWrapper
random.seed(1337)

class DQNModel(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.data_processor = RLPreprocessor()
        self.layer3 = nn.Linear(256, n_actions)

    def forward(self, img):
        x = F.relu(self.data_processor(img))
        return self.layer3(x)


Transition = namedtuple(
    typename="Transition", field_names=("state", "action", "next_state", "reward", "done")
)


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args[0]))

    def sample(self, batch_size=BATCH_SIZE):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN:
    def __init__(self, state_dim, action_dim, save_path = "./", load_path = None):
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.steps = 0  # Do not change
        self.target_model = DQNModel(state_dim, action_dim).to(self.device)
        self.policy_model = DQNModel(state_dim, action_dim).to(self.device)
        self.target_model.load_state_dict(self.policy_model.state_dict())
        self.replay_buffer = ReplayMemory(INITIAL_STEPS)
        self.criterion = nn.HuberLoss()
        self.optimizer = Adam(self.policy_model.parameters(), lr=LEARNING_RATE)
        self.tau = 0.005
        self.save_path = save_path
        self.distance_map = None
        if load_path is not None:
            self.load_path = load_path
            self.load()

    def consume_transition(self, transition):
        # Add transition to a replay buffer.
        # Hint: use deque with specified maxlen. It will remove old experience automatically.
        self.replay_buffer.push(transition)

    def sample_batch(self):
        # Sample batch from a replay buffer.
        # Hints:
        # 1. Use random.randint
        # 2. Turn your batch into a numpy.array before turning it to a Tensor. It will work faster
        batch = self.replay_buffer.sample()
        batch = Transition(*zip(*batch))
        return batch

    def train_step(self, batch):
        self.policy_model.train()
        # Use batch to update DQN's network.
        state_batch = torch.tensor(np.concatenate(batch.state)).to(torch.float)
        action_batch = torch.tensor(np.array(batch.action), dtype=torch.int64).to(self.device)

        state_action_values = (
            self.policy_model(state_batch.to(self.device))
            .gather(dim=2, index=action_batch[..., None])
            .squeeze()
            .to(self.device)
        )

        next_state_batch = torch.tensor(np.concatenate(batch.next_state))
        reward_batch = torch.tensor(np.array(batch.reward)).to(self.device)
        done_batch = torch.tensor(np.array(batch.done))

        non_final_mask = torch.tensor(tuple(map(lambda s: not s, done_batch)), device=self.device, dtype=torch.bool)
        non_final_next_states = next_state_batch.reshape(-1, 5, 4, 40, 40)[~done_batch].reshape(-1, 4, 40, 40).to(torch.float).to(self.device)

        next_state_values = torch.zeros((BATCH_SIZE, 5), device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = (
                self.target_model(non_final_next_states).max(2).values
            )
        # Compute the expected Q values
        expected_state_action_values = reward_batch + GAMMA * next_state_values

        loss = self.criterion(state_action_values.to(torch.float), expected_state_action_values.to(torch.float))
        # print(loss.item())
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_model.parameters(), 50)
        self.optimizer.step()

    def soft_update_target_network(self):
        target_net_state_dict = self.target_model.state_dict()
        policy_net_state_dict = self.policy_model.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * \
                self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_model.load_state_dict(target_net_state_dict)

    def act(self, state, info):
        # Compute an action. Do not forget to turn state to a Tensor and then turn an action to a numpy array.
        cur_state = torch.tensor(preprocess_data(state, info)).to(torch.float).to(self.device)
        self.policy_model.eval()
        with torch.no_grad():
            act = self.policy_model(cur_state).argmax(dim=-1).squeeze().cpu().numpy()
        return act
    
    def act_preprocessed(self, state):
        # Compute an action. Do not forget to turn state to a Tensor and then turn an action to a numpy array.
        cur_state = torch.tensor(state).to(torch.float).to(self.device)
        self.policy_model.eval()
        with torch.no_grad():
            act = self.policy_model(cur_state).argmax(dim=-1).squeeze().cpu().numpy()
        return act

    def update(self, transition):
        # You don't need to change this
        self.consume_transition(transition)
        if self.steps % STEPS_PER_UPDATE == 0:
            batch = self.sample_batch()
            self.train_step(batch)
        self.soft_update_target_network()
        self.steps += 1
    
    def reset(self, initial_state, info):
        mask = np.zeros(initial_state.shape[:2], np.bool_)
        mask[np.logical_or(np.logical_and(initial_state[:, :, 0] == -1, initial_state[:, :, 1] >= 0),
                           initial_state[:, :, 0] >= 0)] = True
        mask = mask.reshape(-1)

        coords_amount = initial_state.shape[0] * initial_state.shape[1]
        self.distance_map = (coords_amount + 1) * np.ones((coords_amount, coords_amount))
        np.fill_diagonal(self.distance_map, 0.)
        self.distance_map[np.logical_not(mask)] = (coords_amount + 1)
        self.distance_map[:, np.logical_not(mask)] = (coords_amount + 1)

        indexes_helper = [
            [
                x * initial_state.shape[1] + (y + 1) % initial_state.shape[1],
                x * initial_state.shape[1] + (initial_state.shape[1] + y - 1) % initial_state.shape[1],
                ((initial_state.shape[0] + x - 1) % initial_state.shape[0]) * initial_state.shape[1] + y,
                ((x + 1) % initial_state.shape[0]) * initial_state.shape[1] + y
            ]
            for x in range(initial_state.shape[0]) for y in range(initial_state.shape[1])
        ]

        updated = True
        while updated:
            old_distances = self.distance_map.copy()
            for j in range(coords_amount):
                if mask[j]:
                    for i in indexes_helper[j]:
                        if mask[i]:
                            self.distance_map[j] = np.minimum(self.distance_map[j], self.distance_map[i] + 1)
            updated = (old_distances != self.distance_map).sum() > 0
        self.distance_map = np.where(self.distance_map==(coords_amount + 1), np.nan, self.distance_map)

    def save(self):
        torch.save(self.policy_model.state_dict(), self.save_path + "agent.pkl")
    
    def load(self):
        self.policy_model.load_state_dict(torch.load(self.load_path + "agent.pkl", map_location=self.device))
        self.target_model.load_state_dict(torch.load(self.load_path + "agent.pkl", map_location=self.device))

def evaluate_policy(agent, env, do_render=False, episodes=5):
    if do_render:
        env = RenderedEnvWrapper(env)
    scores = []
    for i in range(episodes):
        done = False
        state, info = env.reset()
        agent.reset(state, info)

        while not done:
            state, done, info = env.step(agent.act(state, info))
        if do_render:
            env.render(f"./Episode_{i+1}")
        scores.append(info["scores"][0])
    return scores