import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from collections import deque, namedtuple
import random

from preprocess import RLPreprocessor
from options import (
    BATCH_SIZE,
    INITIAL_STEPS,
    LEARNING_RATE,
    GAMMA,
    STEPS_PER_UPDATE,
    STEPS_PER_TARGET_UPDATE,
)


class DQNModel(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.data_processor = RLPreprocessor()
        self.layer1 = nn.Linear(n_observations, 256)
        # self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, n_actions)

    def forward(self, img, info):
        x = F.relu(self.data_processor(img, info))
        x = F.relu(self.layer1(x))
        # x = F.relu(self.layer2(x))
        return self.layer3(x)


Transition = namedtuple(
    typename="Transition", field_names=("state", "info", "action", "next_state", "next_info", "reward", "done")
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
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.steps = 0  # Do not change
        self.target_model = DQNModel(state_dim, action_dim).to(self.device)
        self.policy_model = DQNModel(state_dim, action_dim).to(self.device)
        self.target_model.load_state_dict(self.policy_model.state_dict())
        self.replay_buffer = ReplayMemory(INITIAL_STEPS)
        self.criterion = nn.MSELoss()
        self.optimizer = Adam(self.policy_model.parameters(), lr=LEARNING_RATE)

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
        state_batch = torch.tensor(np.array(batch.state)).permute(0, 3, 1, 2).to(torch.float)
        info_batch = torch.tensor(np.array(batch.info)).to(torch.float)
        action_batch = torch.tensor(np.array(batch.action), dtype=torch.int64).to(self.device)

        state_action_values = (
            self.policy_model(state_batch.to(self.device), info_batch.to(self.device))
            .gather(2, action_batch[:, None])
            .squeeze()
            .to(self.device)
        )

        next_state_batch = torch.tensor(np.array(batch.next_state))
        next_info_batch = torch.tensor(np.array(batch.next_info))
        reward_batch = torch.tensor(np.array(batch.reward)).to(self.device)
        done_batch = torch.tensor(np.array(batch.done))

        non_final_mask = torch.tensor(tuple(map(lambda s: not s, done_batch)), device=self.device, dtype=torch.bool)
        non_final_next_states = next_state_batch[~done_batch].permute(0, 3, 1, 2).to(torch.float).to(self.device)
        non_final_next_info = next_info_batch[~done_batch].to(torch.float).to(self.device)

        next_state_values = torch.zeros((BATCH_SIZE, 5), device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = (
                self.target_model(non_final_next_states, non_final_next_info).max(2).values
            )
        # Compute the expected Q values
        expected_state_action_values = reward_batch + GAMMA * next_state_values

        loss = self.criterion(state_action_values.to(torch.float), expected_state_action_values.to(torch.float))
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

    def update_target_network(self):
        # Update weights of a target Q-network here. You may use copy.deepcopy to do this or
        # assign a values of network parameters via PyTorch methods.
        self.target_model.load_state_dict(self.policy_model.state_dict())

    def act(self, state, info):
        # Compute an action. Do not forget to turn state to a Tensor and then turn an action to a numpy array.
        cur_state = torch.from_numpy(np.array(state)).permute(2, 0, 1).unsqueeze(0).to(torch.float).to(self.device)
        info = (
            torch.tensor(np.array([(agent["x"], agent["y"]) for agent in info["predators"]]))
            .unsqueeze(0)
            .to(torch.float)
            .to(self.device)
        )
        self.policy_model.eval()
        with torch.no_grad():
            act = self.policy_model(cur_state, info).argmax(dim=-1).squeeze().cpu().numpy()
        return act

    def update(self, transition):
        # You don't need to change this
        self.consume_transition(transition)
        if self.steps % STEPS_PER_UPDATE == 0:
            batch = self.sample_batch()
            self.train_step(batch)
        if self.steps % STEPS_PER_TARGET_UPDATE == 0:
            self.update_target_network()
        self.steps += 1

    def save(self):
        torch.save(self.policy_model.state_dict(), "agent.pkl")


def calculate_reward(new_info) -> int:
    # eaten словарь из пойманных существ.
    # Ключи - номер команды и индекс пойманного существа,
    # значение - номер команды и индекс существа, которое его поймало
    team_indices = [item["id"] for item in new_info["predators"]]
    cur = np.array(list(new_info["eaten"].values()))
    if not len(cur):
        return np.zeros(5)
    rewards = []
    for index in team_indices:
        # todo: add eaten hunter handling
        cur_eaten = cur[(cur == (0, index)).all(axis=1)]
        rewards.append(cur_eaten.shape[0])
    return np.array(rewards)


def evaluate_policy(agent, env, episodes=5):
    returns = []
    for _ in range(episodes):
        done = False
        state, info = env.reset()
        total_reward = np.zeros(5)

        while not done:
            state, done, info = env.step(agent.act(state, info))

            reward = calculate_reward(info)

            total_reward += reward
        returns.append(total_reward.sum())
    return returns
