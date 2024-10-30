import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.preprocess import RLPreprocessor, ImagePreprocessor

class A3C(nn.Module):
    def __init__(self):
        super(A3C, self).__init__()
        self.s_dim = 256
        self.a_dim = 5

        self.processor = RLPreprocessor()

        # Actor
        self.pi1 = nn.Linear(self.s_dim, 256)
        self.pi2 = nn.Linear(256, self.a_dim)
        self.distribution = torch.distributions.Categorical

        # Critic
        self.v1 = nn.Linear(self.s_dim, 256)
        self.v2 = nn.Linear(256, 1)
        self.set_init([self.pi1, self.pi2, self.v1, self.v2])

    @staticmethod
    def set_init(layers: list[nn.Module]):
        for layer in layers:
            nn.init.normal_(layer.weight, mean=0.0, std=0.1)
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, img):
        # Get state repr
        x = self.processor(img)

        # Actor inference
        pi1 = torch.tanh(self.pi1(x))
        logits = self.pi2(pi1)

        # Critic values
        v1 = torch.tanh(self.v1(x))
        values = self.v2(v1)

        return logits, values.squeeze(-1)

    def act(self, img):
        self.eval()
        # if random.random() > 0.1:
        with torch.no_grad():
            logits, _ = self.forward(img)
        prob = F.softmax(logits, dim=-1).data
        m = self.distribution(prob)
        return m.sample()[0].cpu().numpy()
        # else:
        #     return np.random.randint(0, 5, size = (5,))

    def loss_func(
        self,
        states,
        actions,
        discounted_rewards,
    ):
        self.train()
        logits, critic_value_function_prediction = self.forward(states)
        advantage_function = discounted_rewards - critic_value_function_prediction
        critic_loss = advantage_function.pow(2)

        probs = F.softmax(logits, dim=-1)
        entropies = -(F.log_softmax(logits, dim=-1) * probs).sum(-1)

        current_distribution = self.distribution(probs)
        log_prob = current_distribution.log_prob(actions)
        a_loss = -log_prob * advantage_function.detach()

        total_loss = (a_loss - 0.01 * entropies + critic_loss).sum(dim=-1).mean()
        # print(total_loss.detach().sum() + 0.5 * entropies.detach().sum())
        return total_loss
