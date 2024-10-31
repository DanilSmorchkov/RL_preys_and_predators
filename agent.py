import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.act = nn.LeakyReLU()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class ResConvBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.act = nn.LeakyReLU()
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        y = self.act(y)
        return x + y

class ImagePreprocessor(nn.Module):
    def __init__(self):
        super(ImagePreprocessor, self).__init__()
        # 40 40 2
        self.conv1 = nn.Sequential(
            ConvBlock(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),
            ResConvBlock(8),
            ResConvBlock(8),
            ResConvBlock(8),
            ConvBlock(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            ResConvBlock(16),
            ResConvBlock(16),
            ResConvBlock(16),
            ConvBlock(in_channels=16, out_channels=4, kernel_size=3, stride=1, padding=1),
            ResConvBlock(4),
            ResConvBlock(4),
            nn.Flatten()
        )
        self.linear = nn.Sequential(nn.Linear(40 * 40 * 4, 256))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.linear(x)
        return x


class RLPreprocessor(nn.Module):
    def __init__(self):
        super(RLPreprocessor, self).__init__()
        self.ImagePreprocessor = ImagePreprocessor()
        # self.linear = nn.Linear(256 + 2, 256)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # addtional_features.shape [N_batch, 5, 2]
        # image_features.shape [N_batch, 256] -> [N_batch, 5, 256]
        # N_batch, n_agents, state_dim
        image_features = self.ImagePreprocessor(image)
        return image_features.reshape(-1, 5, 256)

class DQNModel(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.data_processor = RLPreprocessor()
        self.layer3 = nn.Linear(256, n_actions)

    def forward(self, img):
        x = F.relu(self.data_processor(img))
        return self.layer3(x)

class Agent:
    def __init__(self) -> None:
        self.agent = DQNModel(256, 5)
        self.agent.load_state_dict(torch.load(__file__[:-8] + "/agent.pkl", map_location="cpu"))
        self.distance_map = None
    def get_actions(self, state, info):
        return list(self.act(state, info))

    def act(self, state, info):
        # Compute an action. Do not forget to turn state to a Tensor and then turn an action to a numpy array.
        cur_state = torch.tensor(self.preprocess_data(state, info)).to(torch.float)

        self.agent.eval()

        with torch.no_grad():
            act = self.agent(cur_state).argmax(dim=-1).squeeze().cpu().numpy()
        return act
    
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

    def preprocess_data(self, state: np.ndarray, info: dict) -> np.ndarray:
        state = np.array(state)

        hunters_coordinates = np.array([(agent["y"], agent["x"]) for agent in info["predators"]])
        prey_id = state[:,:, 0].max()
        prey_mask = (state[:,:, 0] == prey_id).astype(np.int64)
        hunter_mask = (state[:,:, 0] == 0).astype(np.int64)
        wall_mask = ((state[:,:, 0] == -1) * (state[:,:, 1] == -1)).astype(np.int64)
        # bonuses_mask = ((state[:,:, 0] == -1) * (state[:,:, 1] == 1)).to(int)
        
        states = []
        for hunter_coordinates in hunters_coordinates:
            centred_coods = 20 - hunter_coordinates[0], 20 - hunter_coordinates[1]
            obst_mask = np.roll(wall_mask, centred_coods, axis=(0, 1))
            distance_mask = np.roll(self.distance_map[hunter_coordinates[0]*40 + hunter_coordinates[1]].reshape(40, 40), centred_coods, axis=(0, 1))
            distance_mask = np.nan_to_num(distance_mask, nan=-1)
            distance_mask = distance_mask / distance_mask.max()
            distance_mask = np.where(distance_mask<0, 2, distance_mask)
            hunter_state = np.stack(
                (
                    np.roll(prey_mask, centred_coods, axis=(0, 1)),
                    np.roll(hunter_mask, centred_coods, axis=(0, 1)),
                    obst_mask,
                    distance_mask
                    ),
                    
            )
            states.append(hunter_state.astype(np.float64))
        
        return np.stack(states)