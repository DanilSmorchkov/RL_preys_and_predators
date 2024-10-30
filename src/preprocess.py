import torch
import torch.nn as nn
import numpy as np
from queue import Queue

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
            ConvBlock(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            ResConvBlock(16),
            ResConvBlock(16),
            ConvBlock(in_channels=16, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2),
            # ResConvBlock(64),
            nn.Flatten()
        )
        self.linear = nn.Sequential(nn.Linear(20 * 20 * 4, 256))

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

def get_distance_mask(centered_obstacles_mask, source=(20, 20)):
    queue = Queue()
    queue.put(source)

    distance_mask = np.empty_like(centered_obstacles_mask, dtype=np.float32)
    distance_mask.fill(np.nan)
    distance_mask[source[1], source[0]] = 0

    while not queue.empty():
        x, y = queue.get()

        for nx, ny in get_adjacent_cells(x, y, centered_obstacles_mask, distance_mask):
            queue.put((nx, ny))
            distance_mask[ny, nx] = distance_mask[y, x] + 1

    distance_mask[np.isnan(distance_mask)] = -1
    distance_mask = distance_mask / distance_mask.max()
    distance_mask[distance_mask < 0] = 2
    return distance_mask

def get_adjacent_cells(x, y, obstacles_mask, distance_mask):
    """Yields adjacent cells to (x, y) that are not obstacles and have not been visited"""
    n, m = obstacles_mask.shape
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nx = (x + dx) % m if x + dx >= 0 else m - 1
        ny = (y + dy) % n if y + dy >= 0 else n - 1
        if obstacles_mask[ny, nx] != 1 and np.isnan(distance_mask[ny, nx]):
            yield (nx, ny)

def preprocess_data(state: np.ndarray, info: dict, count_distance = True) -> torch.Tensor:
    state = torch.tensor(state)

    hunters_coordinates = np.array([(agent["y"], agent["x"]) for agent in info["predators"]])
    prey_id = state[:,:, 0].max()
    prey_mask = (state[:,:, 0] == prey_id).to(int)
    hunter_mask = (state[:,:, 0] == 0).to(int)
    wall_mask = ((state[:,:, 0] == -1) * (state[:,:, 1] == -1)).to(int)
    # bonuses_mask = ((state[:,:, 0] == -1) * (state[:,:, 1] == 1)).to(int)
    
    states = []
    for hunter_coordinates in hunters_coordinates:
        centred_coods = 20 - hunter_coordinates[0], 20 - hunter_coordinates[1]
        obst_mask = torch.roll(wall_mask, centred_coods, dims=[0, 1])
        hunter_state = torch.stack(
            [
                torch.roll(prey_mask, centred_coods, dims=[0, 1]),
                torch.roll(hunter_mask, centred_coods, dims=[0, 1]),
                obst_mask,
                torch.tensor(get_distance_mask(obst_mask) if count_distance else torch.zeros_like(obst_mask)),
                
            ]
                )
        states.append(hunter_state.float())
    
    return torch.stack(states)

if __name__ == "__main__":
    preprocessor = ImagePreprocessor()
    x = torch.randn((1, 40, 40, 2))
    with torch.no_grad():
        result = preprocessor(x.permute(0, 3, 1, 2))
    print(result.shape)
