import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.act = nn.ReLU()
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
        self.act = nn.ReLU()
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
            ConvBlock(in_channels=2, out_channels=16, kernel_size=3, stride=2, padding=0),
            ResConvBlock(16),
            ResConvBlock(16),
            ResConvBlock(16),
            ResConvBlock(16),
            ConvBlock(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=0),
            ResConvBlock(32),
            ResConvBlock(32),
            ResConvBlock(32),
            ResConvBlock(32),
            ConvBlock(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=0),
            ResConvBlock(64),
            ResConvBlock(64),
            ResConvBlock(64),
            ResConvBlock(64),
            nn.Flatten()
        )
        self.linear = nn.Sequential(nn.Linear(4 * 4 * 64, 256))

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


if __name__ == "__main__":
    preprocessor = ImagePreprocessor()
    x = torch.randn((1, 40, 40, 2))
    with torch.no_grad():
        result = preprocessor(x.permute(0, 3, 1, 2))
    print(result.shape)
