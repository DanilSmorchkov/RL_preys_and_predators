import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding)
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
            ConvBlock(in_channels=2, out_channels=64, stride=1),
            ResConvBlock(in_channels=64),
            ConvBlock(in_channels=64, out_channels=64, stride=2),
            ResConvBlock(in_channels=64),
            ConvBlock(in_channels=64, out_channels=64, stride=2),
            ResConvBlock(in_channels=64),
            nn.Flatten(),
        )
        self.linear = nn.Linear(10 * 10 * 64, 256)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.linear(x)
        return x


class RLPreprocessor(nn.Module):
    def __init__(self):
        super(RLPreprocessor, self).__init__()
        self.ImagePreprocessor = ImagePreprocessor()
        self.linear = nn.Linear(258, 256)

    def forward(self, image: torch.Tensor, addition_features: torch.Tensor) -> torch.Tensor:
        # addtional_features.shape [N_batch, 5, 2]
        # image_features.shape [N_batch, 256] -> [N_batch, 5, 256]
        # N_batch, n_agents, state_dim
        image_features = self.ImagePreprocessor(image)
        image_features = image_features.unsqueeze(1).repeat(1, 5, 1)
        image_features = torch.cat((image_features, addition_features), -1)
        return self.linear(image_features)


if __name__ == "__main__":
    preprocessor = ImagePreprocessor()
    x = torch.randn((1, 40, 40, 2))
    with torch.no_grad():
        result = preprocessor(x.permute(0, 3, 1, 2))
    print(result.shape)
