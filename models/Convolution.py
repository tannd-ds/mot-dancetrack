import torch
import torch.nn as nn
import torch.nn.functional as F

from models.Base import BasePositionPredictor


class CNNPositionPredictor(BasePositionPredictor):
    """
    CNN-based model for predicting bounding box deltas.
    Input: conditions (batch_size, interval, 8)
    Output: delta_bbox (batch_size, 4)
    """

    def __init__(self, config):
        super(CNNPositionPredictor, self).__init__(config)

        self.config = config
        input_channels = 8  # Number of features at each time step
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)  # Pooling to reduce temporal dimension to 1

        # Fully connected layers for bounding box prediction
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 4)  # Output: delta_bbox

    def forward(self, conditions):
        # Reshape input: (batch_size, interval, 8) -> (batch_size, 8, interval)
        x = conditions.permute(0, 2, 1)

        # Apply 1D convolutions
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        # Pool across the temporal dimension
        x = self.pool(x).squeeze(-1)  # Shape: (batch_size, 128)

        # Fully connected layers for bounding box prediction
        x = torch.relu(self.fc1(x))
        delta_bbox = self.fc2(x)

        return delta_bbox


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=(1, 3),
                      stride=1,
                      padding=(0, 1),
                      bias=False
                      ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=(1, 3),
                      stride=1,
                      padding=(0, 1),
                      bias=False
                      ),
            nn.BatchNorm2d(out_channels)
        )
        self.residual_connect = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.residual_connect = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.left(x)
        out = out + self.residual_connect(x)
        out = F.relu(out)

        return out


class ResNetPredictor(BasePositionPredictor):
    def __init__(self, config, ResBlock=ResidualBlock):
        super(ResNetPredictor, self).__init__(config)
        self.in_channels = 32
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=self.in_channels,
                               kernel_size=(1, 3),
                               stride=1,
                               padding=(0, 1))
        self.layer1 = self.make_layer(ResBlock, self.in_channels, 2, stride=1)
        self.layer2 = self.make_layer(ResBlock, 64, 2, stride=2)
        self.layer3 = self.make_layer(ResBlock, 128, 2, stride=2)
        self.layer4 = self.make_layer(ResBlock, 256, 2, stride=2)
        self.fc1 = nn.Linear(256 * 8, 64)
        self.fc2 = nn.Linear(64, 4)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, channels, stride))
            self.in_channels = channels
        return nn.Sequential(*layers)

    def forward(self, conditions):
        # x = conditions.permute(0, 2, 1) # (batch_size, interval, 8) -> (batch_size, 8, interval)
        x = conditions.view(-1, 1, 8, self.config['interval']) # (batch_size, 1, 8, interval)

        out = self.conv1(x) # (batch_size, 32, 8, interval)
        out = self.layer1(out) # (batch_size, 32, 8, interval)
        out = self.layer2(out) # (batch_size, 64, 8, interval)
        out = self.layer3(out) # (batch_size, 128, 8, interval)
        out = self.layer4(out) # (batch_size, 256, 8, interval)
        out = F.avg_pool2d(out, (1, 3)) # (batch_size, 256, 8, (interval-2)//3)
        out = out.view(out.size(0), -1) # (batch_size, 256 * 8 * (interval-2)//3)
        out = self.fc1(out)
        delta_bbox = self.fc2(out)
        return delta_bbox


class Conv2dPredictor(BasePositionPredictor):
    def __init__(self, config, input_dim=8):
        super(Conv2dPredictor, self).__init__(config)
        self.input_dim = input_dim
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=32,
                               kernel_size=(1, 3),
                               stride=1,
                               padding=(0, 1))
        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=(1, 3),
                               stride=1,
                               padding=(0, 1))
        self.conv3 = nn.Conv2d(in_channels=64,
                               out_channels=128,
                               kernel_size=(1, 3),
                               stride=1,
                               padding=(0, 1))
        self.conv4 = nn.Conv2d(in_channels=128,
                               out_channels=256,
                               kernel_size=(1, 3),
                               stride=1,
                               padding=0)
        self.conv5 = nn.Conv2d(in_channels=256,
                               out_channels=128,
                               kernel_size=(1, 3),
                               stride=1,
                               padding=0)

        self.fc1 = nn.Linear(self.input_dim * (self.config['interval']-4) * 128, 64)
        self.fc2 = nn.Linear(64, 4)

    def forward(self, conditions):
        x = conditions.view(-1, 1, self.input_dim, self.config['interval']) # (batch_size, 1, 8, interval)
        x = torch.relu(self.conv1(x)) # (batch_size, 32, 8, interval)
        x = torch.relu(self.conv2(x)) # (batch_size, 64, 8, interval)
        x = torch.relu(self.conv3(x)) # (batch_size, 128, 8, interval)
        x = torch.relu(self.conv4(x)) # (batch_size, 256, 8, interval-2)
        x = torch.relu(self.conv5(x)) # (batch_size, 128, 8, interval-4)
        x = x.view(x.size(0), -1) # (batch_size, 8 * (interval-4) * 128)
        x = torch.relu(self.fc1(x))
        delta_bbox = self.fc2(x)
        return delta_bbox
