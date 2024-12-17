import torch
import torch.nn as nn

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


class LargeCNNPredictor(BasePositionPredictor):
    """
    Enhanced CNN model with dilated convolutions, residual connections, and attention.
    Input: conditions (batch_size, interval, 8)
    Output: delta_bbox (batch_size, 4)
    """

    def __init__(self, config):
        super(LargeCNNPredictor, self).__init__(config)

        self.config = config
        input_channels = 8  # Number of features at each time step

        # Convolutional blocks
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)

        # Dilated convolution block
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, dilation=2, padding=2)

        # Residual block
        self.residual = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1),
            nn.Sigmoid()
        )

        # Dynamic pooling
        self.pool_max = nn.AdaptiveMaxPool1d(1)
        self.pool_avg = nn.AdaptiveAvgPool1d(1)

        # Fully connected layers for bounding box prediction
        self.fc1 = nn.Linear(256, 64)  # 128 from max pooling + 128 from avg pooling
        self.fc2 = nn.Linear(64, 4)  # Output: delta_bbox

    def forward(self, conditions):
        # Reshape input: (batch_size, interval, 8) -> (batch_size, 8, interval)
        x = conditions.permute(0, 2, 1)

        # Initial convolutions
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))

        # Dilated convolution with residual connection
        residual = self.residual(x)  # Match dimensions
        x = torch.relu(self.conv3(x) + residual)

        # Attention mechanism
        attention_weights = self.attention(x)
        x = x * attention_weights  # Apply attention

        # Dynamic pooling
        x_max = self.pool_max(x).squeeze(-1)  # (batch_size, 128)
        x_avg = self.pool_avg(x).squeeze(-1)  # (batch_size, 128)
        x = torch.cat([x_max, x_avg], dim=1)  # (batch_size, 256)

        # Fully connected layers
        x = torch.relu(self.fc1(x))
        delta_bbox = self.fc2(x)

        return delta_bbox


import torch
import torch.nn as nn


class LargerCNNBBoxPredictor(BasePositionPredictor):
    """
    Larger CNN model with additional layers, dense connections, and increased width.
    Input: conditions (batch_size, interval, 8)
    Output: delta_bbox (batch_size, 4)
    """

    def __init__(self, config):
        super(LargerCNNBBoxPredictor, self).__init__(config)

        self.config = config
        input_channels = 8  # Number of features at each time step

        # Convolutional layers
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)

        # Additional convolutional layers
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(512, 512, kernel_size=3, padding=2, dilation=2)  # Dilated convolution

        # Pooling layers
        self.pool1 = nn.MaxPool1d(kernel_size=2)  # Reduce temporal dimension
        self.pool2 = nn.AvgPool1d(kernel_size=2)

        # Dense connections
        self.dense1 = nn.Conv1d(128, 128, kernel_size=1)  # Reduce channels for dense skip connections
        self.dense2 = nn.Conv1d(512, 128, kernel_size=1)

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # Reduce temporal dimension to 1

        # Fully connected layers
        self.fc1 = nn.Linear(768, 256)  # Combine dense + global features
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 4)  # Output: delta_bbox

    def forward(self, conditions):
        # Reshape input: (batch_size, interval, 8) -> (batch_size, 8, interval)
        x = conditions.permute(0, 2, 1)

        # Convolutional block 1
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool1(x)  # Temporal reduction

        # Dense connection 1
        dense1 = self.dense1(x)  # (batch_size, 128, temporal_dim)

        # Convolutional block 2
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.pool2(x)  # Further temporal reduction

        # Dense connection 2
        dense2 = self.dense2(x)  # (batch_size, 128, temporal_dim)

        # Dilated convolution for larger receptive field
        x = torch.relu(self.conv5(x))

        # Global pooling
        x = self.global_pool(x).squeeze(-1)  # (batch_size, 512)

        # Concatenate features from dense connections and global pooling
        x = torch.cat([x, dense1.mean(dim=2), dense2.mean(dim=2)], dim=1)  # (batch_size, 768)

        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        delta_bbox = self.fc3(x)

        return delta_bbox


if __name__ == '__main__':
    model = LargeCNNPredictor({})
    print(model)

    print('Number of parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))