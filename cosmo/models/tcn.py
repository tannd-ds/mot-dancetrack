import torch
import torch.nn as nn

from .Base import BasePositionPredictor

class TCNResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(TCNResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding='same',
            dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding='same',
            dilation=dilation
        )
        self.weight_norm1 = nn.utils.parametrizations.weight_norm(self.conv1)
        self.weight_norm2 = nn.utils.parametrizations.weight_norm(self.conv2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = x

        x = self.weight_norm1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.weight_norm2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        residual = self.downsample(residual)
        return x + residual


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout=0.2):
        """
        Temporal Convolutional Network (TCN).

        Args:
            num_inputs: Number of input channels.
            num_channels: List of output channels for each residual block.
            kernel_size: Kernel size for convolutional layers.
            dropout: Dropout rate.
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_blocks = len(num_channels)

        for i in range(num_blocks):
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            dilation = 2 ** i
            layers.append(
                TCNResidualBlock(in_channels, out_channels, kernel_size, dilation, dropout)
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCNPredictor(BasePositionPredictor):
    def __init__(self, config, num_inputs, num_channels, kernel_size, dropout=0.2):
        """
        Args:
            num_inputs: Number of input channels.
            num_channels: List of output channels for each residual block.
            kernel_size: Kernel size for convolutional layers.
            dropout: Dropout rate.
        """
        super(TCNPredictor, self).__init__(config)
        self.tcn = TemporalConvNet(num_inputs, num_channels, kernel_size, dropout)
        self.fc = nn.Linear(num_channels[-1], 4)

    def forward(self, x):
        x = torch.permute(x, [0, 2, 1])
        x = self.tcn(x)
        x = x.mean(dim=-1)
        x = self.fc(x)
        return x


# Example usage
if __name__ == "__main__":
    x = torch.randn(512, 100, 8) # (Batch_size, n_input_feat, interval)
    model = TCNPredictor(config={}, num_inputs=8, num_channels=[32, 64, 128, 256, 128], kernel_size=3, dropout=0.3)
    print('Parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    output = model(x)
    print(output.shape)  # Output shape should be (8, 4)

