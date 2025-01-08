import torch
import torch.nn as nn
import torch.nn.functional as F
from .Base import BasePositionPredictor
from .tcn_new import ResidualBlock, CausalConv1d

class TCNIter(BasePositionPredictor):
    def __init__(self, config, in_channels, residual_channels, skip_channels, out_channels,  kernel_size, num_blocks, num_layers):
        super(TCNIter, self).__init__(config)
        self.config = config
        self.blocks = nn.ModuleList()
        self.num_blocks = num_blocks

        for l in range(num_layers):
            dilation = 2 ** l
            self.blocks.append(ResidualBlock(residual_channels, skip_channels, kernel_size, dilation))

        self.input_conv = CausalConv1d(in_channels, residual_channels, kernel_size=1)
        self.output_conv1 = nn.Conv1d(skip_channels, skip_channels, kernel_size=1)
        self.output_conv2 = nn.Conv1d(skip_channels, out_channels, kernel_size=1)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.input_conv(x)

        skip_connections = []
        for n in range(self.num_blocks):
            for block in self.blocks:
                x, skip = block(x)
                skip_connections.append(skip)
        skip_sum = torch.sum(torch.stack(skip_connections), dim=0)
        combined = self.alpha * skip_sum + (1 - self.alpha) * x
        x = F.relu(combined)
        x = F.relu(self.output_conv1(x))
        x = self.output_conv2(x)

        x = torch.mean(x, dim=-1)  # Global average pooling along time dimension
        return x

if __name__ == '__main__':
    config = {}
    model = TCNIter(config=config,
                    in_channels=8,
                    residual_channels=64,
                    skip_channels=64,
                    out_channels=4,
                    kernel_size=config.get('kernel_size', 2),
                    num_blocks=config.get('num_blocks', 2),
                    num_layers=config.get('num_layers', 4))
