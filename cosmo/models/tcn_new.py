import torch
import torch.nn as nn
import torch.nn.functional as F

from cosmo.models.Base import BasePositionPredictor


# class CausalConv1d(nn.Conv1d):
#     def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
#         self.padding = (kernel_size - 1) * dilation
#         super().__init__(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation, **kwargs)
#
#     def forward(self, x):
#         x = super().forward(x)
#         return x[:, :, :-self.padding] if self.padding != 0 else x

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, dropout=0.2):
        super(CausalConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding='same',
            dilation=dilation)
        self.weight_norm = nn.utils.parametrizations.weight_norm(self.conv)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.weight_norm(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, residual_channels, skip_channels, kernel_size, dilation):
        super(ResidualBlock, self).__init__()
        self.dilated_conv = CausalConv1d(
            in_channels=residual_channels,
            out_channels=2 * residual_channels,
            kernel_size=kernel_size,
            dilation=dilation
        )
        self.residual_out = nn.Conv1d(residual_channels, residual_channels, kernel_size=1)
        self.skip_out = nn.Conv1d(residual_channels, skip_channels, kernel_size=1)

    def forward(self, x):
        conv_out = self.dilated_conv(x)
        gate, filter = torch.chunk(conv_out, 2, dim=1)
        activation = torch.tanh(filter) * torch.sigmoid(gate)

        residual = self.residual_out(activation)
        skip = self.skip_out(activation)

        return (x + residual) * 0.707, skip

class DilatedCausalConvNet(BasePositionPredictor):
    def __init__(self, config, in_channels, residual_channels, skip_channels, out_channels, kernel_size, num_blocks, num_layers):
        super(DilatedCausalConvNet, self).__init__(config)
        self.input_conv = CausalConv1d(in_channels, residual_channels, kernel_size=1)

        self.blocks = nn.ModuleList()
        for b in range(num_blocks):
            for l in range(num_layers):
                dilation = 2 ** l
                self.blocks.append(ResidualBlock(residual_channels, skip_channels, kernel_size, dilation))

        self.output_conv1 = nn.Conv1d(skip_channels, skip_channels, kernel_size=1)
        self.output_conv2 = nn.Conv1d(skip_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.input_conv(x)

        skip_connections = []
        for block in self.blocks:
            x, skip = block(x)
            skip_connections.append(skip)

        skip_sum = torch.sum(torch.stack(skip_connections), dim=0)
        x = F.relu(skip_sum)
        x = F.relu(self.output_conv1(x))
        x = self.output_conv2(x)

        x = torch.mean(x, dim=-1)  # Global average pooling along time dimension

        return x

# Example usage
if __name__ == "__main__":
    model = DilatedCausalConvNet(
        in_channels=1,
        residual_channels=32,
        skip_channels=32,
        out_channels=4,  # Regression to a 4D vector
        kernel_size=2,
        num_blocks=2,
        num_layers=4
    )

    input_tensor = torch.randn(8, 1, 64)  # Batch size 8, 1 input channel, sequence length 64
    output = model(input_tensor)
    print(output.shape)  # Expected: [8, 4]
