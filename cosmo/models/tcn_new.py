import torch
import torch.nn as nn
import torch.nn.functional as F

from Base import BasePositionPredictor


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, dropout=0.2):
        super(CausalConv1d, self).__init__()
        self.causal_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.causal_padding,
            dilation=dilation)
        self.conv = nn.utils.parametrizations.weight_norm(self.conv, name='weight')
        self.ln = nn.LayerNorm(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        if self.causal_padding != 0:
            x = x[:, :, :-self.causal_padding]
        x = x.permute(0, 2, 1)
        x = self.ln(x)
        x = x.permute(0, 2, 1)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, residual_channels, skip_channels, kernel_size, dilation):
        super(ResidualBlock, self).__init__()
        self.dilated_conv = CausalConv1d(
            in_channels=residual_channels,
            out_channels=2 * residual_channels, # the 2 is for the gate and filter
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
    
class TemporalAttention(nn.Module):
    def __init__(self, channels=4, hidden_dim=32, interval=None, use_prior_init=False):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(channels, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, 1, kernel_size=1)
        )
        self.use_prior_init = use_prior_init
        if use_prior_init:
            assert interval is not None, "interval must be provided if use_prior_init is True"
            prior = torch.arange(1, interval+1, 1, dtype=torch.float32)
            prior = prior / prior.sum()
            self.register_buffer("attn_prior", prior.log().view(1, 1, interval))  # [1, 1, interval]
        else:
            self.attn_prior = None

    def forward(self, x):
        # x: [batch, channels, interval]
        attn_scores = self.attn(x)  # [batch, 1, interval]
        if self.use_prior_init and self.attn_prior is not None:
            attn_scores = attn_scores + self.attn_prior  # broadcast add
        attn_weights = torch.softmax(attn_scores, dim=2)  # [batch, 1, interval]
        attended = torch.matmul(x, attn_weights.transpose(1, 2)).squeeze(-1)  # [batch, channels]
        return attended, attn_weights


class DilatedCausalConvNet(BasePositionPredictor):
    def __init__(self, config, in_channels, residual_channels, skip_channels, out_channels, kernel_size, num_blocks, num_layers, interval=None, use_temporal_attention=True, ta_hidden_dim=32, ta_use_prior_init=False):
        super(DilatedCausalConvNet, self).__init__(config)
        self.input_conv = CausalConv1d(in_channels, residual_channels, kernel_size=1)

        self.blocks = nn.ModuleList()
        for b in range(num_blocks):
            for l in range(num_layers):
                dilation = 2 ** l
                self.blocks.append(ResidualBlock(residual_channels, skip_channels, kernel_size, dilation))

        self.output_conv1 = nn.Conv1d(skip_channels, skip_channels, kernel_size=1)
        self.output_conv2 = nn.Conv1d(skip_channels, out_channels, kernel_size=1)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        if config.get('set_alpha', None) is not None:
            self.alpha.requires_grad = False
            self.alpha.data.fill_(config['set_alpha'])

        self.use_temporal_attention = use_temporal_attention
        if use_temporal_attention:
            assert interval is not None, "interval must be provided if using temporal attention"
            self.temporal_attention = TemporalAttention(
                channels=out_channels,
                hidden_dim=ta_hidden_dim,
                interval=interval,
                use_prior_init=ta_use_prior_init
            )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.input_conv(x)

        skip_connections = []
        for block in self.blocks:
            x, skip = block(x)
            skip_connections.append(skip)

        skip_sum = torch.sum(torch.stack(skip_connections), dim=0)
        combined = self.alpha * skip_sum + (1 - self.alpha) * x
        x = F.relu(combined)
        x = F.relu(self.output_conv1(x))
        x = self.output_conv2(x)  # [batch, out_channels, interval]

        if self.use_temporal_attention:
            x, attn_weights = self.temporal_attention(x)  # [batch, out_channels]
            return x
        else:
            # Optionally, you could use mean pooling if not using attention
            x = torch.mean(x, dim=-1)
            return x
    
if __name__ == "__main__":
    model = DilatedCausalConvNet(
        config={},
        in_channels=8,
        residual_channels=64,
        skip_channels=64,
        out_channels=4,
        kernel_size=2,
        num_blocks=2,
        num_layers=4,
        use_temporal_attention=True, # From this one, all arguments are for temporal attention
        interval=9, 
        ta_hidden_dim=32,
        ta_use_prior_init=False
    )
    print('Number of parameters:', sum(p.numel() for p in model.parameters()))

    input_tensor = torch.randn(16, 9, 8)  # Batch size 16, 9 interval, 8 sequence length
    output = model(input_tensor)
    print(output.shape)  # Expected: [16, 4, 1]