import torch
import torch.nn as nn
from torchvision.models import resnet18


class SimpleDiffMOTModel(nn.Module):
    def __init__(self):
        super(SimpleDiffMOTModel, self).__init__()
        # Fully connected layers for processing conditions
        self.fc1 = nn.Linear(9*8, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 1024)
        self.fc5 = nn.Linear(1024, 512)
        self.fc6 = nn.Linear(512, 256)
        self.fc7 = nn.Linear(256, 128)
        self.fc8 = nn.Linear(128, 64)
        self.fc9 = nn.Linear(64, 4)  # Output delta_bbox

    def forward(self, conditions):
        batch_size = conditions.size(0)
        x = conditions.view(batch_size, -1)  # Flatten the input
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        x = torch.relu(self.fc8(x))
        delta_bbox = self.fc9(x)
        return delta_bbox


import torch
import torch.nn as nn


class ResidualDiffMOTModel(nn.Module):
    def __init__(self):
        super(ResidualDiffMOTModel, self).__init__()

        self.layers = [128, 256, 512, 1024, 2048]
        self.encoders = [nn.Linear(self.layers[i], self.layers[i+1]) for i in range(len(self.layers)-1)]
        self.decoders = [nn.Linear(self.layers[i], self.layers[i-1]) for i in range(len(self.layers)-1, 0, -1)]
        self.final_fc = nn.Linear(128, 4)

        # Dropout layers to prevent overfitting
        self.dropout = nn.Dropout(0.3)

        # Linear layer to match dimensions for residuals if necessary
        self.match_fc1 = nn.Linear(9 * 8, self.layers[0])  # For the first residual match

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for i in range(len(self.encoders)):
            self.encoders[i] = self.encoders[i].to(self.device)
        for i in range(len(self.decoders)):
            self.decoders[i] = self.decoders[i].to(self.device)


    def forward(self, conditions):
        batch_size = conditions.size(0)
        x = conditions.view(batch_size, -1)  # Flatten the input
        x = self.match_fc1(x)

        residuals = []
        for i, encoder in enumerate(self.encoders):
            residual = x
            x = torch.relu(encoder(x))
            x = self.dropout(x)
            residuals.append(residual)

        for i, decoder in enumerate(self.decoders):
            x = torch.relu(decoder(x) + residuals[-i-1])
            x = self.dropout(x)

        delta_bbox = self.final_fc(x)
        return delta_bbox


        # batch_size = conditions.size(0)
        # x = conditions.view(batch_size, -1)  # Flatten the input
        #
        # # Residual connection for first block
        # residual = x
        # x = self.bn1(torch.relu(self.fc1(x)))
        # x = self.dropout(x)
        #
        #
        # # Passing through subsequent layers with batch normalization, activation, and dropout
        # x = self.bn2(torch.relu(self.fc2(x)))
        # x = self.dropout(x)
        #
        # x = self.bn3(torch.relu(self.fc3(x)))
        # x = self.dropout(x)
        #
        # x = self.bn4(torch.relu(self.fc4(x)))
        # x = self.dropout(x)
        #
        # x = self.bn5(torch.relu(self.fc5(x)))
        # x = self.dropout(x)
        #
        # x = self.bn6(torch.relu(self.fc6(x)))
        # x = self.dropout(x)
        #
        # x = self.bn7(torch.relu(self.fc7(x)))
        # x = self.dropout(x)
        #
        # x = self.bn8(torch.relu(self.fc8(x)))
        #
        # delta_bbox = self.fc9(x)  # No activation needed for output
        # return delta_bbox


class TransformerDiffMOTModel(nn.Module):
    def __init__(self):
        super(TransformerDiffMOTModel, self).__init__()

        # TransformerDecoderLayer setup
        self.embedding_dim = 8  # Input feature size (x, y, w, h, delta_x, delta_y, delta_w, delta_h)
        self.num_heads = 2
        self.hidden_dim = 64
        self.num_layers = 2

        self.encoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embedding_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(self.encoder_layer,
                                                         num_layers=self.num_layers)

        # Fully connected layer for output
        self.fc = nn.Linear(self.embedding_dim, 4)  # Output delta_bbox (batch_size, 4)

    def forward(self, conditions):
        # conditions: (batch_size, 4, 8)
        output = self.transformer_decoder(conditions, conditions)  # Decode conditions
        output = self.fc(output[:, -1, :])  # Use the last position (batch_size, 4)
        return output


class TransformerDiffMOTModelV2(nn.Module):
    def __init__(self):
        super(TransformerDiffMOTModelV2, self).__init__()

        # TransformerDecoderLayer setup
        self.embedding_dim = 8  # Input feature size (x, y, w, h, delta_x, delta_y, delta_w, delta_h)
        self.num_heads = 4
        self.hidden_dim = 256
        self.num_layers = 4

        self.input_projection = nn.Linear(self.embedding_dim, self.hidden_dim) # Make input into higher dim

        self.encoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim, nhead=self.num_heads, dim_feedforward=self.hidden_dim * 2, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(self.encoder_layer,
                                                         num_layers=self.num_layers)

        self.positional_encoding = nn.Parameter(torch.zeros(1, (4+5), self.hidden_dim)) # positional info

        # Fully connected layers for output
        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.fc2 = nn.Linear(self.hidden_dim // 2, 4)  # Output delta_bbox (batch_size, 4)

    def forward(self, conditions):
        # conditions: (batch_size, 4, 8)
        conditions = self.input_projection(conditions)  # Project input to hidden dimensions
        conditions = conditions + self.positional_encoding  # Add positional encoding

        output = self.transformer_decoder(conditions, conditions)  # Decode conditions
        output = torch.relu(self.fc1(output[:, -1, :]))  # Use the last position and apply a non-linearity
        output = self.fc2(output)  # Final output layer)
        return output


if __name__ == '__main__':
    model = TransformerDiffMOTModelV2().to('cuda')
    print(model)
    print('N parameters:', sum(p.numel() for p in model.parameters()))
    data = torch.randn(32, 9, 8).to('cuda')
    output = model(data)
    print(output.size())

