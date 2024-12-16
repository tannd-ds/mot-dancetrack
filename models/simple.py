import numpy as np
import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config = config

    def forward(self, x):
        raise NotImplementedError

    def generate(self, conditions, img_w, img_h, **kwargs):
        cond_encodeds = []
        for i in range(len(conditions)):
            tmp_c = torch.tensor(np.array(conditions[i]), dtype=torch.float)
            # pad the condition to the interval
            if len(tmp_c) != self.config['interval']:
                pad_conds = tmp_c[-1].repeat((self.config['interval'] - len(tmp_c), 1))
                tmp_c = torch.cat((tmp_c, pad_conds), dim=0)
            cond_encodeds.append(tmp_c.unsqueeze(0))

        cond_encodeds = torch.cat(cond_encodeds)
        cond_encodeds[:, :, 0::2] = cond_encodeds[:, :, 0::2] / img_w
        cond_encodeds[:, :, 1::2] = cond_encodeds[:, :, 1::2] / img_h
        track_pred = self.forward(cond_encodeds.to("cuda"))
        return track_pred.cpu().detach().numpy()


class SimpleDiffMOTModel(BaseModel):
    def __init__(self, config):
        super(SimpleDiffMOTModel, self).__init__(config)

        self.fc1 = nn.Linear(self.config['interval'] * 8, 128) # Input feature size (batch_size, interval, 8) with 8 stands for (x, y, w, h, delta_x, delta_y, delta_w, delta_h)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 256)
        self.fc7 = nn.Linear(256, 128)
        self.fc8 = nn.Linear(128, 64)
        self.fc9 = nn.Linear(64, 4) # Output delta_bbox (batch_size, 4)

    def forward(self, conditions):
        batch_size = conditions.size(0)
        x = conditions.view(batch_size, -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        x = torch.relu(self.fc8(x))
        delta_bbox = self.fc9(x)
        return delta_bbox


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len=500):
        super().__init__()
        self.encoding = nn.Parameter(torch.zeros(1, max_len, hidden_dim))

    def forward(self, seq_len):
        return self.encoding[:, :seq_len, :]


class TransformerDiffMOTModel(BaseModel):
    def __init__(self, config):
        super(TransformerDiffMOTModel, self).__init__(config)

        # TransformerDecoderLayer setup
        self.embedding_dim = 8  # Input feature size (x, y, w, h, delta_x, delta_y, delta_w, delta_h)
        self.num_heads = 4
        self.hidden_dim = 256
        self.num_layers = 4
        self.dropout_rate = 0.1

        self.input_projection = nn.Linear(self.embedding_dim, self.hidden_dim)  # Make input into higher dim
        self.dropout = nn.Dropout(self.dropout_rate)

        self.encoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 2,
            dropout=self.dropout_rate,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(self.encoder_layer, num_layers=self.num_layers)

        # Positional encoding: learnable encoding
        self.positional_encoding = LearnablePositionalEncoding(self.hidden_dim)

        # Fully connected layers for output
        self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.fc2 = nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4)
        self.fc3 = nn.Linear(self.hidden_dim // 4, 4)  # Output delta_bbox (batch_size, 4)

        # Batch normalization for regularization
        self.batch_norm1 = nn.BatchNorm1d(self.hidden_dim // 2)

    def forward(self, conditions):
        # conditions: (batch_size, 4, 8)
        batch_size, seq_len, _ = conditions.size()

        # Project input to hidden dimensions
        conditions = self.input_projection(conditions)
        conditions = self.dropout(conditions)  # Apply dropout

        # Add positional encoding
        positional_enc = self.positional_encoding(seq_len)  # Match sequence length
        conditions = conditions + positional_enc

        # Split into query and memory
        query = conditions[:, -1:, :]  # Last step as query (batch_size, 1, hidden_dim)
        memory = conditions[:, :-1, :]  # Remaining steps as memory (batch_size, seq_len-1, hidden_dim)

        # Decode using Transformer
        output = self.transformer_decoder(query, memory)  # Decode conditions
        output = output[:, -1, :]  # Use the last position
        output = self.fc1(output)
        output = self.batch_norm1(output)
        output = torch.relu(output)
        output = self.dropout(output)
        output = self.fc2(output)
        output = torch.relu(output)
        output = self.fc3(output)
        return output
