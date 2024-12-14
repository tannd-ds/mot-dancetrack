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
            tmp_c = conditions[i]
            tmp_c = np.array(tmp_c)
            tmp_c[:, 0::2] = tmp_c[:, 0::2] / img_w
            tmp_c[:, 1::2] = tmp_c[:, 1::2] / img_h
            tmp_conds = torch.tensor(tmp_c, dtype=torch.float)
            if len(tmp_conds) != self.config['interval']:
                pad_conds = tmp_conds[-1].repeat((self.config['interval'], 1))
                tmp_conds = torch.cat((tmp_conds, pad_conds), dim=0)[:self.config['interval']]
            cond_encodeds.append(tmp_conds.unsqueeze(0))
        cond_encodeds = torch.cat(cond_encodeds)
        track_pred = self.forward(cond_encodeds.to("cuda"))
        return track_pred.cpu().detach().numpy()


class SimpleDiffMOTModel(BaseModel):
    def __init__(self, config):
        super(SimpleDiffMOTModel, self).__init__(config)

        self.fc1 = nn.Linear(self.config['interval'] * 8, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 1024)
        self.fc5 = nn.Linear(1024, 512)
        self.fc6 = nn.Linear(512, 256)
        self.fc7 = nn.Linear(256, 128)
        self.fc8 = nn.Linear(128, 64)
        self.fc9 = nn.Linear(64, 4)

    def forward(self, conditions):
        batch_size = conditions.size(0)
        x = conditions.view(batch_size, -1)
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



class TransformerDiffMOTModel(BaseModel):
    def __init__(self, config):
        super(TransformerDiffMOTModel, self).__init__(config)

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
