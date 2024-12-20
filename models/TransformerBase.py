import torch
import torch.nn as nn

from .Base import BasePositionPredictor

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len=500):
        super().__init__()
        self.encoding = nn.Parameter(torch.zeros(1, max_len, hidden_dim))

    def forward(self, seq_len):
        return self.encoding[:, :seq_len, :]


class MLP(nn.Module):
  def __init__(self, in_features, out_features, dropout = 0.1):
    super(MLP, self).__init__()
    self.layer_norm = nn.LayerNorm(in_features)
    self.dropout = nn.Dropout(dropout)
    self.dense_layer = nn.Sequential(
        nn.Linear(in_features, in_features * 2),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(in_features * 2, in_features),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(in_features, out_features)
    )

  def forward(self, x):
    normalized_x = self.layer_norm(x)
    ffn_x = self.dense_layer(normalized_x)
    output_x = self.dropout(ffn_x)
    return output_x


class TransformerPositionPredictor(BasePositionPredictor):
    def __init__(self, config):
        super(TransformerPositionPredictor, self).__init__(config)

        # TransformerDecoderLayer setup
        self.embedding_dim = 8  # Input feature size (x, y, w, h, delta_x, delta_y, delta_w, delta_h)
        self.num_heads = 4
        self.hidden_dim = 128
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
        # self.fc1 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        # self.fc2 = nn.Linear(self.hidden_dim // 2, 4)  # Output delta_bbox (batch_size, 4)
        self.predictor = MLP(self.hidden_dim, 4, self.dropout_rate)

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
        # output = self.fc1(output)
        # output = self.batch_norm1(output)  # Apply batch normalization
        # output = torch.relu(output)  # Apply non-linearity
        # output = self.dropout(output)  # Additional dropout
        # output = self.fc2(output)  # Final output layer
        output = self.predictor(output)
        return output
