from .Base import BasePositionPredictor

import torch
import torch.nn as nn


class AutoEncoderPositionPredictor(BasePositionPredictor):
    """
    Autoencoder-based model for learning latent features and predicting delta_bbox.
    Input: conditions (batch_size, interval, 8)
    Output: delta_bbox (batch_size, 4)
    """

    def __init__(self, config):
        super(AutoEncoderPositionPredictor, self).__init__(config)

        self.config = config
        input_dim = self.config['interval'] * 8  # Flattened input

        # Encoder: Compress the input
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)  # Latent space dimension
        )

        # Decoder: Reconstruct the input
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()  # Assuming input is normalized
        )

        # Bounding Box Predictor: Predict delta_bbox from latent space
        self.bbox_predictor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # Output: delta_bbox
        )

    def forward(self, conditions):
        batch_size = conditions.size(0)

        # Flatten input
        x = conditions.view(batch_size, -1)

        # Encode to latent space
        latent = self.encoder(x)

        # Decode (reconstruct the input)
        reconstruction = self.decoder(latent)

        # Predict delta_bbox
        delta_bbox = self.bbox_predictor(latent)

        return reconstruction, delta_bbox
