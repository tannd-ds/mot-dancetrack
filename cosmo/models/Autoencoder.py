
import torch
import torch.nn as nn
import torch.nn.functional as F

from .Base import BasePositionPredictor

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


class VAEPositionPredictor(BasePositionPredictor):
    """
    Variational Autoencoder-based model for generating latent features and predicting delta_bbox.
    Input: conditions (batch_size, interval, 8)
    Output: delta_bbox (batch_size, 4)
    """

    def __init__(self, config):
        super(VAEPositionPredictor, self).__init__(config)

        self.config = config
        input_dim = self.config['interval'] * 8  # Flattened input

        # Encoder: Compress the input and learn mean and variance
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(128, 64)  # Latent space mean
        self.fc_logvar = nn.Linear(128, 64)  # Latent space log variance

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

    @staticmethod
    def reparameterize(mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) using N(0, 1).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, conditions):
        batch_size = conditions.size(0)

        # Flatten input
        x = conditions.view(batch_size, -1)

        # Encode to latent space
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # Decode (reconstruct the input)
        reconstruction = self.decoder(z)

        # Predict delta_bbox
        delta_bbox = self.bbox_predictor(z)

        return reconstruction, delta_bbox, mu, logvar

    @staticmethod
    def loss_function(reconstruction, conditions, mu, logvar):
        """
        Compute the VAE loss: Reconstruction + KL Divergence.
        """
        # Reconstruction loss (MSE or BCE)
        recon_loss = F.mse_loss(reconstruction, conditions.view(reconstruction.size()), reduction='sum')

        # KL Divergence: D_KL(q(z|x) || p(z)) where q is the learned distribution and p is N(0,1)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return recon_loss + kl_loss
