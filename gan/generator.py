"""
Generator — WGAN-GP
Input : noise vector (latent_dim,)
Output: fake DDoS traffic sample (input_dim,)
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int, hidden_dims: list = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 256, 256, 128]

        layers = []
        prev_dim = latent_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.LeakyReLU(0.2),
            ]
            prev_dim = h

        # Output layer — không activation vì data đã được StandardScaler
        # (range roughly -3 to 3, tanh sẽ clip mất info)
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

        # Weight init
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z):
        """
        z: (batch, latent_dim)
        returns: (batch, output_dim)
        """
        return self.net(z)

    def sample(self, n: int, device: torch.device):
        """Utility: sample n fake DDoS records"""
        z = torch.randn(n, self.net[0].in_features, device=device)
        with torch.no_grad():
            return self(z)