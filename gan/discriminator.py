"""
Discriminator (Critic) — WGAN-GP
WGAN dùng critic thay vì sigmoid discriminator.
KHÔNG dùng BatchNorm vì gradient penalty yêu cầu per-sample gradient.
Thay bằng LayerNorm hoặc không norm.
"""

import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h),
                nn.LayerNorm(h),      # LayerNorm thay vì BatchNorm
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2),
            ]
            prev_dim = h

        # Output: scalar score (không sigmoid — WGAN critic)
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: (batch, input_dim)
        returns: (batch,) — Wasserstein score
        """
        return self.net(x).squeeze(1)


def gradient_penalty(critic: Critic, real: torch.Tensor, fake: torch.Tensor,
                     device: torch.device, lambda_gp: float = 10.0) -> torch.Tensor:
    """
    Tính gradient penalty cho WGAN-GP.
    Interpolate giữa real và fake, enforce ||grad|| = 1.
    """
    batch_size = real.size(0)
    # Random interpolation coefficient
    alpha = torch.rand(batch_size, 1, device=device)
    alpha = alpha.expand_as(real)

    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    critic_interp = critic(interpolated)

    # Tính gradient của critic output với interpolated input
    grads = torch.autograd.grad(
        outputs=critic_interp,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_interp),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Flatten và tính norm
    grads = grads.view(batch_size, -1)
    grad_norm = grads.norm(2, dim=1)
    gp = lambda_gp * ((grad_norm - 1) ** 2).mean()
    return gp