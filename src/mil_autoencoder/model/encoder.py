import torch
import typing as tp
from torch import nn


class Encoder(nn.Module):
    def __init__(
        self,
        img_size=32,
        downsamplings=3,
        latent_size=128,
        linear_hidden_size=256,
        down_channels=4,
    ):
        super().__init__()

        self.latent_size = latent_size
        self.downsamplings = downsamplings
        self.linear_hidden_size = linear_hidden_size
        self.down_channels = down_channels

        n_channels = self.down_channels
        hidden_dims = [n_channels * 2**i for i in range(1, self.downsamplings + 1)]

        self.down_pipe = self.build_encoder(hidden_dims)

        if (img_size / 2**self.downsamplings) < 1:
            raise ValueError("Img size is too small or #downsaplings is too big.")

        down_final_size = (
            img_size // 2**self.downsamplings
        ) ** 2 * self.linear_hidden_size

        self.fc1_mu = nn.Linear(down_final_size, self.linear_hidden_size)
        self.fc1_sigma = nn.Linear(down_final_size, self.linear_hidden_size)
        self.fc2_mu = nn.Linear(self.linear_hidden_size, self.latent_size)
        self.fc2_sigma = nn.Linear(self.linear_hidden_size, self.latent_size)

    def forward(self, x):
        enc_x = self.down_pipe(x)
        enc_x = torch.flatten(enc_x, start_dim=1)
        mu = self.fc2_mu(torch.relu(self.fc1_mu(enc_x)))
        log_var = self.fc2_sigma(torch.relu(self.fc1_sigma(enc_x)))
        std = torch.exp(log_var)
        eps = torch.randn_like(std)
        kld = 0.5 * (log_var.exp() + mu**2 - log_var - 1)
        return eps * std + mu, kld

    def build_encoder(self, hidden_dims: tp.List[int]) -> nn.Sequential:
        in_channels = 3
        modules: tp.List[nn.Module] = []

        modules.append(
            nn.Conv2d(
                in_channels,
                self.down_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        )

        in_channels = self.down_channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels=h_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                ),
            )
            in_channels = h_dim

        modules.append(
            nn.Conv2d(
                hidden_dims[-1],
                self.linear_hidden_size,
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )

        return nn.Sequential(*modules)
