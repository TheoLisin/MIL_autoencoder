import torch
from torch import nn


class Decoder(nn.Module):
    def __init__(
        self,
        img_size=32,
        upsamplings=3,
        latent_size=128,
        linear_hidden_size=256,
        up_channels=8,
    ):
        super().__init__()

        self.latent_size = latent_size
        self.upsamplings = upsamplings
        self.linear_hidden_size = linear_hidden_size
        self.up_channels = up_channels

        self.img_down_size = img_size // 2**self.upsamplings

        if self.img_down_size * 2**self.upsamplings != img_size:
            raise ValueError("Final img size != img_size after decoding.")

        up_start_size = self.img_down_size**2 * self.linear_hidden_size

        if latent_size < up_start_size // 2:
            self.decoder_fc = nn.Sequential(
                nn.Linear(latent_size, up_start_size // 2),
                nn.ReLU(),
                nn.Linear(up_start_size // 2, up_start_size),
            )
        else:
            self.decoder_fc = nn.Linear(latent_size, up_start_size)

        self.up_pipe = self.build_decoder()

    def forward(self, embed):
        x_dec = self.decoder_fc(embed)
        x_dec = x_dec.view(
            -1,
            self.linear_hidden_size,
            self.img_down_size,
            self.img_down_size,
        )
        x_dec = self.up_pipe(x_dec)
        return torch.tanh(x_dec)

    def build_decoder(self):
        hidden_dims = [self.up_channels * 2**i for i in range(self.upsamplings)]
        hidden_dims.reverse()
        prev_dim = self.linear_hidden_size
        modules = []

        for next_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        prev_dim, next_dim, kernel_size=4, stride=2, padding=1
                    ),
                    nn.BatchNorm2d(next_dim),
                    nn.LeakyReLU(),
                )
            )
            prev_dim = next_dim

        modules.append(nn.Conv2d(self.up_channels, 3, 1, 1, 0))

        return nn.Sequential(*modules)
