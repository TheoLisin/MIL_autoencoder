from torch import nn

from mil_autoencoder.model.encoder import Encoder
from mil_autoencoder.model.decoder import Decoder


class VAE(nn.Module):
    def __init__(
        self,
        img_size=32,
        downsamplings=3,
        latent_size=128,
        linear_hidden_size=256,
        down_channels=4,
        up_channels=8,
    ):
        super().__init__()

        self.latent_size = latent_size
        self.downsamplings = downsamplings
        self.linear_hidden_size = linear_hidden_size
        self.down_channels = down_channels
        self.up_channels = up_channels

        self.encoder = Encoder(
            img_size=img_size,
            downsamplings=downsamplings,
            latent_size=latent_size,
            linear_hidden_size=linear_hidden_size,
            down_channels=down_channels,
        )

        self.decoder = Decoder(
            img_size=img_size,
            upsamplings=downsamplings,
            latent_size=latent_size,
            linear_hidden_size=linear_hidden_size,
            up_channels=up_channels,
        )

    def forward(self, x):
        z, kld = self.encoder(x)
        x_pred = self.decoder(z)
        return x_pred, kld
