from torch import nn, Tensor


class ClassificationHead(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        latent_size: int,
        start_dim: int = 32,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.encoder.requires_grad_(False)
        self.num_of_classes = 10
        self.head = self.build_head(latent_size, start_dim)

    def build_head(self, latent_size: int, start_size: int = 32) -> nn.Module:
        """Create classification part."""
        modules = []
        dims = [start_size]
        start_pow = len(f"{start_size - 1:b}") - 1
        dims.extend([2**i for i in range(start_pow, 3, -1)])
        prev_dim = latent_size

        for dim in dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(prev_dim, dim),
                    nn.BatchNorm1d(dim),
                    nn.ReLU(),
                ),
            )
            prev_dim = dim

        modules.append(nn.Linear(prev_dim, self.num_of_classes))
        return nn.Sequential(*modules)

    def forward(self, img: Tensor) -> Tensor:
        latent_repr, _ = self.encoder(img)
        return self.head(latent_repr)

    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder weights."""
        self.encoder.requires_grad_(True)
