import torch


def mse_loss(rec_img: torch.Tensor, img: torch.Tensor):
    return ((rec_img - img) ** 2).sum()


def logcosh_loss(
    rec_img: torch.Tensor,
    img: torch.Tensor,
    alpha: float = 25.0,
    eps: float = 1e-12,
):
    cosh = torch.cosh(alpha * (img - rec_img))
    return (1.0 / alpha * torch.log(cosh + eps)).sum()
