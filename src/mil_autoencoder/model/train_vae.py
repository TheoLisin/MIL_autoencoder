import typing as tp
from torch import nn, no_grad
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm import tqdm
from logging import getLogger

logger = getLogger(__name__)


def train_vae(
    model: nn.Module,
    optim: Optimizer,
    device: tp.Any,
    reconstruction_loss_func: tp.Callable,
    trainloader: DataLoader,
    testloader: DataLoader,
    kld_lambda: float = 0.1,
    epochs=20,
    **loss_kwargs,
) -> tp.Tuple[tp.List[float], tp.List[float]]:
    train_batch_size = trainloader.batch_size
    test_batch_size = testloader.batch_size
    model.to(device)
    train_loss_history = []
    test_loss_history = []

    for ep in range(1, epochs + 1):
        total_batches = 0
        rec_loss_avg: float = 0
        kld_loss_avg: float = 0
        fin_loss: float = 0

        model.train()

        for batch, _ in tqdm(trainloader):
            total_batches += 1
            img = batch.to(device)
            rec_img, kld = model(img)
            kld_loss = kld.sum() / train_batch_size
            rec_loss = (
                reconstruction_loss_func(rec_img, img, **loss_kwargs) / train_batch_size
            )
            loss = rec_loss + kld_lambda * kld_loss
            optim.zero_grad()
            loss.backward()
            optim.step()
            kld_loss_avg += kld_loss.item()
            rec_loss_avg += rec_loss.item()
            fin_loss += loss.item()

        train_loss_history.append(fin_loss / total_batches)
        logger.info(
            (
                f"Epoch {ep} |"
                f"Reconstruction loss: {rec_loss_avg / total_batches} |"
                f"KLD loss: {kld_loss_avg / total_batches}"
            )
        )

        model.eval()
        with no_grad():
            total_batches = 0
            rec_loss_avg = 0
            kld_loss_avg = 0
            fin_loss = 0
            for batch, _ in tqdm(testloader):
                total_batches += 1
                img = batch.to(device)
                rec_img, kld = model(img)
                kld_loss = kld.sum() / test_batch_size
                rec_loss = (
                    reconstruction_loss_func(rec_img, img, **loss_kwargs) / test_batch_size
                )
                kld_loss_avg += kld_loss.item()
                rec_loss_avg += rec_loss.item()
                loss = rec_loss + kld_lambda * kld_loss
                fin_loss += loss.item()

        test_loss_history.append(fin_loss / total_batches)
        logger.info(
            (
                f"Epoch {ep} |"
                f"Reconstruction test loss: {rec_loss_avg / total_batches} |"
                f"KLD test loss: {kld_loss_avg / total_batches}"
            )
        )

    return train_loss_history, test_loss_history
