import typing as tp
import numpy as np

from logging import getLogger
from sklearn.metrics import classification_report
from torch import nn, no_grad, Tensor, device as Device
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


logger = getLogger(__name__)


def train_cls(
    model: nn.Module,
    optim: Optimizer,
    loss_func: tp.Callable[[Tensor, Tensor], Tensor],
    epochs: int,
    trainloader: DataLoader,
    testloader: DataLoader,
    labels: tp.Set[str],
    device: Device,
):

    train_loss_history = []
    test_loss_hitory = []

    model.to(device)

    for ep in range(1, epochs + 1):
        avg_loss: float = 0
        total_batches = 0

        model.train()
        for imgs, target in tqdm(trainloader, desc=f"Train #{ep}"):
            total_batches += 1
            imgs.to(device)
            preds = model(imgs)
            loss = loss_func(preds, target)
            optim.zero_grad()
            loss.backward()
            optim.step()
            avg_loss += loss.item()

        avg_loss /= total_batches
        train_loss_history.append(avg_loss)
        logger.info(f"Epoch #{ep} | Train Avg Loss: {avg_loss:.4f} ")

        model.eval()
        with no_grad():
            avg_loss = 0
            total_batches = 0
            preds_labels = np.array([])
            target_labels = np.array([])
            for imgs, target in tqdm(testloader, desc=f"Test #{ep}"):
                total_batches += 1
                imgs.to(device)
                preds = model(imgs)
                avg_loss += loss_func(preds, target).item()
                preds = preds.detach().cpu().numpy()
                batch_pred_labels = preds.argmax(axis=1)
                preds_labels = np.concatenate((preds_labels, batch_pred_labels))
                target_labels = np.concatenate((target_labels, target.detach().numpy()))

            avg_loss /= total_batches
            test_loss_hitory.append(avg_loss)
            logger.info(f"Epoch #{ep} | Test Avg Loss: {avg_loss:.4f} ")
            logger.info(
                classification_report(
                    target_labels.astype(int),
                    preds_labels.astype(int),
                    target_names=labels,
                ),
            )

    return train_loss_history, test_loss_hitory
