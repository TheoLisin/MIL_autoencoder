import argparse
import logging
import sys
import torch

from dataclasses import asdict

from mil_autoencoder.train_params import read_params
from mil_autoencoder.data.loader import load_data, create_loaders
from mil_autoencoder.data.transformers import cifar_augment_transform, simple_transform
from mil_autoencoder.model.vae import VAE
from mil_autoencoder.model.train_vae import train_vae
from mil_autoencoder.model.train_cls import train_cls
from mil_autoencoder.model import loss_functions
from mil_autoencoder.model.classifier import ClassificationHead
from mil_autoencoder.paths import DATA, PARAMS, TRAINED_MODELS


logger = logging.getLogger()
str_handler = logging.StreamHandler(sys.stdout)
fmt = logging.Formatter(
    "[%(asctime)s\t%(levelname)s\t%(name)s]: \n%(message)s",
)
logger.setLevel(logging.INFO)
str_handler.setFormatter(fmt)
logger.addHandler(str_handler)


def parse_args():
    parser = argparse.ArgumentParser("Train")
    main_group = parser.add_mutually_exclusive_group(required=True)

    main_group.add_argument(
        "--vae",
        action="store_true",
        help="run train pipline for VAE model using config.yml file.",
    )

    main_group.add_argument(
        "--cls",
        action="store_true",
        help="run train pipline for VAE model using config.yml file.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    train_params = read_params(PARAMS)

    if train_params.augmentation:
        train_transform = cifar_augment_transform()
    else:
        train_transform = simple_transform()

    test_transform = simple_transform()

    loss = getattr(loss_functions, train_params.loss, None)

    if loss is None:
        raise NameError(
            (
                f"Couldn't find loss function {train_params.loss} "
                "in 'loss_functions' module."
            ),
        )

    trainset, testset, classes = load_data(DATA, train_transform, test_transform)
    trainloader, testloader = create_loaders(
        trainset,
        testset,
        train_batch=train_params.train_batch_size,
        test_batch=train_params.test_batch_size,
    )

    if train_params.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Cuda is not available.")
    elif train_params.device != "cpu":
        raise ValueError(f"Unknown device: {train_params.device}")

    device = torch.device(train_params.device)

    optim_func = getattr(torch.optim, train_params.optim, None)

    if optim_func is None:
        raise ValueError(
            f"Can't find optim '{train_params.optim}' in torch.optim module."
        )

    model = VAE(**asdict(train_params.modelparams))

    if args.cls:
        model.load_state_dict(
            torch.load(
                TRAINED_MODELS / train_params.modelname,
                map_location=device,
            ),
        )

        ls = train_params.modelparams.latent_size

        cls_head = ClassificationHead(
            encoder=model.encoder,
            latent_size=ls,
            start_dim=ls // 2,
        )

        loss = torch.nn.CrossEntropyLoss()
        optim = optim_func(cls_head.parameters(), lr=train_params.lr)
        train_loss, test_loss = train_cls(
            model=cls_head,
            optim=optim,
            loss_func=loss,
            device=device,
            trainloader=trainloader,
            testloader=testloader,
            epochs=train_params.epochs,
            labels=classes,
        )
    else:
        optim = optim_func(model.parameters(), lr=train_params.lr)
        train_loss, test_loss = train_vae(
            model=model,
            optim=optim,
            device=device,
            reconstruction_loss_func=loss,
            trainloader=trainloader,
            testloader=testloader,
            kld_lambda=train_params.kld_lambda,
            epochs=train_params.epochs,
        )

    # save model to mlflow
    # save graphics/reconstruction to mlflow


if __name__ == "__main__":
    main()
