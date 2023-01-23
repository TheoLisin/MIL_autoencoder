import yaml
from marshmallow_dataclass import class_schema
from dataclasses import dataclass
from typing import Union
from pathlib import Path


@dataclass
class ModelParams:
    img_size: int
    downsamplings: int
    latent_size: int
    linear_hidden_size: int
    down_channels: int
    up_channels: int


@dataclass
class TrainigParams:
    modelname: str
    modelparams: ModelParams
    optim: str
    lr: float
    device: str
    loss: str
    kld_lambda: float
    epochs: int
    test_batch_size: int
    train_batch_size: int
    augmentation: bool


TrainingPipelineSchema = class_schema(TrainigParams)


def read_params(path: Union[str, Path]) -> TrainigParams:
    with open(path, "r") as param_file:
        schema = TrainingPipelineSchema()
        tparams = schema.load(yaml.safe_load(param_file))

    return tparams
