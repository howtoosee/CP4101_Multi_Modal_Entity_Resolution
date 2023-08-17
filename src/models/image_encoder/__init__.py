import logging
import os.path

import torch
import torchvision
import typing
from PIL import Image
from torch import cuda, nn

from .image_embeddings import BeitEmbeddings, ClipEmbeddings


def get_image_embedding_module(model_name: str, device: typing.Union[str, cuda.device] = None) -> nn.Module:
    model_name = model_name.lower()

    models = {
        "beit": BeitEmbeddings,
        "clip": ClipEmbeddings,
        # "layoutlmv": LayoutLMEmbeddings,
        # "imagegpt": None,
        # "visualbert": None,
    }

    if model_name not in models.keys():
        raise ValueError(f"Model {model_name} not supported")

    model = models[model_name](device)
    # if cuda.is_available():
    #     model = model.cuda()
    logging.info(f"Initialised {model.module_name} on device {model.device}")
    return model


def image_path_to_tensor(path: str) -> torch.Tensor:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File {path} not found")

    with Image.open(path) as image:
        image_tensor = torchvision.transforms.ToTensor()(image).unsqueeze(0)

    return image_tensor
