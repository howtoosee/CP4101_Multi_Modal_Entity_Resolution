import logging

import typing
from torch import cuda, nn

from .text_embeddings import ClipEmbeddings, RoBertaEmbeddings


def get_text_embedding_module(model_name: str, device: typing.Union[str, cuda.device] = None) -> nn.Module:
    model_name = model_name.lower()

    models = {
        "clip": ClipEmbeddings,
        "roberta": RoBertaEmbeddings,
        # "distilbert": DistilBertEmbeddings,
        # "layoutlmv3": LayoutLmV3Embeddings,
    }

    if model_name not in models.keys():
        raise ValueError(f"Model {model_name} is not supported")

    model = models[model_name](device)
    # if cuda.is_available():
    #     model = model.cuda()
    logging.info(f"Initialised {model.module_name} on device {model.device}")
    return model
