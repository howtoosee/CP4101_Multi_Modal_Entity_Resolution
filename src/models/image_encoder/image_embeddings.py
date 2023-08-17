from typing import List, Union

import torch
import typing
from PIL import Image
from torch import cuda, nn


BASE_MODULE_NAME_FORMAT = "IMAGE_ENCODER-{}"


class BeitEmbeddings(nn.Module):
    """
    Torch NN Module for Bidirectional Encoder representation from Image Transformers image encoder
    https://huggingface.co/docs/transformers/model_doc/beit
    """
    MODEL_NAME = "microsoft/beit-base-patch16-224-pt22k-ft22k"


    def __init__(self, device: typing.Union[str, cuda.device] = None):
        super(BeitEmbeddings, self).__init__()
        from transformers import BeitModel, BeitImageProcessor

        self.model_name = "beit"
        self.module_name = BASE_MODULE_NAME_FORMAT.format(self.MODEL_NAME)
        self.processor = BeitImageProcessor.from_pretrained(self.MODEL_NAME)
        self.model = BeitModel.from_pretrained(self.MODEL_NAME)

        if cuda.is_available():
            self.model = self.model.cuda(device)

        self.device = self.model.device


    def forward(self, images: Union[Image.Image, List[Image.Image], torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        inputs = self.processor.preprocess(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        model_outputs = self.model(pixel_values=pixel_values)
        return model_outputs["pooler_output"]


class ClipEmbeddings(nn.Module):
    """
    Torch NN Module for CLIP image encoder
    https://huggingface.co/docs/transformers/model_doc/clip
    """
    MODEL_NAME = "openai/clip-vit-large-patch14"    # output shape 768
    # MODEL_NAME = "openai/clip-vit-base-patch32"   # output shape 512


    def __init__(self, device: str = None):
        super(ClipEmbeddings, self).__init__()
        from transformers import CLIPModel, CLIPImageProcessor

        self.model_name = "clip"
        self.module_name = BASE_MODULE_NAME_FORMAT.format(self.MODEL_NAME)
        self.processor = CLIPImageProcessor.from_pretrained(self.MODEL_NAME)
        self.model = CLIPModel.from_pretrained(self.MODEL_NAME)

        if cuda.is_available():
            self.model = self.model.cuda(device)

        self.device = self.model.device


    def forward(self, images: Union[Image.Image, List[Image.Image], torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        inputs = self.processor.preprocess(images=images, return_tensors="pt", apply_ocr=False)
        pixel_values = inputs["pixel_values"].to(self.device)
        model_output = self.model.get_image_features(pixel_values=pixel_values)
        return model_output


'''
class LayoutLMEmbeddings(nn.Module):
    """
    Torch NN Module for LayoutLM V3 image encoder
    https://huggingface.co/docs/transformers/model_doc/layoutlmv3
    """
    MODEL_NAME = "microsoft/layoutlmv3-base"


    def __init__(self):
        super().__init__()
        from transformers import LayoutLMv3Model, LayoutLMv3ImageProcessor

        self.module_name = BASE_MODULE_NAME_FORMAT.format(self.MODEL_NAME)
        self.processor = LayoutLMv3ImageProcessor.from_pretrained(self.MODEL_NAME)
        self.model = LayoutLMv3Model.from_pretrained(self.MODEL_NAME)


    def __call__(self, images: Union[Image.Image, List[Image.Image], torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        return self.forward(images)


    def forward(self, images: Union[Image.Image, List[Image.Image], torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        inputs = self.processor.preprocess(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"]
        model_output = self.model.forward_image(pixel_values=pixel_values)
        return model_output
'''
