import typing

import torch
from torch import cuda, nn
from transformers import AutoModel, AutoTokenizer, BatchEncoding, CLIPTextModel


BASE_MODULE_NAME_FORMAT = "TEXT_ENCODER-{}"


class TextEmbeddingBaseModule(nn.Module):
    """
    Base class for text embeddings
    """


    def __init__(self, model_name: str, model: nn.Module, tokenizer: nn.Module, device: typing.Union[str, cuda.device]):
        super(TextEmbeddingBaseModule, self).__init__()
        self.module_name = BASE_MODULE_NAME_FORMAT.format(model_name)
        self.model = model
        self.tokenizer = tokenizer

        if cuda.is_available():
            self.model = self.model.cuda(device)

        self.device = self.model.device


    def _encode_texts(self, texts: typing.Union[str, typing.List[str]]) -> BatchEncoding:
        return self.tokenizer(texts, add_special_tokens=True, return_tensors="pt")


class ClipEmbeddings(TextEmbeddingBaseModule):
    """
    Torch NN Module for CLIP text encoder
    https://huggingface.co/docs/transformers/model_doc/clip
    """
    MODEL_NAME = "openai/clip-vit-large-patch14"  # output shape 768
    # MODEL_NAME = "openai/clip-vit-base-patch32"     # output shape 512


    def __init__(self, device: str = None):
        self.model_name = "clip"
        model = CLIPTextModel.from_pretrained(self.MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        super(ClipEmbeddings, self).__init__(self.MODEL_NAME, model, tokenizer, device)


    def forward(self, texts: typing.Union[str, typing.List[str]]) -> torch.Tensor:
        texts = [self.truncate_sentence(text) for text in texts] if isinstance(texts, list) \
            else self.truncate_sentence(texts)

        encoded_text = self._encode_texts(texts)
        encoded_text = encoded_text.to(self.device)
        return self.model(**encoded_text)["pooler_output"]


    def truncate_sentence(self, sentence: str) -> str:
        """
        Truncate a sentence to fit the CLIP max token limit (77 tokens including the
        starting and ending tokens).

        Args:
            sentence(string): The sentence to truncate.
            tokenizer(CLIPTokenizer): Rretrained CLIP tokenizer.
        """
        cur_sentence = sentence
        tokens = self.tokenizer.encode(cur_sentence)

        if len(tokens) > 77:
            # Skip the starting token, only include 75 tokens
            truncated_tokens = tokens[1:76]
            cur_sentence = self.tokenizer.decode(truncated_tokens)

            # Recursive call here, because the encode(decode()) can have different result
            return self.truncate_sentence(cur_sentence)
        else:
            return cur_sentence


class RoBertaEmbeddings(TextEmbeddingBaseModule):
    """
    Torch NN Module for RoBERTa text encoder
    https://huggingface.co/docs/transformers/model_doc/roberta
    """
    MODEL_NAME = "roberta-base"


    def __init__(self, device: str = None):
        self.model_name = "roberta"
        model = AutoModel.from_pretrained(self.MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        super(RoBertaEmbeddings, self).__init__(self.MODEL_NAME, model, tokenizer, device)


    def forward(self, texts):
        encoded_text = self._encode_texts(texts)
        encoded_text = encoded_text.to(self.device)
        return self.model(**encoded_text)["pooler_output"]


'''
class DistilBertEmbeddings(TextEmbeddingBaseModule):
    """
    Torch NN Module for DistilBERT text encoder.
    https://huggingface.co/docs/transformers/model_doc/distilbert
    """
    MODEL_NAME = "distilbert-base-uncased"


    def __init__(self):
        super(DistilBertEmbeddings, self).__init__(self.MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModel.from_pretrained(self.MODEL_NAME)


    def forward(self, texts: typing.Union[str, typing.List[str]]) -> torch.Tensor:
        encoded_text = self.__encode_texts(texts)
        encoded_text = encoded_text.to(self.device)
        return self.model(**encoded_text)[0][:, 0, :]


class LayoutLmV3Embeddings(TextEmbeddingBaseModule):
    """
    Torch NN Module for LayoutLMv3 text encoder
    https://huggingface.co/docs/transformers/model_doc/layoutlmv3
    """
    MODEL_NAME = "microsoft/layoutlmv3-base"


    def __init__(self):
        super(LayoutLmV3Embeddings, self).__init__(self.MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModel.from_pretrained(self.MODEL_NAME)
'''
