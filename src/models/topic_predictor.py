import logging
import typing

import torch
from torch import nn


# def get_logreg(input_shape: int, output_shape: int = 1) -> nn.Module:
#     return nn.Sequential(
#         nn.Linear(input_shape, 128),
#         nn.ReLU(),
#         nn.Linear(128, 64),
#         nn.LayerNorm(64),
#         nn.ReLU(),
#         nn.Linear(64, output_shape),
#         nn.Sigmoid()
#     )


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.Linear(64, output_dim)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # outputs = torch.sigmoid(self.layers(x))
        # return outputs
        return self.layers(x)


class MultiTopicPredictor(nn.Module):
    def __init__(self, num_topics, input_size):
        super(MultiTopicPredictor, self).__init__()
        self.classifiers = nn.ModuleList([LogisticRegression(input_size, 1) for _ in range(num_topics)])
        self.device = next(self.classifiers.parameters()).device


    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        logits = torch.cat([classifier(input_tensor) for classifier in self.classifiers], dim=1)
        return logits


    def predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        logits = self.forward(input_tensor)
        return torch.sigmoid(logits)


class SingleTopicPredictor(nn.Module):
    def __init__(self, num_topics, input_size):
        super(SingleTopicPredictor, self).__init__()
        self.num_topics = num_topics
        self.logreg = LogisticRegression(input_size, num_topics)
        self.softmax = nn.Softmax(dim=1)
        self.device = next(self.layers.parameters()).device


    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        output = self.logreg(input_tensor)
        return self.softmax(output)


    def predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        logits = self.forward(input_tensor)
        indices = logits.argmax(dim=1)
        return nn.functional.one_hot(indices, num_classes=self.num_topics)


def get_topic_predictor(is_multilabel: bool, num_topics: int, input_size=768) -> typing.Union[MultiTopicPredictor, SingleTopicPredictor]:
    if is_multilabel:
        predictor = MultiTopicPredictor(num_topics, input_size)
    else:
        predictor = SingleTopicPredictor(num_topics, input_size)

    # if cuda.is_available():
    #     predictor = predictor.cuda()

    logging.info(
        f"Initialised {'multilabel' if is_multilabel else 'single label'} topic predictor on device {predictor.device}")
    return predictor
