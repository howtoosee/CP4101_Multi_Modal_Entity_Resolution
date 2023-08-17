import logging

import torch
from torch import nn


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


class ModalDiscriminator(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super(ModalDiscriminator, self).__init__()
        self.logreg = LogisticRegression(input_size, output_size)
        self.softmax = nn.Softmax(dim=1)
        self.device = next(self.logreg.parameters()).device


    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        output = self.logreg(input_tensor)
        return self.softmax(output)


def get_modal_discriminator(input_size=768, output_size=2) -> ModalDiscriminator:
    discriminator = ModalDiscriminator(input_size=input_size, output_size=output_size)

    # if cuda.is_available():
    #     discriminator = discriminator.cuda()

    logging.info(f"Initialised modal discriminator on device {discriminator.device}")
    return discriminator
