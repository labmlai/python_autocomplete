from typing import Any

import torch
from torch import nn

from labml_helpers.module import Module
from labml_nn.transformers import Encoder
from labml_nn.transformers.utils import subsequent_mask
from python_autocomplete.models import AutoregressiveModel


class TransformerModel(AutoregressiveModel):
    def __init__(self, n_tokens, d_model, encoder: Encoder, src_embed: Module):
        super().__init__()
        self.src_mask = None
        self.encoder = encoder
        self.src_embed = src_embed
        self.d_model = d_model
        self.fc = nn.Linear(d_model, n_tokens)

    def __call__(self, src: torch.Tensor, _: Any = None):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            self.src_mask = subsequent_mask(len(src)).to(src.device)

        src = self.src_embed(src)
        # with monit.section("transformer"):
        output = self.encoder(src, self.src_mask)
        output = self.fc(output)
        return output, None
