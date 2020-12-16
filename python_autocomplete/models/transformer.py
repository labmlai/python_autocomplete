import numpy as np
import torch
from torch import nn

from labml import monit
from labml_helpers.module import Module
from labml_nn.transformers import Encoder


class TransformerModel(Module):
    def __init__(self, n_tokens, d_model, encoder: Encoder, src_embed: Module):
        super().__init__()
        self.src_mask = None
        self.encoder = encoder
        self.src_embed = src_embed
        self.d_model = d_model
        self.fc = nn.Linear(d_model, n_tokens)

    @staticmethod
    def subsequent_mask(seq_len):
        attn_shape = (seq_len, seq_len)
        mask = np.triu(np.ones(attn_shape, dtype=np.uint8), k=1)
        return (torch.from_numpy(mask) == 0).unsqueeze(-1)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self.subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.src_embed(src)
        # with monit.section("transformer"):
        output = self.encoder(src, self.src_mask)
        output = self.fc(output)
        return output,
