from typing import List, Optional

import torch
from torch import nn

from labml_nn.transformers.xl import TransformerXL
from python_autocomplete.models import AutoregressiveModel


class TransformerXLModel(AutoregressiveModel):
    def __init__(self, n_vocab: int, d_model: int, transformer: TransformerXL):
        super().__init__()
        self.src_embed = nn.Embedding(n_vocab, d_model)
        self.transformer = transformer
        self.generator = nn.Linear(d_model, n_vocab)
        self.mask_x = None
        self.mask_mem = None

    def __call__(self, x: torch.Tensor, mem: Optional[List[torch.Tensor]]):
        m_len = len(mem[0]) if mem else 0
        if self.mask_x is None or self.mask_x.shape[0] < len(x):
            from labml_nn.transformers.utils import subsequent_mask
            self.mask_x = subsequent_mask(len(x)).to(x.device)
        if self.mask_mem is None or self.mask_mem.shape[1] < m_len or self.mask_mem.shape[0] < len(x):
            self.mask_mem = self.mask_x.new_ones(len(x), m_len, 1)

        if m_len:
            mask = torch.cat((self.mask_mem[:len(x), :m_len], self.mask_x[:len(x), :len(x)]), dim=1)
        else:
            mask = self.mask_x[:len(x), :len(x)]

        x = self.src_embed(x)
        res, mem = self.transformer(x, mem, mask)
        res = self.generator(res)

        return res, mem
