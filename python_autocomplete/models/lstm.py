from typing import Optional, Tuple

import torch
from torch import nn

from labml_nn.lstm import LSTM
from python_autocomplete.models import AutoregressiveModel


class LstmModel(AutoregressiveModel):
    def __init__(self, *,
                 n_tokens: int,
                 embedding_size: int,
                 hidden_size: int,
                 n_layers: int):
        super().__init__()

        self.embedding = nn.Embedding(n_tokens, embedding_size)
        self.lstm = LSTM(input_size=embedding_size,
                         hidden_size=hidden_size,
                         n_layers=n_layers)
        self.fc = nn.Linear(hidden_size, n_tokens)

    def __call__(self, x: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]]):
        # shape of x is [seq, batch, feat]
        x = self.embedding(x)
        out, (hn, cn) = self.lstm(x, state)
        logits = self.fc(out)

        return logits, (hn, cn)
