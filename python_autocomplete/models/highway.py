from torch import nn

from labml_helpers.module import Module
from labml_nn.recurrent_highway_networks import RHN


class RhnModel(Module):
    def __init__(self, *,
                 n_tokens: int,
                 embedding_size: int,
                 hidden_size: int,
                 n_layers: int,
                 depth: int):
        super().__init__()

        self.embedding = nn.Embedding(n_tokens, embedding_size)
        self.rhn = RHN(input_size=embedding_size,
                       hidden_size=hidden_size,
                       n_layers=n_layers,
                       depth=depth)
        self.fc = nn.Linear(hidden_size, n_tokens)

    def __call__(self, x, s0=None):
        # shape of x is [seq, batch, feat]
        x = self.embedding(x)
        out, s = self.rhn(x, s0)
        logits = self.fc(out)

        return logits, s
