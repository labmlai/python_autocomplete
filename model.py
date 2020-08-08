from labml.helpers.pytorch.module import Module
from torch import nn


class SimpleLstmModel(Module):
    def __init__(self, *,
                 encoding_size,
                 embedding_size,
                 lstm_size,
                 lstm_layers):
        super().__init__()

        self.embedding = nn.Embedding(encoding_size, embedding_size)
        self.lstm = nn.LSTM(input_size=embedding_size,
                            hidden_size=lstm_size,
                            num_layers=lstm_layers)
        self.fc = nn.Linear(lstm_size, encoding_size)

    def __call__(self, x, h0=None, c0=None):
        # shape of x is [seq, batch, feat]
        x = self.embedding(x)
        state = (h0, c0) if h0 is not None else None
        out, (hn, cn) = self.lstm(x, state)
        logits = self.fc(out)

        return logits, (hn, cn)
