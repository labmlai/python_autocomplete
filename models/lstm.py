from torch import nn

from labml_helpers.module import Module


class LstmModel(Module):
    def __init__(self, *,
                 n_tokens: int,
                 embedding_size: int,
                 lstm_size: int,
                 lstm_layers: int):
        super().__init__()

        self.embedding = nn.Embedding(n_tokens, embedding_size)
        self.lstm = nn.LSTM(input_size=embedding_size,
                            hidden_size=lstm_size,
                            num_layers=lstm_layers)
        self.fc = nn.Linear(lstm_size, n_tokens)

    def __call__(self, x, h0=None, c0=None):
        # shape of x is [seq, batch, feat]
        x = self.embedding(x)
        state = (h0, c0) if h0 is not None else None
        out, (hn, cn) = self.lstm(x, state)
        logits = self.fc(out)

        return logits, (hn, cn)