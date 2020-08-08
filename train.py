from typing import Callable

import torch
import torch.nn as nn
import torchtext
from labml import lab, experiment, tracker
from labml.configs import option, BaseConfigs
from labml.helpers.pytorch.device import DeviceConfigs
from labml.helpers.pytorch.module import Module
from labml.helpers.pytorch.train_valid import TrainValidConfigs
from labml.utils.pytorch import get_modules


class OptimizerConfigs(BaseConfigs):
    optimizer: torch.optim.Adam = 'adam'
    learning_rate = 2.4e-4
    parameters: any
    d_model: int

    def __init__(self):
        super().__init__(_primary='optimizer')


@option(OptimizerConfigs.optimizer, 'adam')
def adam_optimizer(c: OptimizerConfigs):
    return torch.optim.Adam(c.parameters, lr=c.learning_rate)


class SequentialDataLoader:
    def __init__(self, *, text: str, field: torchtext.data.Field,
                 batch_size: int, seq_len: int,
                 device: torch.device):
        self.seq_len = seq_len
        data = field.numericalize([text])
        n_batch = data.size(0) // batch_size
        data = data.narrow(0, 0, n_batch * batch_size)
        data = data.view(batch_size, -1).t().contiguous()
        self.data = data.to(device)
        self.dataset = self.data

    def __len__(self):
        return self.data.shape[0] // self.seq_len

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx >= self.data.shape[0] - 1:
            raise StopIteration()

        seq_len = min(self.seq_len, self.data.shape[0] - 1 - self.idx)
        i = self.idx + seq_len
        data = self.data[self.idx: i]
        target = self.data[self.idx + 1: i + 1]
        self.idx = i
        return data, target


class Configs(DeviceConfigs, TrainValidConfigs):
    model: Module
    field: torchtext.data.Field
    text: 'Dataset'
    batch_size: int = 20
    seq_len: int = 32
    n_tokens: int
    d_model: int = 200
    n_layers: int = 2
    dropout: float = 0.2
    n_heads: int = 2
    d_ff: int = 400
    d_lstm: int = 200
    tokenizer: Callable

    is_save_models = True
    is_relative_attention = False

    def run(self):
        for _ in self.training_loop:
            prompt = 'def train('
            for i in range(50):
                data = self.tokenizer(prompt)
                data = self.field.numericalize([data])
                data = data.to(self.device)
                output, _ = self.model(data)
                output = output.argmax(dim=-1).squeeze()
                prompt += '' + self.field.vocab.itos[output[-1]]

            print(prompt)

            with tracker.namespace('train'):
                self.trainer()
            with tracker.namespace('valid'):
                self.validator()


class SimpleAccuracyFunc(Module):
    def __call__(self, output: torch.Tensor, target: torch.Tensor) -> int:
        pred = output.argmax(dim=-1)
        return pred.eq(target).sum().item() / target.shape[1]


@option(Configs.accuracy_func)
def simple_accuracy():
    return SimpleAccuracyFunc()


@option(Configs.optimizer)
def _optimizer(c: Configs):
    optimizer = OptimizerConfigs()
    optimizer.parameters = c.model.parameters()
    optimizer.d_model = c.d_model
    optimizer.optimizer = 'adam'

    return optimizer


class CrossEntropyLoss(Module):
    def __init__(self, n_tokens: int):
        super().__init__()
        self.n_tokens = n_tokens
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, outputs, targets):
        return self.loss(outputs.view(-1, self.n_tokens), targets.view(-1))


@option(Configs.loss_func)
def _loss_func(c: Configs):
    return CrossEntropyLoss(c.n_tokens)


@option(Configs.n_tokens)
def _n_tokens(c: Configs):
    assert c.text
    return len(c.field.vocab.stoi)


@option(Configs.model)
def lstm_model(c: Configs):
    from model import SimpleLstmModel
    m = SimpleLstmModel(encoding_size=c.n_tokens,
                        embedding_size=c.d_model,
                        lstm_size=c.d_lstm,
                        lstm_layers=c.n_layers)
    return m.to(c.device)


def character_tokenizer(x: str):
    return list(x)


@option(Configs.tokenizer)
def character():
    return character_tokenizer


@option(Configs.field)
def _field(c: Configs):
    return torchtext.data.Field(tokenize=c.tokenizer,
                                init_token='<sos>',
                                eos_token='<eos>',
                                lower=True)


class Dataset:
    def __init__(self, train: str, valid: str):
        self.train = train
        self.valid = valid


@option(Configs.text)
def wiki_text_2(c: Configs):
    with open(str(lab.get_data_path() / 'train.py'), 'r') as f:
        train = f.read()
    with open(str(lab.get_data_path() / 'valid.py'), 'r') as f:
        valid = f.read()

    c.field.build_vocab(train)
    return Dataset(train, valid)


@option(Configs.train_loader)
def train_loader(c: Configs):
    return SequentialDataLoader(text=c.text.train,
                                field=c.field,
                                batch_size=c.batch_size,
                                seq_len=c.seq_len,
                                device=c.device)


@option(Configs.valid_loader)
def train_loader(c: Configs):
    return SequentialDataLoader(text=c.text.valid,
                                field=c.field,
                                batch_size=c.batch_size,
                                seq_len=c.seq_len,
                                device=c.device)


def main():
    conf = Configs()
    # conf.d_model = 512
    # conf.d_ff = 2048
    # conf.n_heads = 16
    # conf.n_layers = 6
    # conf.seq_len = 256
    # conf.epochs = 1024
    # conf.dropout = 0.4
    experiment.create(name="source_code",
                      comment='lstm model')
    experiment.configs(conf, 'run')
    experiment.add_pytorch_models(get_modules(conf))
    # experiment.load('d5ba7f56d88911eaa6629b54a83956dc')
    experiment.start()
    conf.run()


if __name__ == '__main__':
    main()
