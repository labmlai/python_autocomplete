from pathlib import PurePath
from typing import Callable, List

import torch
import torch.nn as nn
from labml import lab, experiment, tracker, monit, logger
from labml.configs import option
from labml.helpers.pytorch.device import DeviceConfigs
from labml.helpers.pytorch.module import Module
from labml.helpers.pytorch.optimizer import OptimizerConfigs
from labml.helpers.pytorch.train_valid import TrainValidConfigs, Mode
from labml.logger import Text
from labml.utils.pytorch import get_modules
from torch.utils.data import IterableDataset


class TextDataset:
    train: str
    valid: str
    standard_tokens: List[str] = []

    @staticmethod
    def load(path: PurePath):
        with open(str(path), 'r') as f:
            return f.read()

    def __init__(self, path: PurePath, tokenizer: Callable):
        self.tokenizer = tokenizer
        self.path = path

        with monit.section("Load data"):
            self.train = self.load(path / 'train.py')
            self.valid = self.load(path / 'valid.py')

        self.create_tokens()

    def create_tokens(self):
        self.n_tokens = len(self.standard_tokens)
        self.stoi = {t: i for i, t in enumerate(self.standard_tokens)}

        with monit.section("Tokenize"):
            tokens = self.tokenizer(self.train + self.valid)
            tokens = sorted(list(set(tokens)))

        for t in monit.iterate("Build vocabulary", tokens):
            self.stoi[t] = self.n_tokens
            self.n_tokens += 1

        self.itos = [''] * self.n_tokens
        for t, n in self.stoi.items():
            self.itos[n] = t

    def text_to_i(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer(text)
        return torch.tensor([self.stoi[s] for s in tokens], dtype=torch.long)

    def __repr__(self):
        return f'{len(self.train)}, {len(self.valid)} - {str(self.path)}'


class SequentialDataLoader(IterableDataset):
    def __init__(self, *, text: str, dataset: TextDataset,
                 batch_size: int, seq_len: int):
        self.seq_len = seq_len
        data = dataset.text_to_i(text)
        n_batch = data.shape[0] // batch_size
        data = data.narrow(0, 0, n_batch * batch_size)
        data = data.view(batch_size, -1).t().contiguous()
        self.data = data
        self.dataset = data.flatten()

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
    text: TextDataset
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
            log = [(prompt, Text.subtle)]
            for i in monit.iterate('Sample', 25):
                data = self.text.text_to_i(prompt).unsqueeze(-1)
                data = data.to(self.device)
                output, _ = self.model(data)
                output = output.argmax(dim=-1).squeeze()
                prompt += '' + self.text.itos[output[-1]]
                log += [('' + self.text.itos[output[-1]], Text.value)]

            logger.log(log)

            with Mode(is_train=True,
                      is_log_parameters=self.is_log_parameters,
                      is_log_activations=self.is_log_activations):
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
    optimizer.optimizer = 'Adam'

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
    return c.text.n_tokens


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


@option(Configs.text)
def source_code(c: Configs):
    return TextDataset(lab.get_data_path(), c.tokenizer)


@option(Configs.train_loader)
def train_loader(c: Configs):
    return SequentialDataLoader(text=c.text.train,
                                dataset=c.text,
                                batch_size=c.batch_size,
                                seq_len=c.seq_len)


@option(Configs.valid_loader)
def train_loader(c: Configs):
    return SequentialDataLoader(text=c.text.valid,
                                dataset=c.text,
                                batch_size=c.batch_size,
                                seq_len=c.seq_len)


def main():
    conf = Configs()
    conf.d_model = 512
    conf.d_ff = 2048
    conf.n_heads = 16
    conf.n_layers = 6
    conf.seq_len = 256
    conf.epochs = 1024
    conf.dropout = 0.4
    experiment.create(name="source_code",
                      comment='lstm model')
    experiment.configs(conf, 'run')
    experiment.add_pytorch_models(get_modules(conf))
    # experiment.load('d5ba7f56d88911eaa6629b54a83956dc')
    experiment.start()
    conf.run()


if __name__ == '__main__':
    main()
