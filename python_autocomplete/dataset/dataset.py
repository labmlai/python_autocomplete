from pathlib import PurePath
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader

from labml import lab, monit
from labml.configs import option, BaseConfigs
from labml_helpers.datasets.text import TextDataset
from python_autocomplete.dataset import Tokenizer
from python_autocomplete.dataset.bpe import BPE, SourceCodeTokenizer


class CharacterTokenizer(Tokenizer):
    def __init__(self, retrain: bool):
        from labml.utils.cache import cache_get

        self.n_tokens = cache_get('n_tokens')
        self.itos = cache_get('itos')
        self.stoi = cache_get('stoi')
        if retrain is None:
            self.is_trained = self.n_tokens and self.itos and self.stoi
        else:
            self.is_trained = not retrain

    def encode(self, data: str, *, is_silent: bool = True):
        return [self.stoi[c] for c in data if c in self.stoi]

    def train(self, data: str):
        with monit.section("Build vocabulary"):
            self.itos = list(sorted(list(set(data))))
            self.n_tokens = len(self.itos)
            self.stoi = {c: i for i, c in enumerate(self.itos)}

        from labml.utils.cache import cache_set

        cache_set(f'n_tokens', self.n_tokens)
        cache_set(f'itos', self.itos)
        cache_set(f'stoi', self.stoi)


class SourceCodeDataset:
    @staticmethod
    def load(path: PurePath):
        with open(str(path), 'r') as f:
            return f.read()

    @staticmethod
    def get_train_valid(path: PurePath, is_load_data: bool):
        if is_load_data:
            with monit.section("Load data"):
                train = TextDataset.load(path / 'train.py')
                valid = TextDataset.load(path / 'valid.py')
        else:
            train = ''
            valid = ''

        return train, valid

    def __init__(self, tokenizer: Tokenizer, train, valid):
        self.train = train
        self.valid = valid
        self.tokenizer = tokenizer

    def __repr__(self):
        return f'{len(self.train) / 1_000_000 :,.2f}M, {len(self.valid) / 1_000_000 :,.2f}'


class SourceCodeDataConfigs(BaseConfigs):
    dataset: SourceCodeDataset
    truncate_data: int = 0
    is_load_data: bool = True
    tokenizer: Tokenizer
    retrain_tokenizer: bool = True

    train_loader: DataLoader
    valid_loader: DataLoader
    is_shuffle: bool = True
    batch_size: int
    seq_len: int

    def text_to_i(self, text: str, *, is_silent: bool = True) -> torch.Tensor:
        return torch.tensor(self.tokenizer.encode(text, is_silent=is_silent), dtype=torch.long)


@option(SourceCodeDataConfigs.dataset, 'default')
def _dataset(c: SourceCodeDataConfigs):
    train, valid = SourceCodeDataset.get_train_valid(lab.get_data_path(), c.is_load_data)
    if c.truncate_data:
        train, valid = train[:c.truncate_data], valid[:c.truncate_data]
    if not c.tokenizer.is_trained:
        c.tokenizer.train(train + valid)
    return SourceCodeDataset(c.tokenizer, train, valid)


@option(SourceCodeDataConfigs.tokenizer, 'bpe')
def _bpe_tokenizer():
    from labml.utils.cache import cache_get
    from python_autocomplete.dataset.bpe import BPEEnDe
    bpe_cache = cache_get('bpe')

    if bpe_cache:
        bpe_en_de = BPEEnDe()
        bpe_en_de.load(**bpe_cache)
    else:
        raise RuntimeError('BPE not cached')

    return BPE(bpe_en_de, SourceCodeTokenizer())


@option(SourceCodeDataConfigs.tokenizer, 'char')
def _char_tokenizer(c: SourceCodeDataConfigs):
    return CharacterTokenizer(c.retrain_tokenizer)


# Data loaders
class TokenDataset(Dataset):
    def __init__(self, *,
                 data: torch.Tensor,
                 batch_size: int,
                 seq_len: int,
                 drop_last: bool = False):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.data = data
        self.n_samples = (self.data.shape[0] - 1) // self.seq_len
        if not drop_last:
            self.n_batches = (self.n_samples + batch_size - 1) // batch_size
        else:
            self.n_batches = self.n_samples // batch_size

    def __len__(self):
        return (self.data.shape[0] - 1) // self.seq_len

    def __getitem__(self, idx):
        batch = idx // self.batch_size
        batch_idx = idx % self.batch_size
        idx = batch_idx * self.n_batches + batch
        start = idx * self.seq_len
        assert start + self.seq_len + 1 <= self.data.shape[0]
        end = start + self.seq_len
        data = self.data[start: end]
        target = self.data[start + 1: end + 1]
        return data, target


def transpose_batch(batch):
    transposed_data = list(zip(*batch))
    src = torch.stack(transposed_data[0], 1)
    tgt = torch.stack(transposed_data[1], 1)

    return src, tgt


@option(SourceCodeDataConfigs.train_loader)
def _train_loader(c: SourceCodeDataConfigs):
    return DataLoader(TokenDataset(data=c.text_to_i(c.dataset.train, is_silent=False),
                                   batch_size=c.batch_size,
                                   seq_len=c.seq_len,
                                   drop_last=True),
                      batch_size=c.batch_size,
                      collate_fn=transpose_batch,
                      shuffle=c.is_shuffle,
                      drop_last=True)


@option(SourceCodeDataConfigs.valid_loader)
def _valid_loader(c: SourceCodeDataConfigs):
    return DataLoader(TokenDataset(data=c.text_to_i(c.dataset.valid, is_silent=False),
                                   batch_size=c.batch_size,
                                   seq_len=c.seq_len,
                                   drop_last=True),
                      batch_size=c.batch_size,
                      collate_fn=transpose_batch,
                      shuffle=c.is_shuffle,
                      drop_last=True)
