from pathlib import PurePath
from typing import Callable, List, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from labml import lab, experiment, monit, logger, tracker
from labml.configs import option
from labml.logger import Text
from labml_helpers.datasets.text import TextDataset, SequentialDataLoader, SequentialUnBatchedDataset
from labml_helpers.device import DeviceConfigs
from labml_helpers.metrics.accuracy import Accuracy
from labml_helpers.metrics.simple_state import SimpleStateModule
from labml_helpers.module import Module
from labml_helpers.train_valid import TrainValidConfigs, hook_model_outputs, BatchIndex
from labml_nn.optimizers.configs import OptimizerConfigs
from labml_nn.transformers import TransformerConfigs
from python_autocomplete.bpe import BPE, SourceCodeTokenizer


class SourceCodeDataset(TextDataset):
    def __init__(self, path: PurePath, tokenizer: Callable):
        with monit.section("Load data"):
            train = self.load(path / 'train.py')  # [:100000]
            valid = self.load(path / 'valid.py')  # [:100000]

        from labml.utils.cache import cache_get

        super().__init__(path, tokenizer, train, valid, '',
                         n_tokens=cache_get('n_tokens'),
                         itos=cache_get('itos'),
                         stoi=cache_get('stoi'))


class BPESourceCodeDataset(TextDataset):
    tokenizer: BPE

    def __init__(self, path: PurePath, bpe: BPE):
        with monit.section("Load data"):
            train = self.load(path / 'train.py')  # [:100_000]
            valid = self.load(path / 'valid.py')  # [:100_000]

        super().__init__(path, bpe, train, valid, '',
                         n_tokens=bpe.n_tokens,
                         itos=bpe.itos,
                         stoi=bpe.stoi)

    def text_to_i(self, text: str) -> torch.Tensor:
        return torch.tensor(self.tokenizer.encode(text))


class Configs(TrainValidConfigs):
    optimizer: torch.optim.Adam
    device: torch.device = DeviceConfigs()

    model: Module
    text: TextDataset
    batch_size: int = 16
    seq_len: int = 512
    n_tokens: int
    n_layers: int = 2
    dropout: float = 0.2
    d_model: int = 512
    rnn_size: int = 512
    rhn_depth: int = 1
    tokenizer: Callable
    inner_iterations = 100

    is_save_models = True

    transformer: TransformerConfigs

    accuracy = Accuracy()
    loss_func: 'CrossEntropyLoss'

    state_updater: 'StateUpdater'
    state = SimpleStateModule()
    mem_len: int = 512
    grad_norm_clip: float = 1.0
    is_token_by_token: bool = False

    itos: List[str]
    stoi: Dict[str, int]

    def init(self):
        tracker.set_queue("loss.*", 20, True)
        tracker.set_scalar("accuracy.*", True)
        hook_model_outputs(self.mode, self.model, 'model')
        self.state_modules = [self.accuracy, self.state]

    def step(self, batch: any, batch_idx: BatchIndex):
        data, target = batch[0].to(self.device), batch[1].to(self.device)

        if self.mode.is_train:
            tracker.add_global_step(len(data))

        with self.mode.update(is_log_activations=batch_idx.is_last):
            state = self.state.get()
            output, new_state = self.model(data, state)
            state = self.state_updater(state, new_state)
            self.state.set(state)

        loss = self.loss_func(output, target)
        tracker.add("loss.", loss)

        self.accuracy(output, target)
        self.accuracy.track()

        if self.mode.is_train:
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_norm_clip)
            self.optimizer.step()
            if batch_idx.is_last:
                tracker.add('model', self.model)
            self.optimizer.zero_grad()

        tracker.save()

    def sample(self):
        prompt = 'def train('
        log = [(prompt, Text.subtle)]
        state = None
        for i in monit.iterate('Sample', 25):
            data = self.text.text_to_i(prompt).unsqueeze(-1)
            data = data.to(self.device)
            output, new_state = self.model(data, state)
            output = output.argmax(dim=-1).squeeze(1)
            prompt += '' + self.itos[output[-1]]
            if self.is_token_by_token:
                prompt = prompt[-1:]
            log += [('' + self.itos[output[-1]], Text.value)]
            state = self.state_updater(state, new_state)

        logger.log(log)


@option(Configs.transformer)
def default_transformer(c: Configs):
    conf = TransformerConfigs()
    conf.d_model = c.d_model
    conf.n_layers = c.n_layers
    conf.n_src_vocab = c.n_tokens
    conf.n_tgt_vocab = c.n_tokens
    conf.dropout = c.dropout

    return conf


@option(Configs.optimizer)
def _optimizer(c: Configs):
    optimizer = OptimizerConfigs()
    optimizer.parameters = c.model.parameters()
    optimizer.optimizer = 'Adam'
    optimizer.d_model = c.d_model

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
    from labml.utils.cache import cache
    return cache('n_tokens', lambda: c.text.n_tokens)


@option(Configs.itos)
def _itos(c: Configs):
    from labml.utils.cache import cache
    return cache('itos', lambda: c.text.itos)


@option(Configs.stoi)
def _stoi(c: Configs):
    from labml.utils.cache import cache
    return cache('stoi', lambda: c.text.stoi)


@option(Configs.model)
def lstm_model(c: Configs):
    from python_autocomplete.models.lstm import LstmModel
    m = LstmModel(n_tokens=c.n_tokens,
                  embedding_size=c.d_model,
                  hidden_size=c.rnn_size,
                  n_layers=c.n_layers)
    return m.to(c.device)


@option(Configs.model)
def rhn_model(c: Configs):
    from python_autocomplete.models.highway import RhnModel
    m = RhnModel(n_tokens=c.n_tokens,
                 embedding_size=c.d_model,
                 hidden_size=c.rnn_size,
                 n_layers=c.n_layers,
                 depth=c.rhn_depth)
    return m.to(c.device)


@option(Configs.model)
def transformer_model(c: Configs):
    from python_autocomplete.models.transformer import TransformerModel
    m = TransformerModel(n_tokens=c.n_tokens,
                         d_model=c.d_model,
                         encoder=c.transformer.encoder,
                         src_embed=c.transformer.src_embed)

    return m.to(c.device)


@option(Configs.model)
def transformer_xl_model(c: Configs):
    from labml_nn.transformers.xl import RelativeMultiHeadAttention
    from labml_nn.transformers.feed_forward import FeedForward
    from labml_nn.transformers.xl import TransformerXL
    from labml_nn.transformers.xl import TransformerXLLayer
    from python_autocomplete.models.xl import TransformerXLModel
    m = TransformerXLModel(c.n_tokens, c.d_model, TransformerXL(
        TransformerXLLayer(d_model=c.d_model,
                           self_attn=RelativeMultiHeadAttention(c.transformer.n_heads, c.d_model, c.dropout),
                           feed_forward=FeedForward(c.d_model, c.transformer.ffn.d_ff, c.dropout),
                           dropout_prob=c.dropout), c.n_layers))
    return m.to(c.device)


class StateUpdater:
    def __call__(self, old_state, new_state):
        return new_state


class MemoryUpdater(StateUpdater):
    def __init__(self, mem_len: int):
        self.mem_len = mem_len

    def __call__(self, old_mem, new_mem):
        if self.mem_len == 0:
            return []

        if old_mem:
            mem = [torch.cat((m, x), dim=0) for m, x in zip(old_mem, new_mem)]
        else:
            mem = new_mem

        if len(mem[0]) > self.mem_len:
            mem = [m[-self.mem_len:] for m in mem]

        return mem


@option(Configs.state_updater)
def simple():
    return StateUpdater()


@option(Configs.state_updater)
def transformer_memory(c: Configs):
    return MemoryUpdater(c.mem_len)


def character_tokenizer(x: str):
    return list(x)


@option(Configs.tokenizer)
def character():
    return character_tokenizer


@option(Configs.text)
def source_code(c: Configs):
    return SourceCodeDataset(lab.get_data_path(), c.tokenizer)


@option(Configs.text)
def source_code_bpe(c: Configs):
    from labml.utils.cache import cache_get
    from python_autocomplete.bpe import BPEEnDe
    bpe_cache = cache_get('bpe')

    if bpe_cache:
        bpe_en_de = BPEEnDe()
        bpe_en_de.load(**bpe_cache)
    else:
        raise RuntimeError('BPE not cached')

    tokenizer = BPE(bpe_en_de, SourceCodeTokenizer())
    return BPESourceCodeDataset(lab.get_data_path(), tokenizer)


@option(Configs.train_loader)
def sequential_train_loader(c: Configs):
    return SequentialDataLoader(text=c.text.train,
                                dataset=c.text,
                                batch_size=c.batch_size,
                                seq_len=c.seq_len)


@option(Configs.valid_loader)
def sequential_valid_loader(c: Configs):
    return SequentialDataLoader(text=c.text.valid,
                                dataset=c.text,
                                batch_size=c.batch_size,
                                seq_len=c.seq_len)


def transpose_batch(batch):
    transposed_data = list(zip(*batch))
    src = torch.stack(transposed_data[0], 1)
    tgt = torch.stack(transposed_data[1], 1)

    return src, tgt


@option(Configs.train_loader)
def shuffled_train_loader(c: Configs):
    return DataLoader(SequentialUnBatchedDataset(text=c.text.train,
                                                 dataset=c.text,
                                                 seq_len=c.seq_len),
                      batch_size=c.batch_size,
                      collate_fn=transpose_batch,
                      shuffle=True)


@option(Configs.valid_loader)
def shuffled_valid_loader(c: Configs):
    return DataLoader(SequentialUnBatchedDataset(text=c.text.valid,
                                                 dataset=c.text,
                                                 seq_len=c.seq_len),
                      batch_size=c.batch_size,
                      collate_fn=transpose_batch,
                      shuffle=True)


def main():
    conf = Configs()
    # Assign one of transformer_mode, lstm_model, or rhn_model
    experiment.create(name="source_code",
                      comment='bpe')
    experiment.configs(conf, {
        # 'text': 'source_code',
        'text': 'source_code_bpe',
        'model': 'transformer_model',
        # 'model': 'transformer_xl_model',
        'n_layers': 6,
        'batch_size': 12,
        'epochs': 32,
        'optimizer.optimizer': 'Noam',
        'optimizer.learning_rate': 1.0,
        'device.cuda_device': 0,
        'seq_len': 512,
        'is_token_by_token': False,
        'state_updater': 'simple',
        'train_loader': 'shuffled_train_loader',
        'valid_loader': 'shuffled_valid_loader',
        # 'train_loader': 'sequential_train_loader',
        # 'valid_loader': 'sequential_valid_loader',
    })
    experiment.add_pytorch_models(model=conf.model)
    with experiment.start():
        conf.run()


if __name__ == '__main__':
    main()
