import torch
import torch.nn as nn

from labml import experiment, monit, logger, tracker
from labml.configs import option
from labml.logger import Text
from labml_helpers.device import DeviceConfigs
from labml_helpers.metrics.accuracy import Accuracy
from labml_helpers.metrics.simple_state import SimpleStateModule
from labml_helpers.module import Module
from labml_helpers.train_valid import TrainValidConfigs, hook_model_outputs, BatchIndex
from labml_nn.optimizers.configs import OptimizerConfigs
from labml_nn.transformers import TransformerConfigs
from python_autocomplete.dataset.dataset import SourceCodeDataConfigs


class Configs(TrainValidConfigs):
    optimizer: torch.optim.Adam
    device: torch.device = DeviceConfigs()

    model: Module
    text = SourceCodeDataConfigs()
    n_tokens: int
    n_layers: int = 2
    dropout: float = 0.2
    d_model: int = 512
    rnn_size: int = 512
    rhn_depth: int = 1
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

    def init(self):
        tracker.set_queue("loss.*", 20, True)
        tracker.set_scalar("accuracy.*", True)
        hook_model_outputs(self.mode, self.model, 'model')
        self.state_modules = [self.accuracy, self.state]

    def step(self, batch: any, batch_idx: BatchIndex):
        data, target = batch[0].to(self.device), batch[1].to(self.device)

        if self.mode.is_train:
            tracker.add_global_step(target.shape[0] * target.shape[1])

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
            prompt += '' + self.text.tokenizer.itos[output[-1]]
            if self.is_token_by_token:
                prompt = self.text.tokenizer.itos[output[-1]]
            else:
                prompt += '' + self.text.tokenizer.itos[output[-1]]
            log += [('' + self.text.tokenizer.itos[output[-1]], Text.value)]
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
    return c.text.tokenizer.n_tokens


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

    def get_from_batch(self, state, batch_idx):
        if state is None:
            return None
        elif isinstance(state, torch.Tensor):
            return state[batch_idx]
        elif isinstance(state, tuple):
            return tuple(s[batch_idx] for s in state)
        elif isinstance(state, list):
            return [s[batch_idx] for s in state]

    def make_batch(self, batch):
        assert isinstance(batch, list)
        if batch[0] is None:
            return None
        elif isinstance(batch[0], torch.Tensor):
            return torch.stack(batch)
        elif isinstance(batch[0], tuple):
            return tuple(torch.stack([b[n] for b in batch]) for n in range(len(batch[0])))
        elif isinstance(batch[0], list):
            return [torch.stack([b[n] for b in batch]) for n in range(len(batch[0]))]


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

    def get_from_batch(self, state, batch_idx):
        if state is None:
            return None

        return [m[:, batch_idx] for m in state]

    def make_batch(self, batch):
        if batch[0] is None:
            return None
        return [torch.stack([b[n] for b in batch], dim=1) for n in range(len(batch[0]))]


@option(Configs.state_updater)
def simple():
    return StateUpdater()


@option(Configs.state_updater)
def transformer_memory(c: Configs):
    return MemoryUpdater(c.mem_len)


@option(Configs.train_loader)
def _train_loader(c: Configs):
    return c.text.train_loader


@option(Configs.valid_loader)
def _valid_loader(c: Configs):
    return c.text.valid_loader


def main():
    conf = Configs()
    # Assign one of transformer_mode, lstm_model, or rhn_model
    experiment.create(name="source_code",
                      comment='bpe')
    experiment.configs(conf, {
        # 'model': 'transformer_model',
        'model': 'transformer_xl_model',
        'n_layers': 6,
        'epochs': 32,
        'optimizer.optimizer': 'AdamW',
        'optimizer.learning_rate': 1.25e-4,
        'device.cuda_device': 0,

        'is_token_by_token': True,
        'state_updater': 'transformer_memory',
        'mem_len': 256,

        'text.is_shuffle': False,
        'text.tokenizer': 'bpe',
        'text.batch_size': 12,
        'text.seq_len': 256,
        #
        # 'inner_iterations': 10,
        # 'text.truncate_data': 100_000,
    })
    experiment.add_pytorch_models(model=conf.model)
    with experiment.start():
        conf.run()


if __name__ == '__main__':
    main()
