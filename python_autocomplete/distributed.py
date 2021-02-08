import datetime

import torch.distributed
import torch.nn as nn

from labml import experiment, monit, logger, tracker
from labml.configs import option
from labml.logger import Text, inspect
from labml_helpers.train_valid import hook_model_outputs, BatchIndex
from python_autocomplete.train import Configs as Configs_


class Configs(Configs_):
    ddp_model: nn.parallel.DistributedDataParallel

    def init(self):
        tracker.set_queue("loss.*", 20, True)
        tracker.set_scalar("accuracy.*", True)
        hook_model_outputs(self.mode, self.ddp_model, 'model')
        self.state_modules = [self.accuracy]

    def step(self, batch: any, batch_idx: BatchIndex):
        data, target = batch[0].to(self.device), batch[1].to(self.device)

        if self.mode.is_train:
            tracker.add_global_step(len(data))

        with self.mode.update(is_log_activations=batch_idx.is_last):
            output, *_ = self.ddp_model(data)

        loss = self.loss_func(output, target)
        self.accuracy(output, target)
        self.accuracy.track()
        tracker.add("loss.", loss)

        if self.mode.is_train:
            loss.backward()

            self.optimizer.step()
            if batch_idx.is_last:
                tracker.add('model', self.ddp_model)
            self.optimizer.zero_grad()

        tracker.save()

    def sample(self):
        prompt = 'def train('
        log = [(prompt, Text.subtle)]
        for i in monit.iterate('Sample', 25):
            data = self.text.text_to_i(prompt).unsqueeze(-1)
            data = data.to(self.device)
            output, *_ = self.ddp_model(data)
            output = output.argmax(dim=-1).squeeze()
            prompt += '' + self.text.itos[output[-1]]
            log += [('' + self.text.itos[output[-1]], Text.value)]

        logger.log(log)


@option(Configs.ddp_model)
def ddp_model(c: Configs):
    return nn.parallel.DistributedDataParallel(c.model, device_ids=[c.device])


def main(local_rank, rank, world_size, uuid, init_method: str = 'tcp://localhost:23456'):
    with monit.section('Distributed'):
        torch.distributed.init_process_group("gloo",
                                             timeout=datetime.timedelta(seconds=30),
                                             init_method=init_method,
                                             rank=rank,
                                             world_size=world_size)
    conf = Configs()
    experiment.create(uuid=uuid,
                      name="source_code_ddp",
                      comment='lstm model')
    experiment.distributed(local_rank, world_size)
    experiment.configs(conf, {
        'model': 'transformer_model',
        'n_layers': 6,
        'batch_size': 12,
        'epochs': 32,
        'optimizer.optimizer': 'Noam',
        'optimizer.learning_rate': 1.0,
        'device.cuda_device': local_rank,
        'seq_len': 512,
        'train_loader': 'shuffled_train_loader',
        'valid_loader': 'shuffled_valid_loader'
    })
    experiment.add_pytorch_models(model=conf.ddp_model)
    with experiment.start():
        conf.run()


def _launcher():
    import os
    world_size = int(os.environ['WORLD_SIZE'])
    run_uuid = os.environ['RUN_UUID']
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    inspect(world_size=os.environ['WORLD_SIZE'],
            run_uuid=os.environ['RUN_UUID'],
            local_rank=os.environ['LOCAL_RANK'],
            rank=os.environ['RANK'],
            master_addr=os.environ['MASTER_ADDR'],
            master_port=os.environ['MASTER_PORT'])
    main(local_rank, rank, world_size, run_uuid, 'env://')


if __name__ == '__main__':
    _launcher()
