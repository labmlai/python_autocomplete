import string
from typing import List, Dict, Set, Optional, Any, Tuple

import numpy as np
import torch
import torch.nn
from torch import nn

from labml import experiment, logger, lab
from labml.logger import Text, Style
from labml.utils.pytorch import get_modules
from labml_helpers.module import Module
from python_autocomplete.train import Configs, StateUpdater


class Predictor:
    def __init__(self, model: Module, stoi: Dict[str, int], itos: List[str], *,
                 state_updater: StateUpdater,
                 is_token_by_token: bool):
        self.is_token_by_token = is_token_by_token
        self.state_updater = state_updater
        self.stoi = stoi
        self.itos = itos
        self.model = model

        # For timing
        self.time_add = 0
        self.time_predict = 0
        self.time_check = 0

    def _get_predictions(self, prompt: str, state: Any) -> Tuple[torch.Tensor, Any]:
        prompt = prompt[-512:]
        data = torch.tensor([[self.stoi[c]] for c in prompt if c in self.stoi],
                            dtype=torch.long,
                            device=self.model.device)

        # Get predictions
        with torch.no_grad():
            prediction, new_state = self.model(data, state)

        state = self.state_updater(state, new_state)

        # Final prediction
        return prediction[-1, :, :], state

    def get_predictions(self, prompt: str, state: Any) -> Tuple[np.ndarray, Any]:
        prediction, state = self._get_predictions(prompt, state)

        return prediction.detach().cpu().numpy(), state

    def get_probabilities(self, prompt: str, state: Any) -> Tuple[np.ndarray, Any]:
        # Final prediction
        prediction, state = self._get_predictions(prompt, state)
        prediction = nn.Softmax(-1)(prediction)

        return prediction.detach().cpu().numpy(), state

    def get_next_char(self, prompt: str, state: Any) -> Tuple[str, Any]:
        prediction, state = self.get_predictions(prompt, state)
        best = prediction.argmax(-1).squeeze().item()
        return self.itos[best], state

    def get_token(self, prompt: str, token_chars: Optional[Set[str]], state: Any) -> Tuple[str, Any]:
        result = ''
        if token_chars is None:
            token_chars = set(string.ascii_letters + string.digits + ' ' + '\n' + '\r')
        while True:
            next_char, state = self.get_next_char(prompt, state)
            if len(result) > 2 and next_char not in token_chars or (next_char.strip() == '' and result.strip() != ''):
                if not result:
                    result += next_char
                return result, state
            result += next_char
            if len(result) > 20:
                return result, state
            prompt += next_char
            if self.is_token_by_token:
                prompt = prompt[-1:]


def evaluate(predictor: Predictor, text: str):
    line_no = 1
    logs = [(f"{line_no: 4d}: ", Text.meta), (text[0], Text.subtle)]

    correct = 0
    i = 0
    right = False
    key_strokes = 0

    while i + 1 < len(text):
        next_token, state = predictor.get_token(text[:i + 1], None, None)
        if next_token == text[i + 1: i + 1 + len(next_token)]:
            correct += len(next_token)
            right = True
        else:
            next_token = text[i + 1]
            right = False

        for j, c in enumerate(next_token):
            if c == '\n':
                logger.log(logs)
                line_no += 1
                logs = [(f"{line_no: 4d}: ", Text.meta)]
            elif c == '\r':
                continue
            else:
                if right:
                    if j == 0:
                        logs.append((c, [Text.meta, Style.underline]))
                    else:
                        logs.append((c, [Text.success, Style.underline]))
                else:
                    logs.append((c, [Text.warning]))

        i += len(next_token)
        key_strokes += 1

    logger.log(logs)

    logger.inspect(accuracy=correct / (len(text) - 1),
                   key_strokes=key_strokes,
                   length=len(text))


def anomalies(predictor: Predictor, text: str):
    line_no = 1
    logs = [(f"{line_no: 4d}: ", Text.meta), (text[0], Text.subtle)]

    i = 0

    while i + 1 < len(text):
        #             print(i, self.predictor.prompt)
        preds, _ = predictor.get_probabilities(text[:i + 1], None)
        preds = preds[0, :]
        c = text[i + 1]

        if c == '\n':
            logger.log(logs)
            line_no += 1
            logs = [(f"{line_no: 4d}: ", Text.meta)]
        elif c == '\r':
            continue
        elif c not in predictor.stoi:
            logs.append(c)
        else:
            next_id = predictor.stoi[c]
            prob = preds[next_id]
            if prob > 0.9:
                logs.append((c, [Style.bold, Text.success, Style.underline]))
            elif prob > 0.75:
                logs.append((c, [Text.success, Style.underline]))
            elif prob > 0.2:
                logs.append(c)
            elif prob > 0.1:
                logs.append((c, [Text.warning, Style.underline]))
            elif prob > 0.01:
                logs.append((c, [Style.bold, Text.warning, Style.underline]))
            elif prob > 0.001:
                logs.append((c, [Text.danger, Style.underline]))
            else:
                logs.append((c, [Style.bold, Text.danger, Style.underline]))

        i += 1

    logger.log(logs)


def complete(predictor: Predictor, text: str, completion: int):
    line_no = 1
    logs = [(f"{line_no: 4d}: ", Text.meta), (text[0], Text.subtle)]

    i = 0
    given = len(text)

    while i + 1 < given + completion:
        if len(text) > i + 1:
            c = text[i + 1]
        else:
            c, _ = predictor.get_next_char(text[:i + 1], None)

        if c == '\n':
            logger.log(logs)
            line_no += 1
            logs = [(f"{line_no: 4d}: ", Text.meta)]
        elif c != '\r':
            if len(text) > i + 1:
                logs.append(c)
            else:
                logs.append((c, [Style.bold]))

        if len(text) <= i + 1:
            text += c

        i += 1

    logger.log(logs)


def get_predictor():
    conf = Configs()
    experiment.evaluate()

    # This will download a pretrained model checkpoint and some cached files.
    # It will download the archive as `saved_checkpoint.tar.gz` and extract it.
    #
    # If you have a locally trained model load it directly with
    # run_uuid = 'RUN_UUID'
    # And for latest checkpoint
    # checkpoint = None

    run_uuid = 'c45857026a2811eba16c27c69839e51f'
    checkpoint = None
    # run_uuid, checkpoint = experiment.load_bundle(
    #     lab.get_path() / 'saved_checkpoint.tar.gz',
    #     url='https://github.com/lab-ml/python_autocomplete/releases/download/0.0.4/transformer_checkpoint.tar.gz')

    conf_dict = experiment.load_configs(run_uuid)
    experiment.configs(conf, conf_dict)
    experiment.add_pytorch_models(get_modules(conf))
    experiment.load(run_uuid, checkpoint)

    experiment.start()
    conf.model.eval()
    return Predictor(conf.model, conf.stoi, conf.itos,
                     state_updater=conf.state_updater,
                     is_token_by_token=conf.is_token_by_token)


def main():
    predictor = get_predictor()

    with open(str(lab.get_data_path() / 'sample.py'), 'r') as f:
        sample = f.read()
    evaluate(predictor, sample)


if __name__ == '__main__':
    main()
