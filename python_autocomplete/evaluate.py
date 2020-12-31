import string
from typing import List, Dict

import torch
import torch.nn
from labml.utils.cache import cache
from torch import nn

from labml import experiment, logger, lab
from labml.logger import Text, Style
from labml.utils.pytorch import get_modules
from labml_helpers.module import Module
from python_autocomplete.train import Configs


class Predictor:
    def __init__(self, model: Module, stoi: Dict[str, int], itos: List[str]):
        self.stoi = stoi
        self.itos = itos
        self.model = model

        # Initial state
        self._state = None

        # For timing
        self.time_add = 0
        self.time_predict = 0
        self.time_check = 0

    def _get_predictions(self, prompt: str) -> torch.Tensor:
        prompt = prompt[-512:]
        data = torch.tensor([[self.stoi[c]] for c in prompt if c in self.stoi],
                            dtype=torch.long,
                            device=self.model.device)

        # Get predictions
        with torch.no_grad():
            prediction, *_ = self.model(data)

        # Final prediction
        return prediction[-1, :, :]

    def get_predictions(self, prompt: str) -> torch.Tensor:
        prediction = self._get_predictions(prompt)

        return prediction.detach().cpu().numpy()

    def get_probabilities(self, prompt: str) -> torch.Tensor:
        # Final prediction
        prediction = nn.Softmax(-1)(self._get_predictions(prompt))

        return prediction.detach().cpu().numpy()

    def get_next_char(self, prompt: str) -> str:
        prediction = self.get_predictions(prompt)
        best = prediction.argmax(-1).squeeze().item()
        return self.itos[best]

    def get_token(self, prompt: str) -> str:
        result = ''
        alnum = set(string.ascii_letters + string.digits + ' ' + '\n' + '\r')
        while True:
            next_char = self.get_next_char(prompt)
            if len(result) > 2 and next_char not in alnum or (next_char.strip() == '' and result.strip() != ''):
                if not result:
                    result += next_char
                return result
            result += next_char
            if len(result) > 20:
                return result
            prompt += next_char


def evaluate(predictor: Predictor, text: str):
    line_no = 1
    logs = [(f"{line_no: 4d}: ", Text.meta), (text[0], Text.subtle)]

    correct = 0
    i = 0
    right = False
    key_strokes = 0

    while i + 1 < len(text):
        next_token = predictor.get_token(text[:i + 1])
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
        preds = predictor.get_probabilities(text[:i + 1])[0, :]
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
            c = predictor.get_next_char(text[:i + 1])

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


def main():
    conf = Configs()
    experiment.evaluate()

    # Replace this with your training experiment UUID
    run_uuid = '39b03a1e454011ebbaff2b26e3148b3d'

    conf_dict = experiment.load_configs(run_uuid)
    experiment.configs(conf, conf_dict)
    experiment.add_pytorch_models(get_modules(conf))
    experiment.load(run_uuid)

    experiment.start()
    predictor = Predictor(conf.model, cache('stoi', lambda: conf.text.stoi), cache('itos', lambda: conf.text.itos))
    conf.model.eval()

    with open(str(lab.get_data_path() / 'sample.py'), 'r') as f:
        sample = f.read()
    evaluate(predictor, sample)


if __name__ == '__main__':
    main()
