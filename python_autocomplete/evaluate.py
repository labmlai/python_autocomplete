from heapq import heappush, heappop
from typing import Any, Tuple, List, Optional, NamedTuple

import torch
import torch.nn
from torch import nn

from labml import experiment, logger, lab, monit
from labml.logger import Text, Style
from labml.utils.pytorch import get_modules
from labml_helpers.module import Module
from python_autocomplete.dataset import Tokenizer, ID_CHARS
from python_autocomplete.train import Configs, StateUpdater

EPS_PROB = 1e-6
MIN_BEAM_PROB = 1e-4

class PredictionComplete:
    def __call__(self, text, token_str: str):
        raise NotImplementedError


class NextWordPredictionComplete(PredictionComplete):
    def __init__(self, prompt: str):
        self.is_id = False
        if prompt and prompt[-1] in ID_CHARS:
            self.is_id = True

    def __call__(self, text, token_str: str):
        prediction = set(token_str)
        intersection = prediction.intersection(ID_CHARS)
        is_id = len(intersection) > 0 and intersection == prediction
        is_not_id = intersection != prediction
        if is_id and is_not_id:
            return True
        return is_id == self.is_id


class BeamSearch:
    def __init__(self, beam_size: int, prediction_complete: PredictionComplete,
                 max_beam_size: int, rest: str,
                 state_updater: 'StateUpdater',
                 probs: Optional[List[float]],
                 is_token_by_token: bool):
        self.is_token_by_token = is_token_by_token
        self.state_updater = state_updater
        self.prediction_complete = prediction_complete
        self.max_beam_size = max_beam_size
        self.rest = rest

        if probs is None:
            probs = [1 / beam_size] * beam_size
        assert len(probs) == beam_size
        self.probs = probs

        self.result_heap = []
        self.text = [''] * beam_size
        self.beam_heap = []

    @staticmethod
    def is_substr(original, token_str):
        if not original:
            return True

        n = min(len(original), len(token_str))
        return original[:n] == token_str[:n]

    def add_prediction(self, prob: float, beam_idx: int, token_str: str, state):
        if len(self.result_heap) == self.max_beam_size:
            if self.result_heap[0][0] > prob - EPS_PROB:
                return False
            heappop(self.result_heap)

        state = self.state_updater.get_from_batch(state, beam_idx)
        text = self.text[beam_idx] + token_str
        heappush(self.result_heap, (prob, (text, state)))

        return True

    def add_beam(self, prob: float, beam_idx: int, token: int):
        if self.result_heap and self.result_heap[0][0] > prob - EPS_PROB:
            return False

        if prob < MIN_BEAM_PROB:
            return False

        if len(self.beam_heap) == self.max_beam_size:
            if self.beam_heap[0][0] > prob - EPS_PROB:
                return False
            heappop(self.beam_heap)

        heappush(self.beam_heap, (prob, (beam_idx, token)))

        return True

    def next_batch(self, prompt: torch.Tensor, state: Any, itos: List[str]):
        if not self.beam_heap:
            return None, None

        new_prompt = []
        new_state = []

        texts = self.text
        self.text = []
        self.probs = []

        for prob, (b, token) in self.beam_heap:
            token = prompt.new_tensor([token])
            if self.is_token_by_token:
                new_prompt.append(token)
            else:
                new_prompt.append(torch.cat((prompt[1:, b], token)))
            new_state.append(self.state_updater.get_from_batch(state, b))
            self.probs.append(prob)
            self.text.append(texts[b] + itos[token])

        new_prompt = torch.stack(new_prompt, dim=1)
        new_state = self.state_updater.make_batch(new_state)

        self.beam_heap = []

        return new_prompt, new_state

    def update(self, next_token, itos: List[str], state):
        self.beam_heap = []

        for b, text in enumerate(self.text):
            text = self.text[b]
            if len(text) >= len(self.rest):
                check_rest = None
            else:
                check_rest = self.rest[len(text):]

            tokens = next_token[b]
            sort_idx = torch.argsort(tokens)

            for i in reversed(range(len(tokens))):
                token = sort_idx[i]
                token_str = itos[token]
                if not self.is_substr(check_rest, token_str):
                    continue

                if self.prediction_complete(text, token_str):
                    if not self.add_prediction(self.probs[b] * tokens[token].item(), b, token_str, state):
                        break
                elif not self.add_beam(self.probs[b] * tokens[token].item(), b, token):
                    break


class Prediction(NamedTuple):
    prob: float
    text: str
    state: Any


class Predictor:
    def __init__(self, model: Module, tokenizer: Tokenizer, *,
                 state_updater: StateUpdater,
                 is_token_by_token: bool):
        self.tokenizer = tokenizer
        self.is_token_by_token = is_token_by_token
        self.state_updater = state_updater
        self.model = model

        # For timing
        self.time_add = 0
        self.time_predict = 0
        self.time_check = 0

    def _get_predictions(self, prompt: torch.Tensor, state: Any) -> Tuple[torch.Tensor, Any]:
        if prompt.shape[0] == 0:
            return prompt.new_ones(prompt.shape[1], len(self.tokenizer.itos)) / len(self.tokenizer.itos), state
        prompt = prompt.to(self.model.device)

        # Get predictions
        with torch.no_grad():
            prediction, new_state = self.model(prompt, state)

        state = self.state_updater(state, new_state)
        prediction = nn.Softmax(-1)(prediction[-1])

        # Final prediction
        return prediction, state

    def get_next_word(self, prompt: torch.Tensor, state: Any, rest: str, probs: List[float],
                      prediction_complete: PredictionComplete,
                      max_beam_size: int) -> \
            List[Prediction]:
        beam = BeamSearch(prompt.shape[1], prediction_complete, max_beam_size, rest, self.state_updater,
                          probs, self.is_token_by_token)

        for _ in range(10):
            next_token, state = self._get_predictions(prompt, state)
            beam.update(next_token, self.tokenizer.itos, state)
            prompt, state = beam.next_batch(prompt, state, self.tokenizer.itos)

            if prompt is None:
                break

        results = [Prediction(r[0], r[1][0], r[1][1]) for r in beam.result_heap]
        return results

    def rstrip(self, prompt: str) -> Tuple[str, List[int]]:
        return self.tokenizer.rstrip(prompt)


def evaluate(predictor: Predictor, text: str):
    line_no = 1
    logs = [(f"{line_no: 4d}: ", Text.meta), (text[0], Text.subtle)]

    correct = 0
    i = 0
    key_strokes = 0

    while i + 1 < len(text):
        prefix = text[:i + 1]
        stripped, prompt = predictor.rstrip(prefix)
        rest = prefix[len(stripped):]
        prediction_complete = NextWordPredictionComplete(stripped)
        prompt = torch.tensor(prompt, dtype=torch.long).unsqueeze(-1)

        predictions = predictor.get_next_word(prompt, None, rest, [1.], prediction_complete, 5)
        predictions.sort(key=lambda x: -x[0])
        if predictions:
            next_token = predictions[0].text[len(rest):]
        else:
            next_token = ''

        if next_token and next_token == text[i + 1: i + 1 + len(next_token)]:
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
        preds, _ = predictor.get_predictions(text[:i + 1], None, calc_probs=True)
        preds = preds[0, :]
        c = text[i + 1]

        if c == '\n':
            logger.log(logs)
            line_no += 1
            logs = [(f"{line_no: 4d}: ", Text.meta)]
        elif c == '\r':
            continue
        elif c not in predictor.tokenizer.stoi:
            logs.append(c)
        else:
            next_id = predictor.tokenizer.stoi[c]
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
            c, _ = predictor.get_next_token(text[:i + 1], None)

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


def get_predictor() -> Predictor:
    conf = Configs()
    experiment.evaluate()

    # This will download a pretrained model checkpoint and some cached files.
    # It will download the archive as `saved_checkpoint.tar.gz` and extract it.
    #
    # If you have a locally trained model load it directly with
    # run_uuid = 'RUN_UUID'
    # And for latest checkpoint
    # checkpoint = None

    run_uuid = 'a6cff3706ec411ebadd9bf753b33bae6'  # bpe
    checkpoint = None
    # run_uuid, checkpoint = experiment.load_bundle(
    #     lab.get_path() / 'saved_checkpoint.tar.gz',
    #     url='https://github.com/lab-ml/python_autocomplete/releases/download/0.0.4/transformer_checkpoint.tar.gz')

    conf_dict = experiment.load_configs(run_uuid)
    conf_dict['text.is_load_data'] = False
    experiment.configs(conf, conf_dict)
    experiment.add_pytorch_models(get_modules(conf))
    experiment.load(run_uuid, checkpoint)

    experiment.start()
    conf.model.eval()
    return Predictor(conf.model, conf.text.tokenizer,
                     state_updater=conf.state_updater,
                     is_token_by_token=conf.is_token_by_token)


def main():
    predictor = get_predictor()

    with open(str(lab.get_data_path() / 'sample.py'), 'r') as f:
        sample = f.read()
    with monit.section('Evaluate'):
        evaluate(predictor, sample)


if __name__ == '__main__':
    main()
