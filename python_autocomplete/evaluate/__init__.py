from heapq import heappush, heappop
from typing import Any, Tuple, List, Optional, NamedTuple

import torch
import torch.nn
from torch import nn

from labml_helpers.module import Module
from python_autocomplete.dataset import Tokenizer, ID_CHARS
from python_autocomplete.train import StateUpdater

EPS_PROB = 1e-6
MIN_BEAM_PROB = 1e-4


class PredictionComplete:
    def __call__(self, text, token_str: str):
        raise NotImplementedError


class NextWordPredictionComplete(PredictionComplete):
    def __init__(self, prompt: str, rest: str, min_length: int):
        self.min_length = min_length
        self.rest = rest
        self.is_id = False
        if prompt and prompt[-1] in ID_CHARS:
            self.is_id = True

    def __call__(self, text, token_str: str):
        if len(text) - len(self.rest) < self.min_length:
            return False

        prev_is_id = text[-1] in ID_CHARS
        last_is_id = token_str[-1] in ID_CHARS

        return prev_is_id != last_is_id


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

    def add_prediction_before_token(self, prob: float, beam_idx: int, state):
        if len(self.result_heap) == self.max_beam_size:
            if self.result_heap[0][0] > prob - EPS_PROB:
                return False
            heappop(self.result_heap)

        state = self.state_updater.get_from_batch(state, beam_idx)
        text = self.text[beam_idx]
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

    def update(self, next_token, itos: List[str], state, old_state):
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
                    if not self.add_prediction_before_token(self.probs[b], b, old_state):
                        break
                    else:
                        break
                    # if not self.add_prediction(self.probs[b] * tokens[token].item(), b, token_str, state):
                    #     break
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
            next_token, new_state = self._get_predictions(prompt, state)
            beam.update(next_token, self.tokenizer.itos, new_state, state)
            prompt, state = beam.next_batch(prompt, new_state, self.tokenizer.itos)

            if prompt is None:
                break

        results = [Prediction(r[0], r[1][0], r[1][1]) for r in beam.result_heap]
        return results

    def rstrip(self, prompt: str) -> Tuple[str, List[int]]:
        return self.tokenizer.rstrip(prompt)
