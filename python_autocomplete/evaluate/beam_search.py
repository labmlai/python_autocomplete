from heapq import heappush, heappop
from typing import Any, List, Optional

import torch
import torch.nn

from python_autocomplete.dataset import ID_CHARS
from python_autocomplete.train import StateUpdater

EPS_PROB = 1e-6
MIN_BEAM_PROB = 1e-4


class PredictionComplete:
    def __call__(self, text, token_str: str):
        raise NotImplementedError


class NextWordPredictionComplete(PredictionComplete):
    def __init__(self, rest: str, min_length: int):
        self.min_length = min_length
        self.rest = rest

    def __call__(self, text, token_str: str):
        if len(text) - len(self.rest) < self.min_length:
            return False

        prev_is_id = text[-1] in ID_CHARS
        last_is_id = token_str[-1] in ID_CHARS

        return prev_is_id != last_is_id


class NextWordNewLinePredictionComplete(PredictionComplete):
    def __init__(self, rest: str, min_length: int):
        self.min_length = min_length
        self.rest = rest

    def __call__(self, text, token_str: str):
        if len(text) - len(self.rest) < self.min_length:
            return False

        if '\n' in token_str:
            return True

        prev_is_id = text[-1] in ID_CHARS
        last_is_id = token_str[-1] in ID_CHARS

        return prev_is_id != last_is_id


class BeamSearch:
    def __init__(self):
        pass

    def next_batch(self, prompt: torch.Tensor, state: Any):
        raise NotImplementedError

    def update(self, next_token, state, old_state):
        raise NotImplementedError


class BeamSearchSimple(BeamSearch):
    def __init__(self, *, beam_size: int, prediction_complete: PredictionComplete,
                 max_beam_size: int, rest: str,
                 state_updater: 'StateUpdater',
                 probs: Optional[List[float]],
                 is_token_by_token: bool,
                 itos: List[str]):
        super().__init__()
        self.itos = itos
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

    def next_batch(self, prompt: torch.Tensor, state: Any):
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
            self.text.append(texts[b] + self.itos[token])

        new_prompt = torch.stack(new_prompt, dim=1)
        new_state = self.state_updater.make_batch(new_state)

        self.beam_heap = []

        return new_prompt, new_state

    def update(self, next_token, state, old_state):
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
                token_str = self.itos[token]
                if not self.is_substr(check_rest, token_str):
                    continue

                if self.prediction_complete(text, token_str):
                    if not self.add_prediction_before_token(self.probs[b], b, old_state):
                        break
                    else:
                        break
                    # if not self.add_prediction(self.probs[b] * tokens[token].item(), b, token_str, state):
                    #     break
                if not self.add_beam(self.probs[b] * tokens[token].item(), b, token):
                    break
