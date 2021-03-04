from heapq import heappush, heappop
from typing import Any, List, Optional

import torch
import torch.nn

from python_autocomplete.evaluate.beam_search import PredictionComplete, BeamSearch
from python_autocomplete.train import StateUpdater

EPS_SCORE = 1e-6
MIN_BEAM_PROB = 1e-4


class BeamSearchLengthy(BeamSearch):
    """
    Use a score instead of probability to make longer predictions
    """
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
    def is_prefix(original, token_str):
        """
        Whether `token_str` is a prefix of `original`
        """
        if not original:
            return True

        n = min(len(original), len(token_str))
        return original[:n] == token_str[:n]

    @staticmethod
    def _get_score(text: str, prob: float):
        return len(text) ** 0.2 * prob

    def _add_prediction_before_token(self, prob: float, beam_idx: int, state):
        """
        Add a prediction before the last token to the results
        """
        text = self.text[beam_idx]
        score = self._get_score(text, prob)

        if len(self.result_heap) == self.max_beam_size:
            if self.result_heap[0][0] > score - EPS_SCORE:
                return
            heappop(self.result_heap)

        state = self.state_updater.get_from_batch(state, beam_idx)
        heappush(self.result_heap, (score, (text, state)))

    def _add_beam(self, prob: float, beam_idx: int, token: int):
        """Add to the beam"""
        text = self.text[beam_idx] + self.itos[token]
        score = self._get_score(text, prob)

        # if self.result_heap and self.result_heap[0][0] > score - EPS_SCORE:
        #     return False

        if prob < MIN_BEAM_PROB:
            return False

        if len(self.beam_heap) == self.max_beam_size:
            if self.beam_heap[0][0] > score - EPS_SCORE:
                return False
            heappop(self.beam_heap)

        heappush(self.beam_heap, (score, (beam_idx, prob, token)))

        return True

    def next_batch(self, prompt: torch.Tensor, state: Any):
        """Get the next batch (beam)"""
        if not self.beam_heap:
            return None, None

        new_prompt = []
        new_state = []

        texts = self.text
        self.text = []
        self.probs = []

        for _, (b, prob, token) in self.beam_heap:
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
        """Update beam search with the sampled data"""
        self.beam_heap = []

        for b, text in enumerate(self.text):
            text = self.text[b]
            if len(text) >= len(self.rest):
                check_rest = None
            else:
                check_rest = self.rest[len(text):]

            tokens = next_token[b]
            sort_idx = torch.argsort(tokens)
            added_to_results = False

            for i in reversed(range(len(tokens))):
                token = sort_idx[i]
                token_str = self.itos[token]
                if not self.is_prefix(check_rest, token_str):
                    continue

                if not added_to_results and self.prediction_complete(text, token_str):
                    added_to_results = True
                    self._add_prediction_before_token(self.probs[b], b, old_state)

                if check_rest and len(token_str) <= len(check_rest):
                    p = 1.0
                else:
                    p = tokens[token].item()

                if not self._add_beam(self.probs[b] * p, b, token):
                    break
