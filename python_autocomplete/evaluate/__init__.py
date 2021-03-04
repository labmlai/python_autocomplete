from typing import Any, Tuple, List, NamedTuple

import torch
import torch.nn
from torch import nn

from labml import monit
from labml_helpers.module import Module
from python_autocomplete.dataset import Tokenizer
from python_autocomplete.evaluate.beam_search import PredictionComplete, BeamSearch, BeamSearchSimple
from python_autocomplete.evaluate.beam_search_lengthy import BeamSearchLengthy
from python_autocomplete.train import StateUpdater


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
        beam = BeamSearchSimple(beam_size=prompt.shape[1],
                                prediction_complete=prediction_complete,
                                max_beam_size=max_beam_size,
                                rest=rest,
                                state_updater=self.state_updater,
                                probs=probs,
                                is_token_by_token=self.is_token_by_token,
                                itos=self.tokenizer.itos)

        for _ in range(10):
            with monit.section('Predict', is_silent=True):
                next_token, new_state = self._get_predictions(prompt, state)
            with monit.section('Beam', is_silent=True):
                beam.update(next_token, new_state, state)
                prompt, state = beam.next_batch(prompt, new_state)

            if prompt is None:
                break

        results = [Prediction(r[0], r[1][0], r[1][1]) for r in beam.result_heap]
        return results

    def rstrip(self, prompt: str) -> Tuple[str, List[int]]:
        return self.tokenizer.rstrip(prompt)
