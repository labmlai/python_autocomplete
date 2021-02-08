from typing import Any

import torch

from labml_helpers.module import Module


class AutoregressiveModel(Module):
    def __init__(self):
        super().__init__()

    def __call__(self, src: torch.Tensor, state: Any):
        pass
