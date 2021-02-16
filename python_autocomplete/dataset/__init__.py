import string
from typing import Dict, List, Tuple

ID_CHARS = set(string.ascii_letters + string.digits + '_')


class Tokenizer:
    n_tokens: int
    itos: List[str]
    stoi: Dict[str, int]
    is_trained: int

    def encode(self, data: str, *, is_silent: bool = True):
        raise NotImplementedError

    def train(self, data: str):
        pass

    def rstrip(self, data: str) -> Tuple[str, List[int]]:
        return data, self.encode(data)
