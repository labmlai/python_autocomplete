from typing import List, Tuple

from labml import monit
from python_autocomplete.dataset import ID_CHARS


class WordTokenizer:
    def collect_words(self, data: str):
        raise NotImplementedError

    def get_words(self) -> Tuple[List[str], List[int]]:
        raise NotImplementedError

    def tokenize(self, data: str, *, is_silent: bool = False) -> List[str]:
        raise NotImplementedError


class SourceCodeTokenizer(WordTokenizer):
    def __init__(self):
        self.words = {}

    def add_word(self, word):
        if not word:
            return

        if word not in self.words:
            self.words[word] = 1
        else:
            self.words[word] += 1

    def tokenize(self, data: str, *, is_silent: bool = False) -> List[str]:
        last_idx = 0
        is_id = False
        res = []

        for i, c in monit.enum('Collect words', data, is_silent=is_silent):
            if c in ID_CHARS:
                if not is_id:
                    if last_idx < i:
                        res.append(data[last_idx:i])
                    last_idx = i
                    is_id = True
            else:
                if is_id:
                    if last_idx < i:
                        res.append(data[last_idx:i])
                    last_idx = i
                    is_id = False

        if last_idx < len(data):
            res.append(data[last_idx:])

        return res

    def collect_words(self, data: str):
        last_idx = 0
        is_id = False

        for i, c in monit.enum('Collect words', data):
            if c in ID_CHARS:
                if not is_id:
                    self.add_word(data[last_idx:i])
                    last_idx = i
                    is_id = True
            else:
                if is_id:
                    self.add_word(data[last_idx:i])
                    last_idx = i
                    is_id = False

        self.add_word(data[last_idx:])

    def get_words(self):
        words_list = [(f, w) for w, f in self.words.items()]
        words_list.sort(key=lambda x: -x[0])

        return [w for _, w in words_list], [f for f, _ in words_list]


class NoTokenizer(WordTokenizer):
    def __init__(self):
        self.data = ''

    def collect_words(self, data):
        self.data += data

    def get_words(self):
        return [self.data], [1]

    def tokenize(self, data: str, *, is_silent: bool = False) -> List[str]:
        return [data]
