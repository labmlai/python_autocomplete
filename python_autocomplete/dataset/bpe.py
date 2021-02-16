from functools import lru_cache
from heapq import heappush, heappop
from typing import List

from labml import lab, monit
from labml.utils.cache import cache_set
from python_autocomplete.dataset import Tokenizer
from python_autocomplete.dataset.break_words import SourceCodeTokenizer


class BPE(Tokenizer):
    def __init__(self, bpe_en_de: 'BPEEnDe', word_tokenizer):
        self.bpe = bpe_en_de
        self.word_tokenizer = word_tokenizer
        self.is_trained = True

    @property
    def n_tokens(self):
        return len(self.bpe.bpe)

    @property
    def itos(self):
        return self.bpe.bpe_itos

    @property
    def stoi(self):
        return self.bpe.bpe_stoi

    def encode(self, data: str, *, is_silent: bool = True):
        words = self.word_tokenizer.tokenize(data, is_silent=is_silent)

        res = []
        for w in monit.iterate('Encode words', words, is_silent=is_silent):
            res += self.bpe.encode(w)

        return res

    def __call__(self, data: str):
        encoded = self.encode(data)
        return [self.itos[c] for c in encoded]

    def rstrip(self, data: str):
        words = self.word_tokenizer.tokenize(data, is_silent=True)
        words = words[:-1]
        res = []
        for w in words:
            res += self.bpe.encode(w)

        return ''.join(words), res


class _BPEEncoder:
    def __init__(self, pairs):
        self.pairs = pairs
        self.codes = []
        self.next_idx = []
        self.prev_idx = []
        self.heap = []

    def encode(self, codes: List[int]):
        self.codes = codes
        self.next_idx = BPELearner.default_next_pointers(len(codes))
        self.prev_idx = BPELearner.default_prev_pointers(len(codes))
        self.heap = []

        for i in range(len(self.codes) - 1):
            self.add_pair((self.codes[i], self.codes[i + 1]), i)

        while self.heap:
            _, idx, pair = heappop(self.heap)
            self.merge(idx, pair)

        return [c for c in self.codes if c != -1]

    def merge(self, p2, pair):
        p3 = self.next_idx[p2]

        if p3 == -1 or pair[0] != self.codes[p2] or pair[1] != self.codes[p3]:
            return

        self.codes[p2] = self.pairs[pair]
        self.codes[p3] = -1
        p1 = self.prev_idx[p2]
        p4 = self.next_idx[p3]

        if p1 != -1:
            self.add_pair((self.codes[p1], self.codes[p2]), p1)
        self.next_idx[p2] = p4
        if p4 != -1:
            self.prev_idx[p4] = p2
            self.add_pair((self.codes[p2], self.codes[p4]), p2)

    def add_pair(self, pair, idx):
        if pair not in self.pairs:
            return

        heappush(self.heap, (self.pairs[pair], idx, pair))


class BPEEnDe:
    def __init__(self):
        self.char_itos = []
        self.char_stoi = {}
        self.bpe = []
        self.popular_words = {}

        self.bpe_itos = []
        self.bpe_stoi = {}
        self.pairs = {}
        self.encoder = None

    def load(self, char_itos, char_stoi, bpe):
        self.char_itos = char_itos
        self.char_stoi = char_stoi
        self.bpe = bpe

        self.calc()

    def set_popular_words(self, popular_words):
        self.popular_words = popular_words

    def calc(self):
        self.bpe_itos = self.calc_bpe_itos()
        self.bpe_stoi = {s: i for i, s in enumerate(self.bpe_itos)}
        self.pairs = {(p[0], p[1]): c for c, p in enumerate(self.bpe) if not isinstance(p, int)}

        self.encoder = _BPEEncoder(self.pairs)

    def to_char_stoi(self, w: str):
        return [self.char_stoi[c] for c in w]

    def calc_bpe_itos(self):
        itos = list(self.char_itos)
        for p1, p2 in self.bpe[len(self.char_itos):]:
            itos.append(itos[p1] + itos[p2])
        return itos

    @lru_cache(1024)
    def encode(self, word: str):
        if word in self.popular_words:
            return self.popular_words[word]

        return self.encoder.encode([self.char_stoi[c] for c in word if c in self.char_stoi])


class BPELearner:
    def __init__(self, words_list: List[str], word_freq: List[int]):
        self.words_list = words_list
        self.word_freq = word_freq

        self.heap = []
        self.heap_modified = set()
        self.char_itos = []
        self.char_stoi = {}
        self.bpe = []
        self.word_codes = []
        self.word_code_prev = []
        self.word_code_next = []

        self.counts = {}
        self.locations = {}

        self.build_vocab()
        self.build_word_arrays()
        self.collect_pairs()

    def learn(self, merges: int):
        for i in monit.iterate('BPE', merges):
            while True:
                res = self.merge_pair()
                if res is not None:
                    break

    def build_vocab(self):
        vocab = set()
        for k in self.words_list:
            for c in k:
                vocab.add(c)

        self.char_itos = list(sorted(vocab))
        self.char_stoi = {c: i for i, c in enumerate(self.char_itos)}

        self.bpe = [i for i in range(len(self.char_stoi))]

    def to_char_stoi(self, w: str):
        return [self.char_stoi[c] for c in w]

    @staticmethod
    def default_next_pointers(length: int):
        return [i + 1 for i in range(length - 1)] + [-1]

    @staticmethod
    def default_prev_pointers(length: int):
        return [i - 1 for i in range(length)]

    def build_word_arrays(self):
        self.word_codes = [self.to_char_stoi(w) for w in self.words_list]
        self.word_code_next = [self.default_next_pointers(len(w)) for w in self.word_codes]
        self.word_code_prev = [self.default_prev_pointers(len(w)) for w in self.word_codes]

    def heap_add_all(self):
        for pair in self.heap_modified:
            if pair in self.counts:
                heappush(self.heap, (-self.counts[pair], pair))

    def add_pair(self, w, i, nxt):
        pair = self.word_codes[w][i], self.word_codes[w][nxt]
        assert pair[0] != -1 and pair[1] != -1

        if pair not in self.counts:
            self.counts[pair] = 0
            self.locations[pair] = {}

        if w not in self.locations[pair]:
            self.locations[pair][w] = set()

        self.counts[pair] += self.word_freq[w]
        self.locations[pair][w].add(i)

        self.heap_modified.add(pair)

    def collect_pairs(self):
        for w, v in monit.enum('Collect pairs', self.word_codes):
            f = self.word_freq[w]

            for i in range(len(v) - 1):
                self.add_pair(w, i, i + 1)

        self.heap_add_all()

    def remove_pair(self, w, i, nxt):
        pair = self.word_codes[w][i], self.word_codes[w][nxt]
        assert pair[0] != -1 and pair[1] != -1
        if pair not in self.counts:
            return
        self.locations[pair][w].remove(i)
        self.counts[pair] -= self.word_freq[w]
        self.heap_modified.add(pair)

    def merge_pair(self):
        cnt, pair = heappop(self.heap)
        if pair not in self.counts or self.counts[pair] != -cnt:
            return None

        n = len(self.bpe)
        self.bpe.append(pair)
        del self.counts[pair]
        for w, locs in self.locations[pair].items():
            locs = list(reversed(sorted(locs)))
            prev = None
            merged = []
            for p2 in locs:
                p1 = self.word_code_prev[w][p2]
                p3 = self.word_code_next[w][p2]
                assert p3 != -1
                if p3 == prev:
                    continue
                p4 = self.word_code_next[w][p3]

                if p1 != -1:
                    self.remove_pair(w, p1, p2)
                if p4 != -1 and p4 != prev:
                    self.remove_pair(w, p3, p4)

                prev = p2
                merged.append(p2)

            for p2 in merged:
                p3 = self.word_code_next[w][p2]
                p4 = self.word_code_next[w][p3]
                self.word_codes[w][p2] = n
                self.word_codes[w][p3] = -1
                self.word_code_next[w][p3] = -1
                self.word_code_prev[w][p3] = -1
                if p4 != -1:
                    self.word_code_next[w][p2] = p4
                    self.word_code_prev[w][p4] = p2
                else:
                    self.word_code_next[w][p2] = -1

            for p2 in merged:
                p1 = self.word_code_prev[w][p2]
                p3 = self.word_code_next[w][p2]

                if p1 != -1:
                    self.add_pair(w, p1, p2)
                if p3 != -1:
                    self.add_pair(w, p2, p3)

        self.heap_add_all()

        return pair

    def bpe_itos(self):
        itos = list(self.char_itos)
        for p1, p2 in self.bpe[len(self.char_itos):]:
            itos.append(itos[p1] + itos[p2])

        return itos

    def get_length(self):
        res = 0
        for w, v in enumerate(self.word_codes):
            cnt = 0
            for idx in v:
                if idx != -1:
                    cnt += 1
            res += cnt * self.word_freq[w]

        return res


def main():
    path = lab.get_data_path() / 'train.py'

    with open(str(path), 'r') as f:
        data = f.read()

    tokenizer = SourceCodeTokenizer()
    tokenizer.collect_words(data)

    bpe = BPELearner(*tokenizer.get_words())
    bpe.learn(1000)
    print(len(bpe.bpe))
    print(bpe.bpe_itos()[len(bpe.char_itos):])
    print(len(data), bpe.get_length())

    cache_set('bpe', {
        'char_itos': bpe.char_itos,
        'char_stoi': bpe.char_stoi,
        'bpe': bpe.bpe
    })

    bpe_en_de = BPEEnDe()
    bpe_en_de.load(bpe.char_itos, bpe.char_stoi, bpe.bpe)


if __name__ == '__main__':
    main()
