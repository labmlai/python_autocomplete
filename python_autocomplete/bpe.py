import string
from heapq import heappush, heappop

from labml import lab, monit

ID_CHARS = set(string.ascii_letters + string.digits + '_')


class BPE:
    def __init__(self):
        self.char_itos = []
        self.char_stoi = {}
        self.bpe_itos = []
        self.bpe = []
        self.common = {}

        self.bpe_itos = self.calc_bpe_itos()

    def to_char_stoi(self, w: str):
        return [self.char_stoi[c] for c in w]

    def calc_bpe_itos(self):
        itos = list(self.char_itos)
        itos += [itos[p1] + itos[p2] for p1, p2 in self.bpe[len(self.char_itos):]]
        return itos


class BPELearner:
    def __init__(self, data: str):
        self.data = data
        self.words = {}
        self.heap = []
        self.heap_modified = set()
        self.char_itos = []
        self.char_stoi = {}
        self.bpe = []
        self.word_codes = []
        self.word_code_prev = {}
        self.word_code_next = {}

        self.counts = {}
        self.locations = {}

        self.collect_words()
        self.build_vocab()
        self.build_word_arrays()
        self.collect_pairs()

    def learn(self, merges: int):
        for i in monit.iterate('BPE', merges):
            while True:
                res = self.merge_pair()
                if res is not None:
                    break

    def add_word(self, word):
        if not word:
            return

        if word not in self.words:
            self.words[word] = 1
        else:
            self.words[word] += 1

    def collect_words(self):
        last_idx = 0
        is_id = False

        for i, c in monit.enum('Collect words', self.data):
            if c in ID_CHARS:
                if not is_id:
                    self.add_word(self.data[last_idx:i])
                    last_idx = i
                    is_id = True
            else:
                if is_id:
                    self.add_word(self.data[last_idx:i])
                    last_idx = i
                    is_id = False

        self.add_word(self.data[last_idx:])
        words_list = [(f, w) for w, f in self.words.items()]
        words_list.sort(key=lambda x: -x[0])

        self.words_list = [w for _, w in words_list]
        self.word_freq = [f for f, _ in words_list]

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
        data = f.read()[:100_000]

    bpe = BPELearner(data)
    bpe.learn(1000)
    print(len(bpe.bpe))
    print(bpe.bpe_itos()[len(bpe.char_itos):])
    print(len(bpe.data), bpe.get_length())


if __name__ == '__main__':
    main()
