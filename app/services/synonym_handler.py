import io
import random
import string
from typing import List

import numpy as np
from tqdm import tqdm
from fairseq.models.roberta import RobertaModel
from fairseq.data.encoders.fastbpe import fastBPE

from app.core.config import FASTTEXT_PATH, PHOBERT_PATH, STOPWORD_PATH
from app.services.base_augmenter import Augmenter


class BPE:
    bpe_codes = PHOBERT_PATH + "/bpe.codes"


class SynonymHandler(Augmenter):
    fasttext_data = None
    phobert = None

    @classmethod
    def get_model(cls):
        if cls.fasttext_data is None and cls.phobert is None:
            cls.stop_words = cls.load_stop_words()
            cls.fasttext_data = cls.load_vectors(FASTTEXT_PATH)

            args = BPE()
            cls.phobert = RobertaModel.from_pretrained(PHOBERT_PATH, checkpoint_file='model.pt')
            cls.phobert.bpe = fastBPE(args)

    @staticmethod
    def load_stop_words():
        with open(STOPWORD_PATH, "r", encoding="utf8") as f:
            stop_words = f.readlines()
            stop_words = [word.strip() for word in stop_words if word.split() == 1]
            return stop_words

    @staticmethod
    def load_vectors(filename, num_words=10000):
        fin = io.open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
        fin.readline().split()
        data = {}
        for i, line in enumerate(tqdm(fin)):
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = np.array([float(val) for val in tokens[1:]])
            data[tokens[0]] /= np.linalg.norm(data[tokens[0]])
            if i > num_words:
                break
        return data

    def _is_eligible_token(self, token):
        return token not in string.punctuation and token not in self.stop_words and not token.startswith("MASK")

    def _find_similar_word(self, word, num_similar=1):
        ls_similar_word = []

        if word not in self.fasttext_data:
            return []

        ref_v = self.fasttext_data[word]

        top_num = 20
        top_w = [''] * top_num
        top_s = [-1] * top_num

        for k in self.fasttext_data.keys():
            if word == k:
                continue
            score = np.dot(ref_v, self.fasttext_data[k])
            if score < np.min(top_s):
                continue

            for i in range(top_num):
                if score >= top_s[i]:
                    top_w[i + 1:] = top_w[i:top_num - 1]
                    top_w[i] = k
                    top_s[i + 1:] = top_s[i:top_num - 1]
                    top_s[i] = score
                    break

        count = 0
        for i in range(top_num):
            if count >= num_similar:
                return ls_similar_word

            if top_w[i].lower() == word:
                continue

            ls_similar_word.append(top_w[i].lower())
            count += 1

        return ls_similar_word

    def _get_synonyms(self, idx, tokens, num_similar=5, num_keep=1):
        chosen_synonyms = []

        synonyms = self._find_similar_word(tokens[idx], num_similar=num_similar)

        tokens[idx] = '<mask>'
        if idx > 256:
            start, end = idx - 100, idx + 100
        else:
            start, end = 0, 256
        phobert_filled = [_[2].lower() for _ in self.phobert.fill_mask(' '.join(tokens[start:end]), topk=50)]

        for synonym in synonyms:
            # if len(chosen_synonyms) < num_keep:
            if synonym in phobert_filled and len(chosen_synonyms) < num_keep:
                chosen_synonyms.append(synonym)

        return chosen_synonyms

    def validate_action(self, action):
        assert action in ["substitute", "insert"], "Please choose action in {substitute, insert}"

    def transform(self, action, tokens, eligible_indices, num_similar=5, num_keep=1, **kwargs) -> List[str]:
        augmented_data = []
        tmp = tokens.copy()

        for idx in eligible_indices:
            synonyms = self._get_synonyms(idx, tokens.copy(), num_similar=num_similar, num_keep=num_keep)

            for synonym in synonyms:
                if action == "substitute":
                    tmp[idx] = synonym
                else:
                    random_idx = random.choice(range(len(tmp)))
                    tmp.insert(random_idx, synonym)

        augmented_data.append(" ".join(tmp))

        return self.remove_duplicate(augmented_data)
