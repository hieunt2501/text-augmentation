from abc import ABC

import random as normal_random
import nlpaug.augmenter.word as naw
from nlpaug.util import Action, Doc

import app.services.spelling.utils as utils


class DuplicateWordHandler(naw.SpellingAug, ABC):
    def __init__(self, dict_path=None, name='DuplicateWordAugmenter', aug_min=1, aug_max=10, aug_p=0.3, stopwords=None,
                 tokenizer=None, reverse_tokenizer=None, include_reverse=True, stopwords_regex=None,
                 verbose=0):
        super().__init__(dict_path=dict_path, name=name, aug_min=aug_min, aug_max=aug_max, aug_p=aug_p,
                         stopwords=stopwords,
                         tokenizer=tokenizer, reverse_tokenizer=reverse_tokenizer, include_reverse=include_reverse,
                         stopwords_regex=stopwords_regex,
                         verbose=verbose)

    def substitute(self, data):
        if not utils.is_valid_text(data):
            return data

        tokens = self.tokenizer(data)
        gaps = utils.find_all_gaps(data, tokens)

        aug_idxes = self._get_random_aug_idxes(tokens)
        if aug_idxes is None or len(aug_idxes) == 0:
            return data
        aug_idxes.sort(reverse=True)
        for idx in aug_idxes:
            tokens.insert(idx, tokens[idx])
            gaps.insert(idx + 1, ' ')
        return utils.reverse_tokenizer(tokens, gaps)


class InsertIrrelevantWordHandler(naw.SpellingAug, ABC):
    def __init__(self, dict_path=None, name='InsertIrrelevantWordAugmenter', aug_min=1, aug_max=2, aug_p=0.3,
                 stopwords=None,
                 tokenizer=None, reverse_tokenizer=None, include_reverse=True, stopwords_regex=None,
                 verbose=0, file_path='edit3.txt'):
        super().__init__(dict_path=dict_path, name=name, aug_min=aug_min, aug_max=aug_max, aug_p=aug_p,
                         stopwords=stopwords,
                         tokenizer=tokenizer, reverse_tokenizer=reverse_tokenizer, include_reverse=include_reverse,
                         stopwords_regex=stopwords_regex,
                         verbose=verbose)

        self.vocab = None
        self.create_vocab(file_path)

    def create_vocab(self, file_path):
        with open(file_path, encoding='utf-8') as f:
            contents = f.read().replace('\n', ' ')
            self.vocab = list(set(contents.split(' ')))

    def substitute(self, data):
        if not utils.is_valid_text(data):
            return data

        tokens = self.tokenizer(data)
        gaps = utils.find_all_gaps(data, tokens)

        aug_idxes = self._get_random_aug_idxes(tokens)
        if aug_idxes is None or len(aug_idxes) == 0:
            return data
        aug_idxes.sort(reverse=True)
        for idx in aug_idxes:
            sample = normal_random.choices(self.vocab, k=1)[0]
            while sample == tokens[idx]:
                sample = normal_random.choices(self.vocab, k=1)[0]
            tokens.insert(idx, sample)
            gaps.insert(idx + 1, ' ')
        return utils.reverse_tokenizer(tokens, gaps)


class EditDistanceHandler(naw.SpellingAug, ABC):
    def __init__(self, dict_path=None, name='MyEditDistanceAugmenter', aug_min=1, aug_max=10, aug_p=0.3, stopwords=None,
                 tokenizer=None, reverse_tokenizer=None, include_reverse=True, stopwords_regex=None,
                 verbose=0):
        super().__init__(dict_path=dict_path, name=name, aug_min=aug_min, aug_max=aug_max, aug_p=aug_p,
                         stopwords=stopwords,
                         tokenizer=tokenizer, reverse_tokenizer=reverse_tokenizer, include_reverse=include_reverse,
                         stopwords_regex=stopwords_regex,
                         verbose=verbose)
        if dict_path:
            self.model.dict = self.read(dict_path)

    @staticmethod
    def read(model_path):
        with open(model_path, 'r', encoding="utf-8") as f:
            ans = {}

            for line in f.readlines():
                tokens = line.split(' ')
                # Last token include newline separator
                tokens[-1] = tokens[-1].replace('\n', '')

                key = tokens[0]
                values = tokens[1:]
                if key not in ans:
                    ans[key] = []

                ans[key].extend(values)
                # Remove duplicate mapping
                ans[key] = list(set(ans[key]))
            return ans

    def substitute(self, data):
        if not data or not data.strip():
            return data

        change_seq = 0
        doc = Doc(data, self.tokenizer(data))
        gaps = utils.find_all_gaps(data, doc.get_original_tokens())

        aug_idxes = self._get_aug_idxes(doc.get_original_tokens())

        if aug_idxes is None or len(aug_idxes) == 0:
            if self.include_detail:
                return data, []
            return data

        for aug_idx, original_token in enumerate(doc.get_original_tokens()):
            # Skip if no augment for word
            if aug_idx not in aug_idxes:
                continue

            candidate_words = self.model.predict(original_token)
            if candidate_words:
                substitute_token = self.sample(candidate_words, 1)[0]
            else:
                # Unexpected scenario. Adding original token
                substitute_token = original_token

            if aug_idx == 0:
                substitute_token = self.align_capitalization(original_token, substitute_token)

            change_seq += 1
            doc.add_change_log(aug_idx, new_token=substitute_token, action=Action.SUBSTITUTE,
                               change_seq=self.parent_change_seq + change_seq)

        if self.include_detail:
            return utils.reverse_tokenizer(doc.get_augmented_tokens(), gaps), doc.get_change_logs()
        else:
            return utils.reverse_tokenizer(doc.get_augmented_tokens(), gaps)
