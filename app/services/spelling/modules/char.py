import re
import string
from abc import ABC

import numpy as np
import numpy.random as random
import nlpaug.augmenter.char as nac
from nlpaug.util import Action, Method, Doc

import app.services.spelling.utils as utils
from app.services.spelling.modules.base_module import COMPOSITION_CHARS
from app.services.spelling.modules.typo import TelexHandler


class RandomCharHandler(nac.CharAugmenter, ABC):
    def __init__(self, name='MyRandomCharAugmenter', min_char=2, aug_char_min=1, aug_char_max=10, aug_char_p=0.3,
                 aug_word_min=1, aug_word_max=10, aug_word_p=0.3, tokenizer=None, reverse_tokenizer=None,
                 stopwords=None, verbose=0, stopwords_regex=None):
        super().__init__(
            name=name, action="substitute", min_char=min_char, aug_char_min=aug_char_min,
            aug_char_max=aug_char_max, aug_char_p=aug_char_p, aug_word_min=aug_word_min,
            aug_word_max=aug_word_max, aug_word_p=aug_word_p, tokenizer=tokenizer,
            reverse_tokenizer=reverse_tokenizer, stopwords=stopwords, device='cpu',
            verbose=verbose, stopwords_regex=stopwords_regex)
        self.telexDecomposer = TelexHandler()
        self.randomizers = [
            DuplicateHandler(aug_word_p=1, min_char=min_char, aug_char_min=aug_char_min, aug_char_max=aug_char_max,
                             aug_char_p=aug_char_p),
            nac.RandomCharAug(action="delete", aug_word_p=1, min_char=min_char, aug_char_min=aug_char_min,
                              aug_char_max=aug_char_max, aug_char_p=aug_char_p),
            nac.RandomCharAug(action="swap", aug_word_p=1, min_char=min_char, aug_char_min=aug_char_min,
                              aug_char_max=aug_char_max, aug_char_p=aug_char_p),
            nac.RandomCharAug(action="insert", aug_word_p=1, min_char=min_char,
                              aug_char_min=aug_char_min, aug_char_max=aug_char_max,
                              aug_char_p=aug_char_p, include_upper_case=False, spec_char="",
                              include_numeric=False)
            #   nac.RandomCharAug(action="substitute", aug_word_p=1, min_char=3),
        ]
        self.pdf = [0.4, 0.3, 0.2, 0.1]

    def substitute(self, data):
        results = []
        # Tokenize a text (e.g. The quick brown fox jumps over the lazy dog) to tokens (e.g. ['The', 'quick', ...])
        tokens = self.tokenizer(data)
        gaps = utils.find_all_gaps(data, tokens)

        # Get target tokens
        aug_word_idxes = self._get_aug_idxes(tokens, self.aug_word_min, self.aug_word_max, self.aug_word_p, Method.WORD)
        for token_i, token in enumerate(tokens):
            # Do not augment if it is not the target
            if token_i not in aug_word_idxes:
                results.append(token)
                continue
            token = self.telexDecomposer.generate_word_error(token)
            action = np.random.choice(self.randomizers, p=self.pdf)
            result = action.augment(token)
            results.append(result)

        return utils.reverse_tokenizer(results, gaps)


class SubstituteHandler(nac.CharAugmenter, ABC):
    def __init__(self, name='SubstituteAugmenter', min_char=2, aug_char_p=0.3,
                 aug_word_min=1, aug_word_max=100, aug_word_p=0.3, tokenizer=None, reverse_tokenizer=None,
                 stopwords=None, verbose=0, stopwords_regex=None):
        super().__init__(
            name=name, action="substitute", min_char=min_char, aug_char_min=1,
            aug_char_max=10, aug_char_p=aug_char_p, aug_word_min=aug_word_min,
            aug_word_max=aug_word_max, aug_word_p=aug_word_p, tokenizer=tokenizer,
            reverse_tokenizer=reverse_tokenizer, stopwords=stopwords, device='cpu',
            verbose=verbose, stopwords_regex=stopwords_regex)
        self.eligibleCharacters = COMPOSITION_CHARS
        self.vowels = ['a', 'â', 'ă', 'e', 'ê', 'o', 'ô', 'ơ', 'u', 'ư', 'y']
        self.consonants_1 = ['b', 'd', 'h', 'l', 'm', 'n', 'p', 'r', 's', 't', 'v', 'x', 'đ', 'g', 'k', 'c']
        self.consonants_2 = ['tr', 'th', 'ch', 'ph', 'nh', 'kh', 'gi', 'qu', 'ng', 'gh']

    @staticmethod
    def check_if_vowel(char):
        return char.lower() in ['a', 'â', 'ă', 'e', 'ê', 'o', 'ô', 'ơ', 'u', 'ư', 'y']

    def substitute(self, data):
        results = []
        tokens = self.tokenizer(data)
        aug_word_idxes = self._get_aug_idxes(tokens, self.aug_word_min,
                                             self.aug_word_max, self.aug_word_p, Method.WORD)

        for token_i, token in enumerate(tokens):
            if token_i not in aug_word_idxes:
                results.append(token)
                continue

            result = ''
            chars = self.token2char(token)
            i = 0
            while i < len(chars):
                if random.random() < self.aug_char_p:
                    if self.check_if_vowel(chars[i]):
                        sub_char = self.sample(self.vowels, 1)[0]
                        if chars[i].isupper():
                            result += sub_char.upper()
                        else:
                            result += sub_char
                    else:
                        if ''.join(chars[i:i + 2]).lower() in self.consonants_2:
                            sub_char = self.sample(self.consonants_2, 1)[0]
                            if chars[i].isupper():
                                result += sub_char[0].upper()
                            else:
                                result += sub_char[0]
                            if chars[i + 1].isupper():
                                result += sub_char[1].upper()
                            else:
                                result += sub_char[1]
                            i += 1
                        elif chars[i].lower() in self.consonants_1:
                            sub_char = self.sample(self.consonants_1, 1)[0]
                            if chars[i].isupper():
                                result += sub_char.upper()
                            else:
                                result += sub_char
                        else:
                            result += chars[i]
                else:
                    if ''.join(chars[i:i + 2]).lower() in self.consonants_2:
                        result += ''.join(chars[i:i + 2])
                        i += 1
                    else:
                        result += chars[i]
                i += 1

            results.append(result)

        return self.reverse_tokenizer(results)


class MisspellVowelHandler(nac.CharAugmenter, ABC):
    def __init__(self, name='MisspellVowelAugment', min_char=2, aug_char_p=0.3,
                 aug_word_min=1, aug_word_max=100, aug_word_p=0.3, tokenizer=None, reverse_tokenizer=None,
                 stopwords=None, verbose=0, stopwords_regex=None):
        super().__init__(
            name=name, action="substitute", min_char=min_char, aug_char_min=1,
            aug_char_max=10, aug_char_p=aug_char_p, aug_word_min=aug_word_min,
            aug_word_max=aug_word_max, aug_word_p=aug_word_p, tokenizer=tokenizer,
            reverse_tokenizer=reverse_tokenizer, stopwords=stopwords, device='cpu',
            verbose=verbose, stopwords_regex=stopwords_regex)

        self.model = {
            "iếu": ["ếu"],
            "iều": ["ều"],
            "oanh": ["anh"],
            "ếu": ["iếu"],
            "ều": ["iều"],
            "anh": ["oanh"]
        }
        self.eligibleCharacters = self.model.keys()

    def is_eligible(self, word):
        return any(map(lambda c: c in word, self.eligibleCharacters))

    def _get_vowel(self, token):
        for vowel in self.eligibleCharacters:
            if vowel in token:
                return vowel
        return None

    def substitute(self, data):
        results = []
        tokens = self.tokenizer(data)
        gaps = utils.find_all_gaps(data, tokens)
        temp = [tok if self.is_eligible(tok) else '' for tok in tokens]
        aug_word_idxes = self._get_aug_idxes(temp, self.aug_word_min, self.aug_word_max, self.aug_word_p, Method.WORD)

        for token_i, token in enumerate(tokens):
            if token_i not in aug_word_idxes:
                results += [token]
                continue
            vowel = self._get_vowel(token)
            token = re.sub(vowel, self.sample(self.model[vowel], 1)[0], token)
            results += [token]

        return utils.reverse_tokenizer(results, gaps)


class DuplicateHandler(nac.CharAugmenter, ABC):
    def __init__(self, name='DuplicateAugmenter', min_char=1, aug_char_min=0, aug_char_max=100, aug_char_p=0.3,
                 aug_word_min=0, aug_word_max=100, aug_word_p=0.3, tokenizer=None, reverse_tokenizer=None,
                 stopwords=None, verbose=0, stopwords_regex=None):
        super().__init__(
            name=name, action="insert", min_char=min_char, aug_char_min=aug_char_min,
            aug_char_max=aug_char_max, aug_char_p=aug_char_p, aug_word_min=aug_word_min,
            aug_word_max=aug_word_max, aug_word_p=aug_word_p, tokenizer=tokenizer,
            reverse_tokenizer=reverse_tokenizer, stopwords=stopwords, device='cpu',
            verbose=verbose, stopwords_regex=stopwords_regex)

    def insert(self, data):
        if not data or not data.strip():
            return data

        change_seq = 0

        doc = Doc(data, self.tokenizer(data))
        gaps = utils.find_all_gaps(data, doc.get_original_tokens())

        aug_word_idxes = self._get_aug_idxes(
            doc.get_original_tokens(), self.aug_word_min, self.aug_word_max, self.aug_word_p, Method.WORD)

        if aug_word_idxes is None:
            return data

        for token_i, token in enumerate(doc.get_original_tokens()):
            if token_i not in aug_word_idxes:
                continue

            chars = self.token2char(token)
            aug_char_idxes = self._get_aug_idxes(chars, self.aug_char_min, self.aug_char_max, self.aug_char_p,
                                                 Method.CHAR)
            if aug_char_idxes is None:
                continue

            aug_char_idxes.sort(reverse=True)
            for char_i in aug_char_idxes:
                chars.insert(char_i, chars[char_i])

            # No capitalization alignment as this augmenter try to simulate random error

            new_token = ''.join(chars)
            change_seq += 1
            doc.add_change_log(token_i, new_token=new_token, action=Action.INSERT,
                               change_seq=self.parent_change_seq + change_seq)

        if self.include_detail:
            return utils.reverse_tokenizer(doc.get_augmented_tokens(), gaps), doc.get_change_logs()
        else:
            return utils.reverse_tokenizer(doc.get_augmented_tokens(), gaps)


class WhitespaceHandler(nac.CharAugmenter, ABC):
    def __init__(self, name='WhitespaceAugmenter', min_char=2, aug_char_p=0.3,
                 aug_word_min=1, aug_word_max=2, aug_word_p=0.3, tokenizer=None, reverse_tokenizer=None,
                 stopwords=None, verbose=0, stopwords_regex=None):
        super().__init__(
            name=name, action="substitute", min_char=min_char, aug_char_min=1,
            aug_char_max=10, aug_char_p=aug_char_p, aug_word_min=aug_word_min,
            aug_word_max=aug_word_max, aug_word_p=aug_word_p, tokenizer=tokenizer,
            reverse_tokenizer=reverse_tokenizer, stopwords=stopwords, device='cpu',
            verbose=verbose, stopwords_regex=stopwords_regex)

        self.eligibleCharacters = COMPOSITION_CHARS

    @staticmethod
    def _reverse_tokenizer(tokens, results, text):
        tmp = text
        final_string = ""
        j = 0
        for token in tokens:
            x = re.search(token, tmp)
            if token in final_string and token not in string.punctuation:
                tmp = tmp[x.span()[1] + 1:]
                continue
            if x and x.span()[0] == 0:
                final_string += results[j]
                j += 1
                tmp = tmp[x.span()[1]:]
                if tmp:
                    while tmp[0] == " ":
                        tmp = tmp[1:]
                        final_string += " "

        return final_string

    def substitute(self, data):
        results = []
        concat_word = []
        tokens = self.tokenizer(data)
        whitespaces = re.findall(" ", ' '.join(tokens))
        aug_word_idxes = self._get_aug_idxes(whitespaces, self.aug_word_min,
                                             self.aug_word_max, self.aug_word_p, Method.CHAR)
        for whitespace_i, _ in enumerate(whitespaces):
            if whitespace_i not in aug_word_idxes:
                if tokens[whitespace_i] not in concat_word:
                    if whitespace_i != len(whitespaces) - 1:
                        results += [tokens[whitespace_i]]
                    else:
                        results += [tokens[whitespace_i], tokens[whitespace_i + 1]]
                    continue
                else:
                    if whitespace_i == len(whitespaces) - 1:
                        results += [tokens[whitespace_i + 1]]
                    continue
            elif tokens[whitespace_i] in string.punctuation or tokens[whitespace_i + 1] in string.punctuation:
                if tokens[whitespace_i] in concat_word:
                    # results += [tokens[whitespace_i + 1]]
                    continue
                elif whitespace_i != len(whitespaces) - 1:
                    results += [tokens[whitespace_i]]
                else:
                    results += [tokens[whitespace_i], tokens[whitespace_i + 1]]
            elif random.random() < self.aug_char_p:
                if tokens[whitespace_i] not in concat_word:
                    results += [tokens[whitespace_i] + tokens[whitespace_i + 1]]
                    concat_word += [tokens[whitespace_i], tokens[whitespace_i + 1]]
                else:
                    results[-1] = results[-1] + tokens[whitespace_i + 1]
                    concat_word += [tokens[whitespace_i + 1]]
            else:
                if whitespace_i != len(whitespaces) - 1:
                    results += [tokens[whitespace_i]]
                else:
                    results += [tokens[whitespace_i], tokens[whitespace_i + 1]]

        return self.reverse_tokenizer(results)
