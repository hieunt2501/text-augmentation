import string
import re
from abc import ABC

import numpy as np
import numpy.random as random
import nlpaug.augmenter.char as nac
from nlpaug.util import Method

import app.services.spelling.utils as utils

TOKENIZER_REGEX = re.compile(r'(\W)')
COMPOSITION_CHARS = ['á', 'à', 'ạ', 'ả', 'ã', 'â', 'ấ', 'ầ', 'ậ', 'ẩ', 'ẫ', 'ă', 'ắ', 'ằ', 'ặ', 'ẳ',
                     'ẵ', 'í', 'ì', 'ỉ', 'ĩ', 'ị', 'ú', 'ù', 'ủ', 'ũ', 'ụ', 'ư', 'ứ', 'ừ', 'ử', 'ữ',
                     'ự', 'é', 'è', 'ẻ', 'ẽ', 'ẹ', 'ê', 'ế', 'ề', 'ể', 'ễ', 'ệ', 'ó', 'ò', 'ỏ', 'õ',
                     'ọ', 'ô', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ', 'ơ', 'ớ', 'ờ', 'ở', 'ỡ', 'ợ', 'đ', 'ý', 'ỳ',
                     'ỷ', 'ỹ', 'ỵ']
punctuation = string.punctuation


def default_tokenizer(text):
    tokens = TOKENIZER_REGEX.split(text)
    return [t for t in tokens if len(t.strip()) > 0]
    # tokenized_lst = text.split(seperator)
    # tokenized_lst = list(filter(lambda token: token != '', tokenized_lst))
    # return tokenized_lst


class TypoAugmenter(nac.CharAugmenter, ABC):
    def __init__(self, name='TypoAugmenter', min_char=2, aug_char_min=1, aug_char_max=10, aug_char_p=0.3,
                 aug_word_min=1, aug_word_max=10, aug_word_p=0.3, tokenizer=None, reverse_tokenizer=None,
                 stopwords=None, verbose=0, stopwords_regex=None):
        super().__init__(
            name=name, action="substitute", min_char=min_char, aug_char_min=aug_char_min,
            aug_char_max=aug_char_max, aug_char_p=aug_char_p, aug_word_min=aug_word_min,
            aug_word_max=aug_word_max, aug_word_p=aug_word_p, tokenizer=tokenizer,
            reverse_tokenizer=reverse_tokenizer, stopwords=stopwords, device='cpu',
            verbose=verbose, stopwords_regex=stopwords_regex)
        self.typo = {}
        self.eligibleCharacters = COMPOSITION_CHARS

    def _is_typo(self, character):
        return character in self.typo

    def _word_is_decomposable(self, word):
        return any(map(self._is_typo, word.lower()))

    @staticmethod
    def _random_decompose_twice():
        return np.random.uniform() < 0.50

    @staticmethod
    def _generate_cdf(length):
        if length == 1:
            return [1]
        if length == 2:
            return [0.1, 0.9]

        mid = length - 2
        end_prob = 0.4
        other_prob = (1 - end_prob * 2) / mid if mid else 0
        return [other_prob, end_prob] + [other_prob] * (mid - 1) + [end_prob]

    def _get_new_decomposition(self, base_word, comp_word):
        raise NotImplementedError

    def _get_typo_dictionary(self):
        raise NotImplementedError

    @staticmethod
    def _contains_uo(word):
        lst_uo = ['ươ', 'ướ', 'ườ', 'ưở', 'ượ', "ưỡ"]
        return any(map(lambda x: x in word, lst_uo))

    @staticmethod
    def _insert_base_word(list_of_chars, index, base_word):
        if len(base_word) == 1:
            list_of_chars[index] = base_word
        else:
            list_of_chars[index] = list_of_chars[0]
            list_of_chars[index + 1] = list_of_chars[1]

    def substitute(self, data):
        results = []
        tokens = self.tokenizer(data)
        gaps = utils.find_all_gaps(data, tokens)
        temp = [tok if self._word_is_decomposable(tok) else '' for tok in tokens]
        aug_word_idxes = self._get_aug_idxes(temp,
                                             self.aug_word_min, self.aug_word_max,
                                             self.aug_word_p, Method.WORD)
        for token_i, token in enumerate(tokens):
            if token_i not in aug_word_idxes:
                results.append(token)
                continue
            result = self.generate_word_error(token)
            results.append(result)
        return utils.reverse_tokenizer(results, gaps)

    @staticmethod
    def recasing(telex_char, base_word, comp_word):
        if telex_char.isupper():
            base_word = base_word.upper()
            comp_word = [c.upper() for c in comp_word]
        elif not telex_char.islower():
            first, second = base_word
            if telex_char[0].isupper():
                first = first.upper()
            if telex_char[1].isupper():
                second = second.upper()

            base_word = first + second

        return base_word, comp_word

    def generate_word_error(self, word):
        # if not word[0].isalpha(): return word
        list_chars = [w for w in word]
        if self._contains_uo(word):
            index = list_chars.index('ư') if "ư" in list_chars \
                else list_chars.index('Ư')

            telex_char = ''.join(list_chars[index:index + 2])
            if telex_char.lower() not in ['ươ', 'ướ', 'ườ', 'ưở', 'ượ', "ưỡ"]:
                return word
            base_word, comp_word = self.typo[telex_char.lower()][0], self.typo[telex_char.lower()][1:]
        else:
            index, telex_char = next(((i, c) for i, c in enumerate(list_chars) if self._is_typo(c.lower())),
                                     (None, None))
            if not telex_char:
                return ''.join(list_chars)
            base_word, comp_word = self.typo[telex_char.lower()][0], self.typo[telex_char.lower()][1:]

        is_decomposed_twice = len(comp_word) != 1 and self._random_decompose_twice()
        if not is_decomposed_twice:
            base_word, comp_word = self._get_new_decomposition(base_word, comp_word)
        base_word, comp_word = self.recasing(telex_char, base_word, comp_word)
        self._insert_base_word(list_chars, index, base_word)
        if comp_word:
            self._insert_random(list_chars, index, base_word, comp_word[0])
            if is_decomposed_twice:
                self._insert_random(list_chars, index, base_word, comp_word[1])
        result = ''.join(list_chars)

        return result

    def _insert_random(self, list_chars, index_base, base_character, telex_character):
        possible_indices = list(range(index_base + int(len(base_character) == 2), len(list_chars) + 2))
        cdf = self._generate_cdf(len(possible_indices))
        index_to_insert = np.random.choice(possible_indices, p=cdf)
        if index_to_insert == len(list_chars):
            list_chars.append(telex_character)
        else:
            list_chars[index_to_insert:index_to_insert] = telex_character


class AccentAugmenter(nac.CharAugmenter, ABC):
    def __init__(self, name='AccentAugmenter', min_char=2, aug_char_p=0.3,
                 aug_word_min=1, aug_word_max=100, aug_word_p=0.3, tokenizer=None, reverse_tokenizer=None,
                 stopwords=None, verbose=0, stopwords_regex=None):
        super().__init__(
            name=name, action="substitute", min_char=min_char, aug_char_min=1,
            aug_char_max=10, aug_char_p=aug_char_p, aug_word_min=aug_word_min,
            aug_word_max=aug_word_max, aug_word_p=aug_word_p, tokenizer=tokenizer,
            reverse_tokenizer=reverse_tokenizer, stopwords=stopwords, device='cpu',
            verbose=verbose, stopwords_regex=stopwords_regex)
        self.eligibleCharacters = COMPOSITION_CHARS
        self.model = {}

    def _word_is_eligible(self, word):
        return any(map(lambda x: x.lower() in self.eligibleCharacters, word))

    def substitute(self, data):
        results = []
        tokens = self.tokenizer(data)
        gaps = utils.find_all_gaps(data, tokens)
        temp = [tok if self._word_is_eligible(tok) else '' for tok in tokens]
        aug_word_idxes = self._get_aug_idxes(temp, self.aug_word_min, self.aug_word_max, self.aug_word_p, Method.WORD)
        for token_i, token in enumerate(tokens):
            if token_i not in aug_word_idxes:
                results.append(token)
                continue

            result = ''
            chars = self.token2char(token)
            for char in chars:
                lchar = char.lower()
                if lchar not in self.eligibleCharacters:
                    result += char
                    continue
                if random.random() < self.aug_char_p:
                    eligible_replacement = self.model[lchar]
                    if char.isupper():
                        eligible_replacement = [c.upper() for c in eligible_replacement]
                    result += self.sample(eligible_replacement, 1)[0]
                else:
                    result += char

            results.append(result)

        return utils.reverse_tokenizer(results, gaps)


finalConsonant = ['i', 'y', 'c', 't', 'n', 'ng', 'nh']
beginConsonant = ['x', 's', 'd', 'đ', 'c', 'k', 'ngh', 'ng', 'gh', 'g', 'gi', 'd', 'r', 'tr', 'ch', 'n', 'l', 'kh',
                  'qu', 'u', 'v', 'nh']


class SpellingReplacementAugmenter(nac.CharAugmenter, ABC):
    def __init__(self, name='SpellingReplacementAugmenter', min_char=2, aug_char_p=0.3,
                 aug_word_min=1, aug_word_max=100, aug_word_p=0.3, tokenizer=None, reverse_tokenizer=None,
                 stopwords=None, verbose=0, stopwords_regex=None):
        super().__init__(
            name=name, action="substitute", min_char=min_char, aug_char_min=1,
            aug_char_max=10, aug_char_p=aug_char_p, aug_word_min=aug_word_min,
            aug_word_max=aug_word_max, aug_word_p=aug_word_p, tokenizer=tokenizer,
            reverse_tokenizer=reverse_tokenizer, stopwords=stopwords, device='cpu',
            verbose=verbose, stopwords_regex=stopwords_regex)
        self.begin_consonant = {
            "x": ["s"],
            "s": ["x"],
            "d": ['đ', "gi", "r", "v"],
            "đ": ['d'],
            "c": ["k"],
            "k": ["c", "kh"],
            "ngh": ["ng"],
            "ng": ["ngh"],
            "gh": ["g"],
            "g": ["gh"],
            "gi": ["d", "r", "v"],
            "v": ["gi", "d"],
            "r": ["d", "gi"],
            "tr": ["ch"],
            "ch": ["tr"],
            "n": ["l"],
            "l": ["n"],
            "kh": ["k"],
            "qu": ["u"],
            "u": ["qu"],
            "nh": ["nh"]
        }
        self.final_consonant = {
            "c": ["t"],
            "t": ["c"],
            "n": ["ng", "nh"],
            "ng": ["n"],
            "nh": ["n"],
            "i": ["y"],
            "y": ["i"]
        }

    @staticmethod
    def check_pos_consonant(word, mode):
        if mode == 'begin':
            if any(char.isalpha() for char in word):
                while not word[0].isalpha():
                    word = word[1:]
            if word[0] in beginConsonant or word[:2] in beginConsonant or word[:3] in beginConsonant:
                return True
        else:
            if any(char.isalpha() for char in word):
                while not word[-1].isalpha():
                    word = word[:-1]
            if word[-1] in finalConsonant or word[-2:] in finalConsonant:
                return True
        return False

    def sample_uppercase(self, word, mode):
        result = ''
        if mode == 'begin':
            prefix = self.sample(self.begin_consonant[word.lower()], 1)[0]
            if word.isupper():
                result += prefix.upper()
            elif word[0].isupper():
                result += prefix[0].upper()
                if len(prefix) > 1:
                    result += prefix[1:]
            else:
                result += prefix
        else:
            prefix = self.sample(self.final_consonant[word.lower()], 1)[0]
            if word.isupper():
                result += prefix.upper()
            else:
                result += prefix
        return result

    def substitute_data(self, data, mode):
        tokens = self.tokenizer(data)
        new_tokens = []
        index_new_tokens = []
        for token_i, token in enumerate(tokens):
            if self.check_pos_consonant(token.lower(), mode):
                new_tokens.append(token)
                index_new_tokens.append(token_i)
        if new_tokens:
            aug_word_idxes = self._get_aug_idxes(new_tokens, self.aug_word_min, self.aug_word_max, self.aug_word_p,
                                                 Method.WORD)
            for token_i, token in enumerate(new_tokens):
                if token_i not in aug_word_idxes:
                    continue
                result = ''
                if mode == 'begin':
                    if any(char.isalpha() for char in token):
                        while not token[0].isalpha():
                            result += token[0]
                            token = token[1:]
                    if token[:3].lower() in beginConsonant:
                        if np.random.random() < self.aug_char_p:
                            result += self.sample_uppercase(token[:3], mode)
                            result += token[3:]
                    elif token[:2].lower() in beginConsonant:
                        if np.random.random() < self.aug_char_p:
                            result = self.sample_uppercase(token[:2], mode)
                            result += token[2:]
                    else:
                        if np.random.random() < self.aug_char_p:
                            result = self.sample_uppercase(token[:1], mode)
                            result += token[1:]
                else:
                    end_consonants = ''
                    if any(char.isalpha() for char in token):
                        while not token[-1].isalpha():
                            end_consonants += token[-1]
                            token = token[:-1]
                    if token[-2:].lower() in finalConsonant:
                        if np.random.random() < self.aug_char_p:
                            result = token[:-2] + self.sample_uppercase(token[-2:], mode) + end_consonants
                    else:
                        if np.random.random() < self.aug_char_p:
                            result = token[:-1] + self.sample_uppercase(token[-1], mode) + end_consonants

                if result:
                    tokens[index_new_tokens[token_i]] = result

            return self.reverse_tokenizer(tokens)
        return data
