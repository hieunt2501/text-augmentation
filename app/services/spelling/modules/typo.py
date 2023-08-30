from abc import ABC

import nlpaug.augmenter.char as nac
from nlpaug.util import Action, Method, Doc

import app.services.spelling.utils as utils
from app.services.spelling.modules.base_module import TypoAugmenter


class TelexHandler(TypoAugmenter, ABC):
    def __init__(self, name='TelexAugmenter', min_char=2, aug_char_min=1, aug_char_max=10, aug_char_p=0.3,
                 aug_word_min=1, aug_word_max=10, aug_word_p=0.3, tokenizer=None, reverse_tokenizer=None,
                 stopwords=None, verbose=0, stopwords_regex=None):
        super().__init__(
            name=name, min_char=min_char, aug_char_min=aug_char_min,
            aug_char_max=aug_char_max, aug_char_p=aug_char_p, aug_word_min=aug_word_min,
            aug_word_max=aug_word_max, aug_word_p=aug_word_p, tokenizer=tokenizer,
            reverse_tokenizer=reverse_tokenizer, stopwords=stopwords,
            verbose=verbose, stopwords_regex=stopwords_regex)
        self.typo = self._get_typo_dictionary()

    def _get_typo_dictionary(self):
        return {
            "á": ["a", "s"], "à": ["a", "f"],
            "ạ": ["a", "j"], "ả": ["a", "r"],
            "ã": ["a", "x"], "â": ["a", "a"],
            "ấ": ["a", "a", "s"], "ầ": ["a", "a", "f"],
            "ậ": ["a", "a", "j"], "ẩ": ["a", "a", "r"],
            "ẫ": ["a", "a", "x"], "ă": ["a", "w"],
            "ắ": ["a", "w", "s"], "ằ": ["a", "w", "f"],
            "ặ": ["a", "w", "j"], "ẳ": ["a", "w", "r"],
            "ẵ": ["a", "w", "x"], "í": ["i", "s"],
            "ì": ["i", "f"], "ỉ": ["i", "r"],
            "ĩ": ["i", "x"], "ị": ["i", "j"],
            "ú": ["u", "s"], "ù": ["u", "f"],
            "ủ": ["u", "r"], "ũ": ["u", "x"],
            "ụ": ["u", "j"], "ư": ["u", "w"],
            "ứ": ["u", "w", "s"], "ừ": ["u", "w", "f"],
            "ử": ["u", "w", "r"], "ữ": ["u", "w", "x"],
            "ự": ["u", "w", "j"], "é": ["e", "s"],
            "è": ["e", "f"], "ẻ": ["e", "r"],
            "ẽ": ["e", "x"], "ẹ": ["e", "j"],
            "ê": ["e", "e"], "ế": ["e", "e", "s"],
            "ề": ["e", "e", "f"], "ể": ["e", "e", "r"],
            "ễ": ["e", "e", "x"], "ệ": ["e", "e", "j"],
            "ó": ["o", "s"], "ò": ["o", "f"],
            "ỏ": ["o", "r"], "õ": ["o", "x"],
            "ọ": ["o", "j"], "ô": ["o", "o"],
            "ố": ["o", "o", "s"], "ồ": ["o", "o", "f"],
            "ổ": ["o", "o", "r"], "ỗ": ["o", "o", "x"],
            "ộ": ["o", "o", "j"], "ơ": ["o", "w"],
            "ớ": ["o", "w", "s"], "ờ": ["o", "w", "f"],
            "ở": ["o", "w", "r"], "ỡ": ["o", "w", "x"],
            "ợ": ["o", "w", "j"], 'đ': ['d', 'd'],
            "ý": ["y", "s"], "ỳ": ["y", "f"],
            "ỷ": ["y", "r"], "ỹ": ["y", "x"],
            "ỵ": ["y", "j"], "ươ": ['uo', "w"],
            "ướ": ['uo', "w", "s"], "ườ": ['uo', 'w', 'f'],
            "ưỡ": ['uo', "w", "x"], "ưở": ['uo', 'w', 'r'],
            "ượ": ['uo', "w", "j"]}

    def _get_new_decomposition(self, base_word, comp_word):
        if len(comp_word) == 1:
            return base_word, comp_word
        if base_word == 'o':
            base_word = ('ô' if comp_word[0] == 'o' else 'ơ')
        elif base_word == 'a':
            base_word = ('â' if comp_word[0] == 'a' else 'ă')
        elif base_word == 'e':
            base_word = 'ê'
        elif len(base_word) == 2:
            base_word = 'ươ'
        comp_word = comp_word[1:]

        return base_word, comp_word


class VNIHandler(TypoAugmenter, ABC):
    def __init__(self, name='VNIAugmenter', min_char=2, aug_char_min=1, aug_char_max=10, aug_char_p=0.3,
                 aug_word_min=1, aug_word_max=10, aug_word_p=0.3, tokenizer=None, reverse_tokenizer=None,
                 stopwords=None, verbose=0, stopwords_regex=None):
        super().__init__(
            name=name, min_char=min_char, aug_char_min=aug_char_min,
            aug_char_max=aug_char_max, aug_char_p=aug_char_p, aug_word_min=aug_word_min,
            aug_word_max=aug_word_max, aug_word_p=aug_word_p, tokenizer=tokenizer,
            reverse_tokenizer=reverse_tokenizer, stopwords=stopwords,
            verbose=verbose, stopwords_regex=stopwords_regex)
        self.typo = self._get_typo_dictionary()

    def _get_typo_dictionary(self):
        return {
            "á": ["a", "1"], "à": ["a", "2"],
            "ạ": ["a", "5"], "ả": ["a", "3"],
            "ã": ["a", "4"], "â": ["a", "a"],
            "ấ": ["a", "6", "1"], "ầ": ["a", "6", "2"],
            "ậ": ["a", "6", "5"], "ẩ": ["a", "6", "3"],
            "ẫ": ["a", "6", "4"], "ă": ["a", "7"],
            "ắ": ["a", "7", "1"], "ằ": ["a", "7", "2"],
            "ặ": ["a", "7", "5"], "ẳ": ["a", "7", "3"],
            "ẵ": ["a", "7", "4"], "í": ["i", "1"],
            "ì": ["i", "2"], "ỉ": ["i", "3"],
            "ĩ": ["i", "4"], "ị": ["i", "5"],
            "ú": ["u", "1"], "ù": ["u", "2"],
            "ủ": ["u", "3"], "ũ": ["u", "4"],
            "ụ": ["u", "5"], "ư": ["u", "7"],
            "ứ": ["u", "7", "1"], "ừ": ["u", "7", "2"],
            "ử": ["u", "7", "3"], "ữ": ["u", "7", "4"],
            "ự": ["u", "7", "5"], "é": ["e", "1"],
            "è": ["e", "2"], "ẻ": ["e", "3"],
            "ẽ": ["e", "4"], "ẹ": ["e", "5"],
            "ê": ["e", "6"], "ế": ["e", "6", "1"],
            "ề": ["e", "6", "2"], "ể": ["e", "6", "3"],
            "ễ": ["e", "6", "4"], "ệ": ["e", "6", "5"],
            "ó": ["o", "1"], "ò": ["o", "2"],
            "ỏ": ["o", "3"], "õ": ["o", "4"],
            "ọ": ["o", "5"], "ô": ["o", "o"],
            "ố": ["o", "6", "1"], "ồ": ["o", "6", "2"],
            "ổ": ["o", "6", "3"], "ỗ": ["o", "6", "4"],
            "ộ": ["o", "6", "5"], "ơ": ["o", "7"],
            "ớ": ["o", "7", "1"], "ờ": ["o", "7", "2"],
            "ở": ["o", "7", "3"], "ỡ": ["o", "7", "4"],
            "ợ": ["o", "7", "5"], 'đ': ['d', '9'],
            "ý": ["y", "1"], "ỳ": ["y", "2"],
            "ỷ": ["y", "3"], "ỹ": ["y", "4"],
            "ỵ": ["y", "5"], 'ươ': ['uo', '7'],
            "ướ": ['uo', "7", "1"], "ườ": ['uo', '7', '2'],
            "ưỡ": ['uo', "7", "4"], "ưở": ['uo', '7', '3'],
            "ượ": ['uo', "7", "5"]
        }

    def _get_new_decomposition(self, base_word, comp_word):
        if len(comp_word) == 1:
            return base_word, comp_word
        if base_word == 'o':
            base_word = ('ô' if comp_word[0] == '6' else 'ơ')
        elif base_word == 'a':
            base_word = ('â' if comp_word[0] == '6' else 'ă')
        elif base_word == 'e':
            base_word = 'ê'
        elif len(base_word) == 2:
            base_word = 'ươ'
        comp_word = comp_word[1:]
        return base_word, comp_word


class KeyboardHandler(nac.KeyboardAug, ABC):
    def __init__(self, name='KeyboardHandler', aug_char_min=1, aug_char_max=10, aug_char_p=0.3,
                 aug_word_p=0.3, aug_word_min=1, aug_word_max=10, stopwords=None,
                 tokenizer=None, reverse_tokenizer=None, include_special_char=False, include_numeric=False,
                 include_upper_case=False, lang="en", verbose=0, stopwords_regex=None, model_path=None,
                 min_char=1):
        super().__init__(name=name, aug_char_min=aug_char_min, aug_char_max=aug_char_max, aug_char_p=aug_char_p,
                         aug_word_p=aug_word_p, aug_word_min=aug_word_min, aug_word_max=aug_word_max,
                         stopwords=stopwords,
                         tokenizer=tokenizer, reverse_tokenizer=reverse_tokenizer,
                         include_special_char=include_special_char,
                         include_numeric=include_numeric, include_upper_case=include_upper_case, lang=lang,
                         verbose=verbose,
                         stopwords_regex=stopwords_regex, model_path=model_path, min_char=min_char)
        self.telexDecomposer = TelexHandler()

    def substitute(self, data):
        if not data or not data.strip():
            return data

        change_seq = 0

        doc = Doc(data, self.tokenizer(data))
        gaps = utils.find_all_gaps(data, doc.get_original_tokens())

        aug_word_idxes = self._get_aug_idxes(doc.get_original_tokens(), self.aug_word_min,
                                             self.aug_word_max, self.aug_word_p, Method.WORD)
        for token_i, token in enumerate(doc.get_original_tokens()):
            if token_i not in aug_word_idxes:
                continue

            new_token = ""
            token = self.telexDecomposer.generate_word_error(token)
            chars = self.token2char(token)
            aug_char_idxes = self._get_aug_idxes(chars, self.aug_char_min, self.aug_char_max,
                                                 self.aug_char_p, Method.CHAR)
            if aug_char_idxes is None:
                continue

            for char_i, char in enumerate(chars):
                if char_i not in aug_char_idxes:
                    new_token += char
                    continue

                new_token += self.sample(self.model.predict(chars[char_i]), 1)[0]

            # No capitalization alignment as this augmenter try to simulate typo
            change_seq += 1
            doc.add_change_log(token_i, new_token=new_token, action=Action.SUBSTITUTE,
                               change_seq=self.parent_change_seq + change_seq)

        if self.include_detail:
            return utils.reverse_tokenizer(doc.get_augmented_tokens(), gaps), doc.get_change_logs()
        else:
            return utils.reverse_tokenizer(doc.get_augmented_tokens(), gaps)
