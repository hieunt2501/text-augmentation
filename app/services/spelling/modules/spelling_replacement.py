from abc import ABC

from app.services.spelling.modules.base_module import SpellingReplacementAugmenter


class SpellingReplacementBeginHandler(SpellingReplacementAugmenter, ABC):
    def __init__(self, name='SpellingReplacementAugmenterBegin', min_char=2, aug_char_p=0.3,
                 aug_word_min=1, aug_word_max=100, aug_word_p=0.3, tokenizer=None, reverse_tokenizer=None,
                 stopwords=None, verbose=0, stopwords_regex=None):
        super().__init__(
            name=name, min_char=min_char, aug_char_p=aug_char_p, aug_word_min=aug_word_min,
            aug_word_max=aug_word_max, aug_word_p=aug_word_p, tokenizer=tokenizer,
            reverse_tokenizer=reverse_tokenizer, stopwords=stopwords,
            verbose=verbose, stopwords_regex=stopwords_regex)

    def substitute(self, data):
        return self.substitute_data(data, 'begin')


class SpellingReplacementFinalHandler(SpellingReplacementAugmenter, ABC):
    def __init__(self, name='SpellingReplacementAugmenterFinal', min_char=2, aug_char_p=0.3,
                 aug_word_min=1, aug_word_max=100, aug_word_p=0.3, tokenizer=None, reverse_tokenizer=None,
                 stopwords=None, verbose=0, stopwords_regex=None):
        super().__init__(
            name=name, min_char=min_char, aug_char_p=aug_char_p, aug_word_min=aug_word_min,
            aug_word_max=aug_word_max, aug_word_p=aug_word_p, tokenizer=tokenizer,
            reverse_tokenizer=reverse_tokenizer, stopwords=stopwords,
            verbose=verbose, stopwords_regex=stopwords_regex)

    def substitute(self, data):
        return self.substitute_data(data, 'final')
