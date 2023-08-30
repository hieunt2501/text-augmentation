from app.services.base_augmenter import Augmenter
from app.services.spelling.modules.base_module import default_tokenizer
from app.services.spelling.modules.typo import *
from app.services.spelling.modules.accent import *
from app.services.spelling.modules.spelling_replacement import *
from app.services.spelling.modules.word import *
from app.services.spelling.modules.char import *
from app.services.utils import tokenize, revert_segmented_tokens, mask_exclude_tokens, reconstruct
from app.services.word_segment.word_segment import TextProcessor
from app.services.eda_handler import EdaHandler
from app.core.config import IRRELEVANT_WORD_PATH, EDIT_DISTANCE_PATH

text_processor = TextProcessor()


def change_config(aug_handler, p_aug, min_aug, max_aug, exclude=None, aug_char_p=None, aug_type="norm"):
    if exclude:
        aug_handler.stop_words = exclude

    if aug_type == "word":
        aug_handler.aug_p = p_aug
        aug_handler.aug_min = min_aug
        aug_handler.aug_max = max_aug
    else:
        aug_handler.aug_word_p = p_aug
        aug_handler.aug_word_min = min_aug
        aug_handler.aug_word_max = max_aug

        if aug_char_p:
            aug_handler.aug_char_p = aug_char_p


class BaseSpellingHandler(Augmenter, ABC):
    def transform_spelling(self, action, text, p_aug, min_aug, max_aug, exclude, **kwargs):
        raise NotImplementedError

    def augment(self, action, text, p_aug, min_aug, max_aug, exclude, is_segmented, segment, **kwargs):
        self.validate_action(action)

        text, exclude_map = mask_exclude_tokens(text, exclude)

        if is_segmented:
            tokens = tokenize(text)
            text = " ".join(revert_segmented_tokens(tokens))

        transform_text = self.transform_spelling(action, text, p_aug, min_aug, max_aug, (exclude_map.keys()), **kwargs)

        if segment:
            transform_text = [text_processor.process(t) for t in transform_text]

        transform_text = [reconstruct(t, exclude_map) for t in transform_text]
        return transform_text


class AccentHandler(BaseSpellingHandler, ABC):
    missing_dialect_aug = None
    no_dialect_aug = None
    wrong_dialect_aug = None

    @classmethod
    def get_model(cls):
        if not cls.missing_dialect_aug or not cls.no_dialect_aug or not cls.wrong_dialect_aug:
            cls.missing_dialect_aug = MissingDialectHandler(aug_word_p=1, aug_word_min=1, aug_word_max=2)
            cls.no_dialect_aug = NoDialectHandler(aug_word_p=1, aug_word_min=1, aug_word_max=2)
            cls.wrong_dialect_aug = WrongDialectHandler(aug_word_p=1, aug_word_min=1, aug_word_max=2)

    def _is_eligible_token(self, token):
        pass

    @staticmethod
    def validate_action(action):
        assert action in ["missing", "none", "wrong"], \
            "Please choose action in {missing, none, wrong}"

    def transform_spelling(self, action, text, p_aug, min_aug, max_aug, exclude, **kwargs):
        if action == "missing_dialect":
            change_config(self.missing_dialect_aug, p_aug, min_aug, max_aug, exclude=exclude)
            transform_text = [self.missing_dialect_aug.augment(text)]
        elif action == "no_dialect":
            change_config(self.no_dialect_aug, p_aug, min_aug, max_aug, exclude=exclude)
            transform_text = [self.no_dialect_aug.augment(text)]
        else:
            change_config(self.wrong_dialect_aug, p_aug, min_aug, max_aug, exclude=exclude)
            transform_text = [self.wrong_dialect_aug.augment(text)]

        return transform_text


class TypoHandler(BaseSpellingHandler, ABC):
    telex_aug = None
    vni_aug = None
    keyboard_aug = None

    @classmethod
    def get_model(cls):
        if not cls.telex_aug or not cls.vni_aug or not cls.keyboard_aug:
            cls.telex_aug = TelexHandler(aug_word_p=1, aug_word_min=1, aug_word_max=2)
            cls.vni_aug = VNIHandler(aug_word_p=1, aug_word_min=1, aug_word_max=2)
            cls.keyboard_aug = KeyboardHandler(aug_word_p=1, aug_word_min=1, aug_word_max=2)

    def _is_eligible_token(self, token):
        pass

    def transform_spelling(self, action, text, p_aug, min_aug, max_aug, exclude, **kwargs):
        if action == "telex":
            change_config(self.telex_aug, p_aug, min_aug, max_aug, exclude=exclude)
            transform_text = [self.telex_aug.augment(text)]
        elif action == "vni":
            change_config(self.vni_aug, p_aug, min_aug, max_aug, exclude=exclude)
            transform_text = [self.vni_aug.augment(text)]
        else:
            change_config(self.keyboard_aug, p_aug, min_aug, max_aug, exclude=exclude)
            transform_text = [self.keyboard_aug.augment(text)]
        return transform_text

    @staticmethod
    def validate_action(action):
        assert action in ["telex", "vni", "keyboard"], \
            "Please choose action in {telex, vni, keyboard}"


class SpellingReplacementHandler(BaseSpellingHandler, ABC):
    begin_aug = None
    final_aug = None

    @classmethod
    def get_model(cls):
        if not cls.begin_aug or not cls.final_aug:
            cls.begin_aug = SpellingReplacementBeginHandler(tokenizer=default_tokenizer,
                                                            aug_word_p=1,
                                                            aug_word_min=1,
                                                            aug_word_max=2,
                                                            aug_char_p=0.6)
            cls.final_aug = SpellingReplacementFinalHandler(tokenizer=default_tokenizer,
                                                            aug_word_p=1,
                                                            aug_word_min=1,
                                                            aug_word_max=2,
                                                            aug_char_p=0.6)

    def _is_eligible_token(self, token):
        pass

    @staticmethod
    def validate_action(action):
        assert action in ["begin", "final"], \
            "Please choose action in {begin, final}"

    def transform_spelling(self, action, text, p_aug, min_aug, max_aug, exclude, **kwargs):
        if action == "begin":
            change_config(self.begin_aug, p_aug, min_aug, max_aug, exclude=exclude, aug_char_p=kwargs["aug_char_p"])
            transform_text = [self.begin_aug.augment(text)]
        else:
            change_config(self.final_aug, p_aug, min_aug, max_aug, exclude=exclude, aug_char_p=kwargs["aug_char_p"])
            transform_text = [self.final_aug.augment(text)]

        return transform_text


class WordHandler(BaseSpellingHandler, ABC):
    duplicate_aug = None
    insert_irrelevant_aug = None
    edit_distance_aug = None
    split_aug = None
    eda_aug = None

    @classmethod
    def get_model(cls):
        if not cls.duplicate_aug or \
                not cls.insert_irrelevant_aug or \
                not cls.edit_distance_aug or \
                not cls.split_aug or not \
                cls.eda_aug:
            cls.duplicate_aug = DuplicateWordHandler(aug_p=1, aug_min=1, aug_max=2)
            cls.insert_irrelevant_aug = InsertIrrelevantWordHandler(file_path=IRRELEVANT_WORD_PATH,
                                                                    aug_p=1,
                                                                    aug_min=1,
                                                                    aug_max=2)
            cls.edit_distance_aug = EditDistanceHandler(dict_path=EDIT_DISTANCE_PATH,
                                                        include_reverse=False,
                                                        aug_p=1,
                                                        aug_min=1,
                                                        aug_max=2)
            cls.split_aug = naw.SplitAug(aug_p=1, aug_min=1, aug_max=2)
            cls.eda_aug = EdaHandler()

    def _is_eligible_token(self, token):
        pass

    @staticmethod
    def validate_action(action):
        assert action in ["duplicate", "insert", "edit_distance", "split", "swap", "delete"], \
            "Please choose action in {duplicate, insert, edit_distance, swap, delete, split}"

    def transform_spelling(self, action, text, p_aug, min_aug, max_aug, exclude, **kwargs):
        if action == "duplicate":
            change_config(self.duplicate_aug, p_aug, min_aug, max_aug, exclude=exclude, aug_type="word")
            transform_text = [self.duplicate_aug.augment(text)]
        elif action == "insert":
            change_config(self.insert_irrelevant_aug, p_aug, min_aug, max_aug, exclude=exclude, aug_type="word")
            transform_text = [self.insert_irrelevant_aug.augment(text)]
        elif action == "split":
            change_config(self.split_aug, p_aug, min_aug, max_aug, exclude=exclude, aug_type="word")
            transform_text = [self.split_aug.augment(text)]
        elif action == "edit_distance":
            change_config(self.edit_distance_aug, p_aug, min_aug, max_aug, exclude=exclude, aug_type="word")
            transform_text = [self.edit_distance_aug.augment(text)]
        else:
            transform_text = self.eda_aug.augment(action,
                                                  text,
                                                  p_aug,
                                                  min_aug,
                                                  max_aug,
                                                  exclude=exclude,
                                                  is_segmented=False,
                                                  segment=False)

        return transform_text


class CharHandler(BaseSpellingHandler, ABC):
    random_aug = None
    substitute_aug = None
    misspell_vowel_aug = None
    duplicate_aug = None
    whitespace_aug = None

    @classmethod
    def get_model(cls):
        if not cls.random_aug \
                or not cls.substitute_aug \
                or not cls.misspell_vowel_aug \
                or not cls.duplicate_aug \
                or not cls.whitespace_aug:
            cls.random_aug = RandomCharHandler(aug_word_p=1, aug_word_min=1, aug_word_max=2)
            cls.substitute_aug = SubstituteHandler(tokenizer=default_tokenizer,
                                                   aug_word_p=1,
                                                   aug_word_min=1,
                                                   aug_word_max=2,
                                                   aug_char_p=0.1)
            cls.misspell_vowel_aug = MisspellVowelHandler(aug_word_p=1, aug_word_min=1, aug_word_max=2)
            cls.duplicate_aug = DuplicateHandler(aug_word_p=1, aug_word_min=1, aug_word_max=2)
            cls.whitespace_aug = WhitespaceHandler(aug_word_p=1, aug_word_min=1, aug_word_max=2)

    def _is_eligible_token(self, token):
        pass

    @staticmethod
    def validate_action(action):
        assert action in ["duplicate", "random", "misspell_vowel", "substitute", "whitespace"], \
            "Please choose action in {duplicate, random, misspell_vowel, substitute, whitespace}"

    def transform_spelling(self, action, text, p_aug, min_aug, max_aug, exclude, **kwargs):
        if action == "random":
            change_config(self.random_aug, p_aug, min_aug, max_aug)
            transform_text = [self.random_aug.augment(text)]
        elif action == "substitute":
            change_config(self.substitute_aug, p_aug, min_aug, max_aug, exclude=exclude, aug_char_p=kwargs["aug_char_p"])
            transform_text = [self.substitute_aug.augment(text)]
        elif action == "misspell_vowel":
            change_config(self.misspell_vowel_aug, p_aug, min_aug, max_aug, exclude=exclude)
            transform_text = [self.misspell_vowel_aug.augment(text)]
        elif action == "duplicate":
            # TODO: check carefully here
            change_config(self.duplicate_aug, p_aug, min_aug, max_aug, exclude=exclude)
            transform_text = [self.duplicate_aug.augment(text)]
        else:
            change_config(self.whitespace_aug, p_aug, min_aug, max_aug, exclude=exclude)
            transform_text = [self.whitespace_aug.augment(text)]

        return transform_text
