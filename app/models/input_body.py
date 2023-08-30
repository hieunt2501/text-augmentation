from typing import AnyStr, List, Dict, Optional

from pydantic import BaseModel, Field


class BaseBody(BaseModel):
    text: AnyStr
    exclude: List[AnyStr] = Field(
        default=[], title="List of tokens to be excluded while augmenting"
    )
    is_segmented: bool = Field(
        default=False, title="Sentence is already word segmented"
    )
    segment: bool = Field(
        default=False, title="Want to word segment"
    )

    @property
    def text_str(self):
        return self.text.decode("utf-8")

    @property
    def exclude_lst(self):
        return [token.decode("utf-8") for token in self.exclude]

    @property
    def is_segmented_bool(self):
        return self.is_segmented

    @property
    def segment_bool(self):
        return self.segment

    def __repr__(self):
        return f"{self.text} {self.exclude} {self.is_segmented} {self.segment}"


class GeneralAugmentationBody(BaseBody):
    action: AnyStr
    p_aug: float = Field(
        default=0.5, title="Percentage of token will be augmented."
    )
    min_aug: int = Field(
        default=1, title="Minimum number of token get augmented"
    )
    max_aug: int = Field(
        default=2, title="Maximum number of token get augmented")

    @property
    def action_str(self):
        return self.action.decode("utf-8")

    @property
    def p_aug_str(self):
        return self.p_aug

    @property
    def min_aug_str(self):
        return self.min_aug

    @property
    def max_aug_str(self):
        return self.max_aug

    def __repr__(self):
        return f"{self.text} {self.action} {self.p_aug} {self.min_aug} {self.max_aug}"


class SynonymBody(GeneralAugmentationBody):
    action: AnyStr = Field(
        title="Action for augmenting, choose from {insert, substitute}"
    )
    num_similar: int = Field(
        default=5, title="Number of synonym to find for each token"
    )
    num_keep: int = Field(
        default=1, title="Number of synonym to keep for each token. Each synonym will create separate sentence"
    )

    @property
    def num_similar_str(self):
        return self.num_similar

    @property
    def num_keep_str(self):
        return self.num_keep

    def __repr__(self):
        return f"{self.text} {self.p_aug} {self.min_aug} {self.max_aug} {self.num_similar} {self.num_keep}"


class SpellingBody(GeneralAugmentationBody):
    aug_char_p: float = Field(
        default=0.3, title="Percentage of character (per token) will be augmented"
    )

    @property
    def aug_char_p_str(self):
        return self.aug_char_p

    def __repr__(self):
        return f"{self.text} {self.p_aug} {self.min_aug} {self.max_aug} {self.aug_char_p}"


class TypoBody(SpellingBody):
    action: AnyStr = Field(
        title="Action for augmenting, choose from {telex, vni, keyboard}"
    )


class AccentBody(SpellingBody):
    action: AnyStr = Field(
        title="Action for augmenting, choose from {missing, none, wrong}"
    )


class WordAugmentationBody(SpellingBody):
    action: AnyStr = Field(
        title="Action for augmenting, choose from {duplicate, insert, edit_distance, swap, delete}"
    )


class CharAugmentationBody(SpellingBody):
    action: AnyStr = Field(
        title="Action for augmenting, choose from {duplicate, random, misspell_vowel, substitute, whitespace}"
    )


class SpellingReplaceBody(SpellingBody):
    action: AnyStr = Field(
        title="Action for augmenting, choose from {begin, final}"
    )


class BackTranslationBody(BaseBody):
    src_language: AnyStr = Field(
        default="vi", title="Source language. Default to Vie",
    )
    languages: List[AnyStr] = Field(
        title="Intermediate language, use googletrans language list"
    )

    @property
    def src_language_str(self):
        return self.src_language.decode("utf-8")

    @property
    def languages_lst(self):
        return [lang.decode("utf-8") for lang in self.languages]

    def __repr__(self):
        return f"{self.text} {self.src_language} {self.languages}"


class BlankNoiseBody(GeneralAugmentationBody):
    action: AnyStr = Field(
        default=None, title="Not use here")


class DependencyTreeBody(BaseBody):
    pass


class PipelineAugBody(BaseModel):
    type: str = Field(
        default="synonym", title="Type of augmenter"
    )
    action: Optional[str]
    languages: Optional[List[str]] = Field(
        title="Intermediate language, use googletrans language list"
    )
    p_aug: Optional[float] = Field(
        default=0.5, title="Percentage of token will be augmented."
    )
    min_aug: Optional[int] = Field(
        default=1, title="Minimum number of token get augmented"
    )
    max_aug: Optional[int] = Field(
        default=2, title="Maximum number of token get augmented")
    aug_char_p: Optional[float] = Field(
        default=0.3, title="Percentage of character (per token) will be augmented"
    )
    num_similar: Optional[int] = Field(
        default=5, title="Number of synonym to find for each token"
    )
    num_keep: Optional[int] = Field(
        default=1, title="Number of synonym to keep for each token. Each synonym will create separate sentence"
    )

    @property
    def num_similar_str(self):
        return self.num_similar

    @property
    def num_keep_str(self):
        return self.num_keep

    @property
    def aug_char_p_str(self):
        return self.aug_char_p

    @property
    def action_str(self):
        return self.action

    @property
    def p_aug_str(self):
        return self.p_aug

    @property
    def min_aug_str(self):
        return self.min_aug

    @property
    def max_aug_str(self):
        return self.max_aug

    @property
    def languages_lst(self):
        return self.languages

    def __repr__(self):
        return f"{self.type} {self.action} {self.p_aug} {self.min_aug} {self.max_aug} {self.aug_char_p} {self.num_similar} {self.languages}"


class PipelineBody(BaseBody):
    pipeline: List[PipelineAugBody] = Field(
        default=[], title="Whole pipeline for text augmentation per sentence"
    )
    n_sent: int

    @property
    def pipeline_lst(self):
        return self.pipeline

    @property
    def n_sent_str(self):
        return self.n_sent

    def __repr__(self):
        return f"{self.text} {self.pipeline}"
