import random
from typing import List

from loguru import logger

from app.services.utils import revert_segmented_tokens, tokenize, mask_exclude_tokens, reconstruct
from app.services.word_segment.word_segment import TextProcessor
from app.services.synonym_handler import SynonymHandler
from app.services.blank_noise_handler import BlankNoiseHandler
from app.services.backtranslation_handler import BackTranslationHandler
from app.services.tree_handler import TreeHandler
from app.services.spelling_handler import (
    TypoHandler,
    AccentHandler,
    SpellingReplacementHandler,
    WordHandler,
    CharHandler
)

fmt = "{time} - {name} - {level} - {message}"
logger.add("logs/pipeline.log", level="DEBUG", format=fmt, backtrace=True)
text_processor = TextProcessor()


class PipelineHandler:
    handlers = dict()

    @classmethod
    def get_model(cls):
        if not cls.handlers:
            cls.handlers["blank"] = BlankNoiseHandler()
            cls.handlers["synonym"] = SynonymHandler()
            cls.handlers["dependency_tree"] = TreeHandler()
            cls.handlers["backtranslation"] = BackTranslationHandler()
            cls.handlers["typo"] = TypoHandler()
            cls.handlers["accent"] = AccentHandler()
            cls.handlers["spelling_replace"] = SpellingReplacementHandler()
            cls.handlers["word"] = WordHandler()
            cls.handlers["char"] = CharHandler()

    @staticmethod
    def remove_duplicate(augmented_list: List[str]):
        return list(set(augmented_list))

    def augment(self, text, exclude, pipeline, is_segmented, segment, n_sent):
        results = []
        transform_text = text

        try:
            if is_segmented:
                tokens = tokenize(transform_text)
                transform_text = " ".join(revert_segmented_tokens(tokens))

            for _ in range(n_sent):
                num_action = random.choice(range(1, len(pipeline))) if len(pipeline) > 1 else 1
                tmp_pipeline = random.sample(pipeline, num_action)

                for aug_type in tmp_pipeline:
                    aug_type = vars(aug_type)
                    tmp_text = self.handlers[aug_type["type"]].augment(text=transform_text,
                                                                    exclude=exclude,
                                                                    is_segmented=False,
                                                                    segment=False,
                                                                    **aug_type)
    
                    if not tmp_text:
                        continue

                    if isinstance(tmp_text, list):
                        tmp_text = tmp_text[0]

                    tmp_text, exclude_map = mask_exclude_tokens(tmp_text, exclude)

                    if segment:
                        tmp_text = text_processor.process(tmp_text)
                    
                    tmp_text = reconstruct(tmp_text, exclude_map)
                    results.append(tmp_text)
                            
        except:
            logger.debug(type(pipeline))
            logger.debug(pipeline)
            logger.exception("ERROR IN PIPELINE")
        
        if not results:
            return [text]

        return self.remove_duplicate(results)
