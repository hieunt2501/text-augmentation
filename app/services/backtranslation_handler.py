from typing import List

from googletrans import Translator

from app.services.utils import mask_exclude_tokens, reconstruct, tokenize, revert_segmented_tokens
from app.services.word_segment.word_segment import TextProcessor

text_processor = TextProcessor()


class BackTranslationHandler:
    translator = None

    @classmethod
    def get_model(cls):
        if cls.translator is None:
            cls.translator = Translator(service_urls=["translate.google.com"])

    def _translate(self, text, src_lang="vi", dest_lang="en"):
        translated_text = self.translator.translate(text, src=src_lang, dest=dest_lang)
        return translated_text.text

    def augment(self, text, src_language, languages, exclude, is_segmented, segment, **kwargs) -> List[str]:
        transform_text = []
        
        if exclude:
            return [text]
        
        # text, exclude_map = mask_exclude_tokens(text, exclude)

        if is_segmented:
            tokens = tokenize(text)
            text = " ".join(revert_segmented_tokens(tokens))

        for lang in languages:
            translated = self._translate(text, src_lang=src_language, dest_lang=lang)
            back_translated = self._translate(translated, src_lang=lang, dest_lang="vi")
            transform_text.append(back_translated)

        # transform_text = [reconstruct(t, exclude_map) for t in transform_text]

        if segment:
            transform_text = [text_processor.process(t) for t in transform_text]

        return transform_text
