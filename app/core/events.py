from typing import Callable

from fastapi import FastAPI


def preload_model():
    """
    In order to load model on memory to each worker
    """
    from app.services.word_segment.word_segment import TextProcessor
    from app.services.synonym_handler import SynonymHandler
    from app.services.backtranslation_handler import BackTranslationHandler
    from app.services.pipeline_handler import PipelineHandler
    from app.services.spelling_handler import (
        TypoHandler,
        AccentHandler,
        SpellingReplacementHandler,
        WordHandler,
        CharHandler
    )

    SynonymHandler.get_model()
    BackTranslationHandler.get_model()
    SpellingReplacementHandler.get_model()
    TypoHandler.get_model()
    AccentHandler.get_model()
    WordHandler.get_model()
    CharHandler.get_model()
    PipelineHandler.get_model()
    TextProcessor.get_model()


def create_start_app_handler(app: FastAPI) -> Callable:
    def start_app() -> None:
        preload_model()

    return start_app
