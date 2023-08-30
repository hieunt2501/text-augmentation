import logging
import sys

from loguru import logger
from starlette.config import Config
from starlette.datastructures import Secret

from app.core.logger import InterceptHandler


config = Config(".env")

API_PREFIX = "/api"
VERSION = "0.1.0"
DEBUG: bool = config("DEBUG", cast=bool, default=False)
MAX_CONNECTIONS_COUNT: int = config("MAX_CONNECTIONS_COUNT", cast=int, default=10)
MIN_CONNECTIONS_COUNT: int = config("MIN_CONNECTIONS_COUNT", cast=int, default=10)
HOST: str = config("HOST", cast=str, default="0.0.0.0")
PORT: int = config("PORT", cast=int, default=35100)
SECRET_KEY: Secret = config("SECRET_KEY", cast=Secret, default="")

PROJECT_NAME: str = config("PROJECT_NAME", default="augmentation")

# logging configuration
LOGGING_LEVEL = logging.DEBUG if DEBUG else logging.INFO
logging.basicConfig(
    handlers=[InterceptHandler(level=LOGGING_LEVEL)], level=LOGGING_LEVEL
)
logger.configure(handlers=[{"sink": sys.stderr, "level": LOGGING_LEVEL}])

FASTTEXT_PATH = config("FASTTEXT_PATH", default="./model/cc.vi.300.vec")
PHOBERT_PATH = config("PHOBERT_PATH", default="./model/PhoBERT_base_fairseq")
STOPWORD_PATH = config("STOPWORD_PATH", default="./data/vietnamese-stopwords.txt")
IRRELEVANT_WORD_PATH = config("IRRELEVANT_WORD_PATH", default="./data/irrelevant_words.txt")
EDIT_DISTANCE_PATH = config("EDIT_DISTANCE_PATH", default="./data/edit_distance.txt")
MAX_CACHE_SIZE = config("MAX_CACHE_SIZE", cast=int, default=1000)
PHO_NLP_URL = config("PHO_NLP_URL", default="http://172.29.13.23:20217/")
VN_CORE_PATH = config("VN_CORE_PATH", default="http://172.29.13.23")
VN_CORE_PORT = config("VN_CORE_PORT", cast=int, default=20215)
