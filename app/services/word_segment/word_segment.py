from vncorenlp import VnCoreNLP

from app.core.config import VN_CORE_PATH, VN_CORE_PORT


class TextProcessor:
    annotator = None

    @classmethod
    def get_model(cls):
        if not cls.annotator:
            cls.annotator = VnCoreNLP(address=VN_CORE_PATH, port=VN_CORE_PORT)

    def process(self, text):
        sentences = self.annotator.tokenize(text)
        annotated_text = [words for sentence in sentences for words in sentence]

        return " ".join(annotated_text)
