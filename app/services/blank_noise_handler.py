import random
import string
from typing import List

from app.services.base_augmenter import Augmenter


class BlankNoiseHandler(Augmenter):
    """
    https://arxiv.org/pdf/1703.02573.pdf
    """

    def _is_eligible_token(self, token):
        return token not in string.punctuation and not token.startswith("MASK")

    def transform(self, action, tokens, eligible_indices, **kwargs) -> List[str]:
        augmented_data = []
        tmp = tokens.copy()

        for idx in eligible_indices:
            tmp[idx] = "_"

        augmented_data.append(" ".join(tmp))

        return self.remove_duplicate(augmented_data)

    @staticmethod
    def validate_action(action):
        pass
