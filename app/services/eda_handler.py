import random
import string
from typing import List

import numpy as np

from app.services.base_augmenter import Augmenter


class EdaHandler(Augmenter):
    """
    Easy Data Augmentation with Swap and Delete augmentation

    https://arxiv.org/pdf/1901.11196.pdf
    """

    def _is_eligible_token(self, token):
        return token not in string.punctuation and not token.startswith("MASK")

    @staticmethod
    def validate_action(action):
        assert action in ["delete", "swap"], "Please choose action in {delete, swap}"

    def transform(self, action, tokens, eligible_indices, **kwargs) -> List[str]:
        augmented_data = []
        tmp = tokens.copy()

        eligible_indices = np.array(eligible_indices)

        for idx in eligible_indices:
            if action == "delete":
                tmp.pop(idx)
                eligible_indices[eligible_indices > idx] = eligible_indices[eligible_indices > idx] - 1
            elif action == "swap":
                org_indices = list(range(len(tmp)))

                random_idx = random.choice(org_indices)
                while tmp[random_idx].startswith("MASK"):
                    random_idx = random.choice(org_indices)
                tmp[idx], tmp[random_idx] = tmp[random_idx], tmp[idx]
        #     if action == "delete":
        #         tmp.pop(idx)
        #     elif action == "swap":
        #         org_indices = list(range(len(tmp)))
        #         org_indices.pop(idx)
        #         tmp[idx], tmp[random_idx] = tmp[random_idx], tmp[idx]

        augmented_data.append(" ".join(tmp))

        return self.remove_duplicate(augmented_data)
