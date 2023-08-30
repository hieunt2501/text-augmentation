import random
from typing import List

import numpy as np

from app.services.utils import mask_exclude_tokens, reconstruct, revert_segmented_tokens, tokenize
from app.services.word_segment.word_segment import TextProcessor

text_processor = TextProcessor()


class Augmenter:
    def __int__(self):
        pass

    @staticmethod
    def remove_duplicate(augmented_list: List[str]):
        return list(set(augmented_list))

    def _get_eligible_indices(self, text, p_aug=0.1, min_aug=1, max_aug=10, exclude=None, is_segmented=False):
        if not exclude:
            exclude = list()

        text, exclude_map = mask_exclude_tokens(text, exclude)

        tokens = tokenize(text)
        if is_segmented:
            tokens = revert_segmented_tokens(tokens)

        mask = np.array([self._is_eligible_token(token) for token in tokens])
        indices = np.where(mask)[0].tolist()
        random.shuffle(indices)
        eligible_indices = []

        for idx in indices:
            if len(eligible_indices) == max_aug:
                break
            if random.uniform(0, 1) < p_aug and idx not in eligible_indices:
                eligible_indices.append(idx)

        if len(eligible_indices) < min_aug:
            n_left = min_aug - len(eligible_indices)
            random_idx = random.sample(indices, n_left)
            eligible_indices.extend(random_idx)

        random.shuffle(eligible_indices)
        return tokens, eligible_indices, exclude_map

    def _is_eligible_token(self, token):
        raise NotImplementedError

    @staticmethod
    def postprocess(text, exclude_map):
        if isinstance(text, str):
            text = [text]

        return [reconstruct(t, exclude_map) for t in text]

    @staticmethod
    def validate_action(action):
        raise NotImplementedError

    def transform(self, action, tokens, eligible_indices, **kwargs):
        raise NotImplementedError

    def augment(self,
                action,
                text,
                p_aug,
                min_aug,
                max_aug,
                exclude,
                is_segmented,
                segment,
                **kwargs):
        self.validate_action(action)

        tokens, eligible_indices, exclude_map = self._get_eligible_indices(text,
                                                                           p_aug=p_aug,
                                                                           min_aug=min_aug,
                                                                           max_aug=max_aug,
                                                                           exclude=exclude,
                                                                           is_segmented=is_segmented)

        transform_text = self.transform(action, tokens, eligible_indices, **kwargs)

        if segment:
            transform_text = [text_processor.process(t) for t in transform_text]

        transform_text = self.postprocess(transform_text, exclude_map)

        return transform_text
