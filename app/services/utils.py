import re


TOKENIZER_REGEX = re.compile(r'(\W)')


def tokenize(text: str):
    tokens = TOKENIZER_REGEX.split(text)
    return [t for t in tokens if len(t.strip()) > 0]


def mask_exclude_tokens(text, exclude):
    exclude_map = {}
    count = 0
    mask = "MASK{}"

    for token in exclude:
        tmp_mask = mask.format(count)
        
        # token = re.sub(r'([()+.?!*])', r'\\\1', token)
        exclude_map[tmp_mask] = token
        text = text.replace(token, tmp_mask)
        # text = re.sub(token, tmp_mask, text, flags=re.IGNORECASE)
        count += 1
    return text, exclude_map


def reconstruct(text, exclude_map):
    for key, value in exclude_map.items():
        # text = re.sub(key, value, text, flags=re.IGNORECASE)
        text = text.replace(key, value)
    return text


def revert_segmented_tokens(tokens):
    results = []
    for idx in range(len(tokens)):
        token = tokens[idx]
        if "_" in token and len(token) > 1:
            token = tokens[idx].split("_")
            results.extend(token)
        else:
            results.append(token)

    return results
