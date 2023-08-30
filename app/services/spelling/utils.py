def reverse_tokenizer(tokens, gaps):
    assert 0 <= len(gaps) - len(tokens) <= 1
    if len(gaps) > len(tokens):
        tokens.append('')
    ans = ''
    for t, g in zip(tokens, gaps):
        ans = ans + g + t
    return ans


def find_all_gaps(text, tokens):
    assert text.startswith(tokens[0])
    gaps = []
    gap = ''
    while text or tokens:
        if not tokens:
            gap = text
            gaps.append(gap)
            break
        if text.startswith(tokens[0]):
            text = text[len(tokens[0]):]
            tokens = tokens[1:]
            gaps.append(gap)
            gap = ''
        else:
            gap += text[0]
            text = text[1:]
    return gaps


def is_valid_text(text):
    return text.strip()
