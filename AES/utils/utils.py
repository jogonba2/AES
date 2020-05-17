import numpy as np

def trigram_blocking():
    pass


def prepare_inputs(text_sents, tokenizer,
                   max_len_sent, max_sents):

    token_ids = []
    segments = []
    positions = []
    masks = []
    n_sents = len(text_sents)
    c_pos = 0

    for i in range(min(n_sents, max_sents)):
        tok_sent = tokenizer.tokenize(text_sents[i])

        # Fill input for CLS token #
        token_ids.append(tokenizer.convert_tokens_to_ids(tokenizer.cls_token))
        positions.append(c_pos)
        c_pos += 1
        segments.append(0 if i % 2 == 0 else 1)
        masks.append(1) # 1 for NON-MASKING, 0 for MASKING

        n_subwords_sent = len(tok_sent)

        for j in range(min(n_subwords_sent, max_len_sent)):
            # Fill input for sentence tokens #
            token_ids.append(tokenizer.convert_tokens_to_ids(tok_sent[j]))
            positions.append(c_pos)
            c_pos += 1
            segments.append(0 if i % 2 == 0 else 1)
            masks.append(1)

        # Pad until max_len_sent is reached #
        while j < max_len_sent - 1:
            # Fill input for pad tokens #
            token_ids.append(tokenizer.convert_tokens_to_ids(tokenizer.pad_token))
            positions.append(c_pos)
            c_pos += 1
            segments.append(0 if i % 2 == 0 else 1)
            masks.append(0)
            j += 1
        # Fill input for sep token #
        token_ids.append(tokenizer.convert_tokens_to_ids(tokenizer.sep_token))
        positions.append(c_pos)
        c_pos += 1
        segments.append(0 if i % 2 == 0 else 1)
        masks.append(1)

    # Pad until max_sents is reached #
    while i < max_sents - 1:
        # Fill input for cls token in padded sentence #
        token_ids.append(tokenizer.convert_tokens_to_ids(tokenizer.cls_token))
        positions.append(c_pos)
        c_pos += 1
        segments.append(0 if i % 2 == 0 else 1)
        masks.append(0) # Enmascarar los SEP (cuidado porque igual puede dar problemas)

        for j in range(max_len_sent):
            # Fill input for pad tokens in padded sentence #
            token_ids.append(tokenizer.convert_tokens_to_ids(tokenizer.pad_token))
            positions.append(c_pos)
            c_pos += 1
            segments.append(0 if i % 2 == 0 else 1)
            masks.append(0)

        # Fill input for sep token in padded sentence #
        token_ids.append(tokenizer.convert_tokens_to_ids(tokenizer.sep_token))
        positions.append(c_pos)
        c_pos += 1
        segments.append(0 if i % 2 == 0 else 1)
        masks.append(0)

        i += 1

    return (np.array(token_ids, dtype="int32"),
            np.array(positions, dtype="int32"),
            np.array(segments, dtype="int32"),
            np.array(masks, dtype="int32"))


def preprocess_text(text, sent_split):
    text = text.strip()
    text_sents = [line for line in text.split(sent_split) if line!=""]
    if len(text_sents) == 0:
        return None
    return text_sents