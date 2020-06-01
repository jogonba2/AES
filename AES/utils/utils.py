import numpy as np
import itertools
from datetime import datetime

def get_candidates_batch(candidates, key):
    token_ids = []
    positions = []
    segments = []
    masks = []
    for i in candidates:
        token_ids.append(candidates[i]["token_ids"])
        positions.append(candidates[i]["positions"])
        segments.append(candidates[i]["segments"])
        masks.append(candidates[i]["masks"])

    return {key+"_token_ids": np.array(token_ids),
            key+"_positions": np.array(positions),
            key+"_segments": np.array(segments),
            key+"_masks": np.array(masks)}

def get_batch(buckets, key1, key2):
    token_ids = []
    positions = []
    segments = []
    masks = []
    for i, doc_index in enumerate(buckets):
        token_ids.append(buckets[doc_index][key1]["token_ids"])
        positions.append(buckets[doc_index][key1]["positions"])
        segments.append(buckets[doc_index][key1]["segments"])
        masks.append(buckets[doc_index][key1]["masks"])

    return {key2+"_token_ids": np.array(token_ids),
            key2+"_positions": np.array(positions),
            key2+"_segments": np.array(segments),
            key2+"_masks": np.array(masks)}

def check_valid_candidate(doc_sents, candidate_index, ngram_blocking):
    candidate = doc_sents[candidate_index].tolist()
    ngrams_bag = set()
    for sentence in candidate:
        ngrams = get_ngrams(sentence, ngram_blocking)
        if not ngrams_bag.intersection(ngrams):
            ngrams_bag = ngrams_bag.union(ngrams)
        else:
            return False
    return candidate

def check_ngram_blocking(candidate, ngram_blocking):
    ngrams_bag = set()
    for sentence in candidate:
        ngrams = get_ngrams(sentence, ngram_blocking)
        if not ngrams_bag.intersection(ngrams):
            ngrams_bag = ngrams_bag.union(ngrams)
        else:
            return True
    return False

def get_combinations(k_best_sentences):
    combinations = []
    for k in range(1, len(k_best_sentences) + 1):
        combinations += map(list, itertools.combinations(k_best_sentences, k))
    return combinations

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


def diff_hours(prev_time):
    act_time = datetime.utcnow()
    diff_hours = (act_time - prev_time).seconds / 3600
    if diff_hours >= 1:
        return True
    return False

def get_ngrams(sentence, n):
    ngrams = set()
    tok_sent = sentence.split()
    for i in range(0, len(tok_sent)-n):
        ngrams.add(tuple(tok_sent[i:i+n]))
    return ngrams