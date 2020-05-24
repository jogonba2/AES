from AES.utils import utils


def selection_summary_ngram_blocking(scores, doc_sents,
                                     max_sents_doc, k,
                                     block_ngrams=3):
    n_sents = len(doc_sents)
    if n_sents < max_sents_doc:
        scores = scores[:n_sents]

    gen_summary_sents = []
    rank = scores.argsort()[::-1]
    if n_sents < k:
        gen_summary_sents = doc_sents

    else:
        ngrams_bag = set()
        considered_sents = 0
        for pos in rank:
            sentence = doc_sents[pos]
            ngrams = utils.get_ngrams(sentence, block_ngrams)

            if not ngrams_bag.intersection(ngrams):
                considered_sents += 1
                gen_summary_sents.append(sentence)
                ngrams_bag = ngrams_bag.union(ngrams)

            if considered_sents == k:
                break

    return gen_summary_sents

def selection_summary(scores, doc_sents,
                      max_sents_doc, k):
    n_sents = len(doc_sents)
    if n_sents < max_sents_doc:
        scores[n_sents:] = 0

    gen_summary_sents = []
    selecteds = scores.argsort()[-k:]
    selecteds.sort() # Doc order

    if n_sents < k:
        gen_summary_sents = doc_sents
    else:
        for s in selecteds:
            gen_summary_sents.append(doc_sents[s])

    return gen_summary_sents