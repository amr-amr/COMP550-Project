from nltk import load


class PosDictionary:
    spacy = {'ADJ': 0,
             'ADP': 1,
             'ADV': 2,
             'AUX': 3,
             'CONJ': 4,
             'CCONJ': 4,
             'DET': 5,
             'INTJ': 6,
             'NOUN': 7,
             'NUM': 8,
             'PART': 9,
             'PRON': 10,
             'PROPN': 11,
             'PUNCT': 12,
             'SCONJ': 13,
             'SYM': 14,
             'VERB': 15,
             'X': 16}

    # -1 since 4 encoded twice
    spacy_len = len(spacy) - 1

    nltk = {key: i for (i, key) in enumerate(load('help/tagsets/upenn_tagset.pickle').keys())}
    nltk_len = len(nltk)
