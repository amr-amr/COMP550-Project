"""
Comp 550 - Final Project - Augmenting Word Embeddings using Additional Linguistic Information
Group 1 - Andrei Mircea (260585208) - Stefan Wapnick (id 260461342)

Script Description:

"""
import os
import pickle
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from nltk import load
from constants import DATA_DIRECTORY


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


class WordIndexCache:
    word_index = None

    @staticmethod
    def get_word_index():
        if WordIndexCache.word_index is None:
            word_index = imdb.get_word_index()
            word_index = {k: (v + 3) for k, v in word_index.items()}
            word_index["<PAD>"] = 0
            word_index["<START>"] = 1
            word_index["<UNK>"] = 2  # unknown
            word_index["<UNUSED>"] = 3

            WordIndexCache.word_index = word_index

        return WordIndexCache.word_index


class EmbeddingsCache:
    embedding_model = None
    embedding_path = 'embeddings.plk'

    @staticmethod
    def get_embedding():
        if EmbeddingsCache.embedding_model is not None:
            return EmbeddingsCache.embedding_model

        embedding_path = os.path.join(DATA_DIRECTORY, EmbeddingsCache.embedding_path)
        if os.path.isfile(embedding_path):
            return pickle.load(open(embedding_path, 'rb'))

        raise Exception('Missing word vector embeddings file %s' % embedding_path)

    @staticmethod
    def initialize(train_text, test_text):
        all_text = train_text + test_text
        tokenizer = Tokenizer(lower=False, oov_token='<OOV>')
        tokenizer.fit_on_texts(all_text)
        EmbeddingsCache.embedding_model = tokenizer

        with open(os.path.join(DATA_DIRECTORY, EmbeddingsCache.embedding_path), 'wb') as f:
            pickle.dump(EmbeddingsCache.embedding_model, f)

        return EmbeddingsCache.embedding_model
