"""
Comp 550 - Final Project - Fall 2018
Augmenting Word Embeddings using Additional Linguistic Information
Group 1 - Andrei Mircea (260585208) - Stefan Wapnick (id 260461342)

Github:         https://github.com/amr-amr/COMP550-Project
Data folder:    https://drive.google.com/drive/folders/1Z0YrLC8KX81HgDlpj1OB4bCM6VGoAXmE?usp=sharing

Script Description:

"""
import os
import pickle
from keras.datasets import imdb
from nltk import load
from constants import DATA_DIRECTORY
from helpers import save_pickle, load_pickle
import gensim.downloader as gensim_api
import numpy as np


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
    _word_index = None
    _word_index_file = 'word_index.pkl'

    @staticmethod
    def get_word_index(fail_if_missing=True):
        if WordIndexCache._word_index is not None:
            return WordIndexCache._word_index

        cache_path = os.path.join(DATA_DIRECTORY, WordIndexCache._word_index_file)
        WordIndexCache._word_index = load_pickle(cache_path)
        if WordIndexCache._word_index is None and fail_if_missing:
            raise Exception('Word index not initialized')

        return WordIndexCache._word_index

    @staticmethod
    def is_initialized():
        return WordIndexCache.get_word_index(fail_if_missing=False) is not None

    @staticmethod
    def initialize(text):

        word_index = {}
        word_index["<PAD>"] = 0
        word_index["<OOV>"] = 1

        i = 2
        for token_list in text:
            for token in token_list:
                if token not in word_index:
                    word_index[token] = i
                    i += 1

        WordIndexCache._word_index = word_index
        save_pickle(os.path.join(DATA_DIRECTORY, WordIndexCache._word_index_file), WordIndexCache._word_index)
        return WordIndexCache._word_index


class EmbeddingsCache:
    _embedding_file = 'wv_embeddings.plk'
    _embeddings_model = None

    _init_embeddings_cache = {}

    @staticmethod
    def get_wv_embeddings():

        if EmbeddingsCache._embeddings_model is not None:
            return EmbeddingsCache._embeddings_model

        embeddings_path = os.path.join(DATA_DIRECTORY, EmbeddingsCache._embedding_file)
        EmbeddingsCache._embeddings_model = load_pickle(embeddings_path)

        if EmbeddingsCache._embeddings_model is None:
            print('Downloading glove embeddings...')
            EmbeddingsCache._embeddings_model = gensim_api.load('glove-wiki-gigaword-100')

        save_pickle(embeddings_path, EmbeddingsCache._embeddings_model)
        return EmbeddingsCache._embeddings_model
