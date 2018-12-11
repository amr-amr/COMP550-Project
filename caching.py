import os
import pickle
import gensim.downloader as gensim_api
from keras.datasets import imdb


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
    glove_100_model = None

    @staticmethod
    def get_glove_100_model():
        if os.path.isfile('glove-100.plk'):
            return pickle.load(open('glove-100.plk', 'rb'))

        embeddings_model = gensim_api.load('glove-wiki-gigaword-100')
        with open('glove-100.plk', 'wb') as f:
            pickle.dump(embeddings_model, f)

        return embeddings_model
