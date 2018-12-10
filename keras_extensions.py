import numpy as np
from keras.utils import Sequence
import math
import gensim.downloader as gensim_api
import math
import numpy as np
from keras.utils import Sequence
from data import ExperimentParameters, ExperimentData
import os
import pickle


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


class TextSequence(Sequence):

    def __init__(self, data: ExperimentData, params: ExperimentParameters):
        self.data = data
        self.params = params
        self.wv_model = self.build_wv_model(self.params.wv_type)

    @staticmethod
    def build_wv_model(wv_type):
        if wv_type == 'gensim-glove-100':
            return EmbeddingsCache.get_glove_100_model()

    def __len__(self):
        return math.ceil(len(self.data.x) / self.params.batch_size)

    def __getitem__(self, idx):
        # build batches
        batch_start = idx * self.params.batch_size
        batch_end = batch_start + self.params.batch_size
        batch_x = self.data.x[batch_start:batch_end]
        batch_y = self.data.y[batch_start:batch_end]

        # process batches
        processed_batch_x = np.array([self.x_process(x) for x in batch_x])
        processed_batch_inputs = [processed_batch_x]
        processed_batch_y = np.array([self.y_process(y) for y in batch_y])

        if self.params.use_pos:
            batch_pos = self.data.x_pos[batch_start:batch_end]
            processed_batch_pos = np.array([self.pos_process(pos) for pos in batch_pos])
            processed_batch_inputs.append(processed_batch_pos)
        elif self.params.use_parse:
            batch_parse = self.data.x_parse[batch_start:batch_end]
            processed_batch_parse = np.array([self.parse_process(parse) for parse in batch_parse])
            processed_batch_inputs.append(processed_batch_parse)

        return processed_batch_inputs, processed_batch_y

    def x_process(self, text):
        wv_tensor = np.zeros((self.params.sent_dim,
                              self.params.wv_dim))  # (np.random.random((self.params.sent_dim, self.params.wv_dim)) - 0.5) / 5
        oov_count = 0
        for i, w in zip(range(self.params.sent_dim), text.split()):
            try:
                wv_tensor[i] = self.wv_model[w]
            except:
                oov_count += 1
                pass
        # print("OOV Count: ", oov_count)
        return wv_tensor

    def y_process(self, y):
        return y

    def pos_process(self, sentence_pos_tags):
        pos_tensor = (self.params.pos_dict_len - 1) * np.ones(self.params.sent_dim)
        valid_pos_len = min(len(sentence_pos_tags), self.params.sent_dim)
        pos_tensor[:valid_pos_len] = sentence_pos_tags[:valid_pos_len]
        return pos_tensor

    def parse_process(self, parse):
        # TODO: process
        return parse
