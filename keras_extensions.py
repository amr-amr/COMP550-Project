import math
import numpy as np
from keras.utils import Sequence

from caching import EmbeddingsCache, WordIndexCache
from data import ExperimentParameters, ExperimentData
from keras.preprocessing import sequence


class TextSequence(Sequence):

    def __init__(self, data: ExperimentData, params: ExperimentParameters):
        self.data = data
        self.params = params
        self.wv_model = EmbeddingsCache.get_glove_100_model()
        self.ovv_count = 0
        self.word_index = WordIndexCache.get_word_index()

    def __len__(self):
        return math.ceil(len(self.data.x) / self.params.batch_size)

    def __getitem__(self, idx):
        # build batches
        batch_start = idx * self.params.batch_size
        batch_end = batch_start + self.params.batch_size
        batch_x = self.data.x[batch_start:batch_end]
        batch_y = self.data.y[batch_start:batch_end]

        # process batches
        if self.params.train_wv:
            processed_batch_x = np.array([self.word_index_process(x) for x in batch_x])
        else:
            processed_batch_x = np.array([self.x_process(x) for x in batch_x])

        processed_batch_inputs = [processed_batch_x]

        if self.params.use_pos:
            batch_pos = self.data.x_pos[batch_start:batch_end]
            processed_batch_pos = sequence.pad_sequences(batch_pos, self.params.sent_dim, padding='post',
                                                         truncating='post', value=(self.params.pos_dict_len - 1))
            processed_batch_inputs.append(processed_batch_pos)
        if self.params.use_parse:
            batch_parse = self.data.x_parse[batch_start:batch_end]
            processed_batch_parse = np.array([self.parse_process(parse) for parse in batch_parse])
            processed_batch_inputs.append(processed_batch_parse)

        return processed_batch_inputs, batch_y

    def x_process(self, text):
        wv_tensor = np.zeros((self.params.sent_dim,
                              self.params.wv_dim))  # (np.random.random((self.params.sent_dim, self.params.wv_dim)) - 0.5) / 5
        for i, w in zip(range(self.params.sent_dim), text.split()):
            try:
                wv_tensor[i] = self.wv_model[w]
            except:
                self.ovv_count += 1
                pass
        # print("OOV Count: ", oov_count)
        return wv_tensor

    def word_index_process(self, text):
        wi_tensor = self.word_index["<PAD>"] * np.ones(self.params.sent_dim)
        for i, w in zip(range(self.params.sent_dim), text.split()):
            try:
                wi_tensor[i] = (self.word_index[w])
            except:
                wi_tensor[i] = (self.word_index["<UNK>"])
        return wi_tensor

    ## TODO: Format
    def parse_process(self, parse):
        parse_tensor = np.eye(self.params.sent_dim)
        for i, dep in zip(range(self.params.sent_dim), parse):
            j = dep[1]  # head of word at index i
            if j < 200 and i < 200 and j != i:
                parse_tensor[i][j] = -1
        return parse_tensor
