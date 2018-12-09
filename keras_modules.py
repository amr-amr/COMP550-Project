import numpy as np
from keras.utils import Sequence
import math
import gensim.downloader as gensim_api
import math
import numpy as np
from keras.utils import Sequence


class ExperimentData:

    def __init__(self, x, x_pos, x_parse, y):
        self.y = y
        self.x = x
        self.x_pos = x_pos
        self.x_parse = x_parse


class ExperimentParameters:

    def __init__(self, batch_size=512, wv_type='gensim-glove-100',
                 use_pos=False, use_parse=False, sent_dim=200, wv_dim=100,
                 pos_dim = 16, epochs=20):
        self.batch_size = batch_size
        self.wv_type = wv_type
        self.use_pos = use_pos
        self.use_parse = use_parse
        self.sent_dim = sent_dim
        self.wv_dim = wv_dim
        self.pos_dim = pos_dim
        self.epochs = epochs

    def __str__(self) -> str:
        return "batch_size=%s, wv_type=%s, use_pos=%s, use_parse=%s, sent_dim=%d, wv_dim=%d, pos_dim=%d, epochs=%d" \
               % (self.batch_size, self.wv_type, self.use_pos, self.use_parse, self.sent_dim,
                  self.wv_dim, self.pos_dim, self.epochs)


class TextSequence(Sequence):

    def __init__(self, data: ExperimentData, params: ExperimentParameters):
        self.data = data
        self.params = params
        self.wv_model = self.build_wv_model(self.params.wv_type)

    @staticmethod
    def build_wv_model(wv_type):
        if wv_type == 'gensim-glove-100':
            return gensim_api.load("glove-wiki-gigaword-100")

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
        processed_batch_inputs = [processed_batch_x]  # TODO: ensure you can return list of 1
        processed_batch_y = np.array([self.y_process(y) for y in batch_y])

        if self.params.use_pos:
            batch_pos = self.data.x_pos[batch_start:batch_end]
            processed_batch_pos = np.array([self.pos_process(pos) for pos in batch_pos])
            processed_batch_inputs = processed_batch_inputs.append(processed_batch_pos)
        elif self.params.use_parse:
            batch_parse = self.data.x_parse[batch_start:batch_end]
            processed_batch_parse = np.array([self.parse_process(parse) for parse in batch_parse])
            processed_batch_inputs = processed_batch_inputs.append(processed_batch_parse)

        return processed_batch_inputs, processed_batch_y

    def x_process(self, text):
        wv_tensor = (np.random.random((self.params.sent_dim, self.params.wv_dim)) - 0.5) / 5
        oov_count = 0
        for i, w in zip(range(self.params.sent_dim), text.split()):
            try:
                wv_tensor[i] = self.wv_model[w]
            except:
                oov_count += 1
                pass
        print("OOV Count: ", oov_count)
        return wv_tensor

    def y_process(self, y):
        return y

    def pos_process(self, pos_tags):
        # TODO: check if processing necessary
        return pos_tags

    def parse_process(self, parse):
        # TODO: process
        return parse
