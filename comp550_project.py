from model import ModelFactory
from keras_extensions import TextSequence, ExperimentParameters, ExperimentData
from data_generation.pos_dicts import PosDictionary

import os
from keras.layers import Dense, Input, CuDNNLSTM, Dropout, SpatialDropout1D, Bidirectional, Embedding, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
import numpy as np
import os
import pandas as pd
from keras.callbacks import TensorBoard
import math
from tqdm import tqdm
import gensim
import spacy
from keras.utils import Sequence
from nltk import pos_tag
import math
from tensorflow import keras
import gensim.downloader as gensim_api
from keras.callbacks import ModelCheckpoint
imdb = keras.datasets.imdb


def train_dev_split(df_train, train_percent=0.9):
    nb_train = int(len(df_train) * train_percent)
    return df_train[:nb_train], df_train[nb_train:]


# df_train, df_dev = train_dev_split(df_train)

class ExperimentWrapper:

    DATA_DIRECTORY = os.path.join('drive', 'My Drive', 'Comp550data')

    def __init__(self):
        self.model_factory = ModelFactory()

    def run(self, train_data: ExperimentData, dev_data: ExperimentData,
              test_data: ExperimentData, params: ExperimentParameters):
        # train
        model = self.model_factory.create(params)
        model.summary()
        training_generator = TextSequence(train_data, params)
        validation_generator = TextSequence(dev_data, params)
        test_generator = TextSequence(test_data, params)

        print(params)

        # tensor_board = TensorBoard(os.path.join(ExperimentWrapper.DATA_DIRECTORY, 'logs', 'test'), histogram_freq=0)

        # tensorboard --logdir=./logs --port 6006
        # keras.backend.get_session().run(tf.global_variables_initializer())
        checkpointer = ModelCheckpoint(filepath="weights.hdf5", verbose=1, save_best_only=True)
        hist = model.fit_generator(training_generator, epochs=params.epochs,
                                   validation_data=validation_generator, verbose=2, callbacks=[checkpointer])
        model.load_weights('weights.hdf5')

        loss, acc = model.evaluate_generator(test_generator)
        # ypred = model.predict_generator(test_generator)
        print('Test accuracy = %f' % acc)
        # loss, acc = model.evaluate(x, y, verbose=0)
        model.save(os.path.join(ExperimentWrapper.DATA_DIRECTORY, 'models', params.file_name()))

        # # plot
        # # TODO: save output somehow?
        # history = pd.DataFrame(hist.history)
        # plt.figure(figsize=(12, 12));
        # plt.plot(history["loss"]);
        # plt.plot(history["val_loss"]);
        # plt.title("Loss with pretrained word vectors");
        # plt.show();
        #

    def evaluate_model(self, test_data_kwargs):
        eval_generator = TextSequence(**test_data_kwargs, **self.sequence_kwargs)
        # TODO: evaluate


# df_train = pd.read_pickle('nltk_pos_int_dataframe.pkl')
# df_train = pd.read_pickle('spacy_data_train.pkl')
# df_test = pd.read_pickle('spacy_data_test.pkl')

df_train = pd.read_pickle('df_train.pkl')
df_test = pd.read_pickle('df_test.pkl')

df_train, df_dev = train_dev_split(df_train, 0.9)

experiment_wrapper = ExperimentWrapper()
exp_params = ExperimentParameters(use_pos=True, epochs=30, pos_dict_len=PosDictionary.nltk_len)

# pos_x = np.array([np.squeeze(x) for x in df_train['pos']])

train_data = ExperimentData.from_df(df_train)
dev_data = ExperimentData.from_df(df_dev)
test_data = ExperimentData.from_df(df_test)


experiment_wrapper.run(train_data, dev_data, test_data, exp_params)

# training_generator = TextSequence(train_x, squeeze_pos_lookup, train_labels, 512)
# hist = model.fit_generator(training_generator, epochs=20)
