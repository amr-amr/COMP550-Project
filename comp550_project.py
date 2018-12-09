import os
from keras.layers import Dense, Input, CuDNNLSTM, Dropout, SpatialDropout1D, Bidirectional, Embedding, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
import numpy as np
import os
import pandas as pd
import math
from tqdm import tqdm
import gensim
import spacy
from keras.utils import Sequence
from nltk import pos_tag
import math
from tensorflow import keras
import gensim.downloader as gensim_api
imdb = keras.datasets.imdb
from model import ModelFactory
from keras_modules import TextSequence, ExperimentParameters, ExperimentData

# (train_x, train_labels), (test_x, test_labels) = load_data()


def train_dev_split(df_train, train_percent=0.9):
    nb_train = int(len(df_train) * train_percent)
    return df_train[:nb_train], df_train[nb_train:]


# df_train, df_dev = train_dev_split(df_train)

class ExperimentWrapper:
    # TODO: define kwargs
    def __init__(self, model_filepath):
        self.model_factory = ModelFactory()
        self.model_filepath = model_filepath

    def train(self, data: ExperimentData, params: ExperimentParameters):
        # train
        model = self.model_factory.create(params)
        model.summary()
        training_generator = TextSequence(data, params)
        hist = model.fit_generator(training_generator, params.epochs)

        # # plot
        # # TODO: save output somehow?
        # history = pd.DataFrame(hist.history)
        # plt.figure(figsize=(12, 12));
        # plt.plot(history["loss"]);
        # plt.plot(history["val_loss"]);
        # plt.title("Loss with pretrained word vectors");
        # plt.show();
        #
        # # save
        # # TODO: check if filepath exists
        # self.model.save(self.model_filepath)

    def evaluate_model(self, test_data_kwargs):
        eval_generator = TextSequence(**test_data_kwargs, **self.sequence_kwargs)
        # TODO: evaluate


df_train = pd.read_pickle('spacy_data_train.pkl')
df_test = pd.read_pickle('spacy_data_test.pkl')
experiment_wrapper = ExperimentWrapper()
experiment_wrapper.train()

# training_generator = TextSequence(train_x, squeeze_pos_lookup, train_labels, 512)
# hist = model.fit_generator(training_generator, epochs=20)