import tensorflow as tf
import numpy as np

import os
import pandas as pd
import math
from tqdm import tqdm

import gensim
import spacy
import re

import matplotlib.pyplot as plt
from keras.layers import Dense, Input, CuDNNLSTM, Dropout, SpatialDropout1D, Bidirectional, Embedding, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization


class ModelFactory:

    def _bilstm_pos_input_tensor(self, sent_dim, wv_dim, pos_dim):

        wv_input = Input(shape=(sent_dim, wv_dim), name='wv_nput')
        pos_input = Input(shape=(sent_dim,), name='pos_input')

        embedding_layer = Embedding(pos_dim, pos_dim, input_length=sent_dim,
                                    embeddings_initializer='glorot_normal',
                                    name='POSEmbeddings')(pos_input)
        concatenate_layer = Concatenate(axis=2,
                                        name='wv_pos_concatenate')([wv_input, embedding_layer])

        return concatenate_layer, [wv_input, pos_input]

    def _bilstm_input_tensor(self, word_dim, wv_dim, pos_dim):
        input_layer = Input(shape=(word_dim, wv_dim + pos_dim), name='input')
        return input_layer, input_layer

    def _create_bilstm_model(self, sent_dim, wv_dim, pos_dim, input_func):

        input_layers, inputs = input_func(sent_dim, wv_dim, pos_dim)

        embedded_sequences = SpatialDropout1D(0.1)(input_layers)
        x = Bidirectional(CuDNNLSTM(64, return_sequences=False))(embedded_sequences)
        x = Dropout(0.1)(x)
        x = BatchNormalization()(x)
        preds = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=preds)

        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=0.001, clipnorm=.25, beta_1=0.7, beta_2=0.99),
                      metrics=['accuracy'])

        return model

    def create(self, model_type, sent_dim, wv_dim, pos_dim):

        if model_type == 'bilstm':
            return self._create_bilstm_model(sent_dim, wv_dim, pos_dim, self._bilstm_input_tensor)
        if model_type == 'bilstm_pos_input':
            return self._create_bilstm_model(sent_dim, wv_dim, pos_dim, self._bilstm_pos_input_tensor)

        raise Exception('Unknown model %s' % model_type)


mf = ModelFactory()
model = mf.create('bilstm_pos_input', 200, 100, 16)
