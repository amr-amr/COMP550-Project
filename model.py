from keras.layers import Dense, Input, CuDNNLSTM, Dropout, SpatialDropout1D, Bidirectional, Embedding, \
    Concatenate, Lambda, Convolution1D, MaxPooling1D, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from caching import WordIndexCache
from keras import backend as K
import numpy as np
from keras.utils import Sequence
from keras.preprocessing import sequence
import math
from dtos import ExperimentParameters, ExperimentData


class TextSequence(Sequence):

    def __init__(self, data: ExperimentData, params: ExperimentParameters):
        self.data = data
        self.params = params
        self.wv_model = EmbeddingsCache.get_embedding()
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
        processed_batch_x = np.array([self.text_process(x) for x in batch_x])
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

    def text_process(self, text):
        wi_tensor = self.word_index["<PAD>"] * np.ones(self.params.sent_dim)
        for i, w in zip(range(self.params.sent_dim), text):
            wi_tensor[i] = (self.word_index[w])
        return wi_tensor

    def parse_process(self, parse):
        parse_tensor = np.eye(self.params.sent_dim)
        for i, dep in zip(range(self.params.sent_dim), parse):
            j = dep[1]  # head of word at index i
            if j < 200 and i < 200 and j != i:
                parse_tensor[i][j] = -1
        return parse_tensor


class ModelFactory:

    @staticmethod
    def pos_input_tensor(params: ExperimentParameters, wv_input_func):

        wv_input_layer, wv_input = wv_input_func(params)
        pos_input = Input(shape=(params.sent_dim,), name='pos_input')

        embedding_layer = Embedding(params.pos_dict_len, params.pos_dim, input_length=params.sent_dim,
                                    embeddings_initializer='glorot_normal',
                                    name='POSEmbeddings')(pos_input)
        concatenate_layer = Concatenate(axis=2,
                                        name='wv_pos_concatenate')([wv_input_layer, embedding_layer])

        return concatenate_layer, [wv_input, pos_input]

    @staticmethod
    def pos_one_hot_input_tensor(params: ExperimentParameters, wv_input_func):

        wv_input_layer, wv_input = wv_input_func(params)
        pos_input = Input(shape=(params.sent_dim,), dtype='uint8', name='pos_input')

        one_hot_layer = Lambda(K.one_hot,
                               arguments={'num_classes': params.pos_dim},
                               output_shape=(params.sent_dim, params.pos_dim))(pos_input)

        concatenate_layer = Concatenate(axis=2,
                                        name='wv_pos_concatenate')([wv_input_layer, one_hot_layer])

        return concatenate_layer, [wv_input, pos_input]

    @staticmethod
    def word_index_input_tensor(params: ExperimentParameters):
        wi_input = Input(shape=(params.sent_dim,), name='word_index_input')
        word_index = WordIndexCache.get_word_index()
        wv_cache = EmbeddingsCache.get_glove_100_model()
        pretrained_wv = 0.1 * np.ones((len(word_index), params.wv_dim))
        for word, index in word_index.items():
            try:
                pretrained_wv[index] = wv_cache[word]
            except:
                pretrained_wv[index] = np.random.random(params.wv_dim)

        embedding_layer = Embedding(len(word_index), params.wv_dim, input_length=params.sent_dim,
                                    embeddings_initializer='glorot_normal', weights=[pretrained_wv],
                                    trainable=params.train_wv, name='WordEmbeddings')(wi_input)
        return embedding_layer, wi_input

    @staticmethod
    def input_tensor(params: ExperimentParameters):
        input_layer = Input(shape=(params.sent_dim, params.wv_dim), name='input')
        return input_layer, input_layer

    @staticmethod
    def create_lstm_model(params: ExperimentParameters, wv_input_func, pos_input_func):

        input_layer, inputs = pos_input_func(params, wv_input_func) if pos_input_func else wv_input_func(params)

        if params.use_parse:
            input_layer, inputs = ModelFactory.create_parse_filter_layer(params, input_layer, inputs)

        embedded_sequences = SpatialDropout1D(params.dropout)(input_layer)
        x = Bidirectional(CuDNNLSTM(64, return_sequences=False))(embedded_sequences)
        x = Dropout(params.dropout)(x)
        x = BatchNormalization()(x)
        preds = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=preds)

        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=0.001, clipnorm=.25, beta_1=0.7, beta_2=0.99),
                      metrics=['accuracy'])

        return model

    @staticmethod
    def create_parse_filter_layer(params: ExperimentParameters, input_layer, inputs):
        filter_mat_input = Input(shape=(params.sent_dim, params.sent_dim), name='filter_input')
        filter_data_dim = params.wv_dim + (params.pos_dim if params.use_pos else 0)
        parse_output_layer = Lambda(lambda x: K.batch_dot(x[0], x[1]),
                                    output_shape=(params.sent_dim, filter_data_dim),
                                    name='parse_layer')([filter_mat_input, input_layer])

        if params.use_parse == 'concat':
            parse_output_layer = Concatenate(axis=2,
                                             name='parse_wv_concatenate')([input_layer, parse_output_layer])

        if isinstance(inputs, list):
            inputs.append(filter_mat_input)
            return parse_output_layer, inputs

        return parse_output_layer, [inputs, filter_mat_input]

    @staticmethod
    def create_cnn_model(params: ExperimentParameters, wv_input_func, pos_input_func):

        # input_layer = Input(shape=(params.sent_dim, params.wv_dim), name='input')
        input_layer, inputs = pos_input_func(params, wv_input_func) if pos_input_func else wv_input_func(params)

        if params.use_parse:
            input_layer, inputs = ModelFactory.create_parse_filter_layer(params, input_layer, inputs)

        filter_sizes = (3, 8)
        num_filters = 10
        hidden_dims = 50
        z = Dropout(params.dropout, name='dropout_input_%.2f' % params.dropout)(input_layer)

        conv_blocks = []
        for sz in filter_sizes:
            conv = Convolution1D(filters=num_filters,
                                 kernel_size=sz,
                                 padding="valid",
                                 activation="relu",
                                 strides=1)(z)
            conv = MaxPooling1D(pool_size=2)(conv)
            conv = Flatten()(conv)
            conv_blocks.append(conv)
        z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]

        z = Dropout(params.dropout, name='dropout_pred_%.2f' % params.dropout)(z)
        z = Dense(hidden_dims, activation="relu")(z)
        model_output = Dense(1, activation="sigmoid")(z)

        model = Model(inputs=inputs, outputs=model_output)
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model

    def create(self, params: ExperimentParameters):

        pos_input_func = None
        if params.use_pos == 'embed':
            pos_input_func = self.pos_input_tensor
        elif params.use_pos == 'one_hot':
            pos_input_func = self.pos_one_hot_input_tensor

        wv_input_func = self.word_index_input_tensor  # if params.train_wv else self.input_tensor

        if params.nn_model == 'cnn':
            return self.create_cnn_model(params, wv_input_func, pos_input_func)
        else:
            return self.create_lstm_model(params, wv_input_func, pos_input_func)


if __name__ == '__main__':
    mf = ModelFactory()
    model = mf.create(ExperimentParameters(nn_model='cnn', train_wv=True))
    model.summary()
