from keras.layers import Dense, Input, CuDNNLSTM, Dropout, SpatialDropout1D, Bidirectional, Embedding, \
    Concatenate, Lambda, Convolution1D, MaxPooling1D, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras_extensions import ExperimentParameters
from keras import backend as K
from keras.datasets import imdb
import numpy as np
from keras.engine.topology import Layer


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
        pretrained_wv = 0.1 * np.ones((len(word_index), params.wv_dim))
        for word, index in word_index.items():
            pretrained_wv[index] = word_index[word]

        embedding_layer = Embedding(len(word_index), params.wv_dim, input_length=params.sent_dim,
                                    embeddings_initializer='glorot_normal',
                                    name='WordEmbeddings')(wi_input)
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
                                    output_shape=(params.sent_dim, filter_data_dim))([filter_mat_input, input_layer])

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

        # model = Model(input_layer, model_output)
        model = Model(inputs=inputs, outputs=model_output)
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model

    def create(self, params: ExperimentParameters):

        pos_input_func = None
        if params.use_pos == 'embed':
            pos_input_func = self.pos_input_tensor
        elif params.use_pos == 'one_hot':
            pos_input_func = self.pos_one_hot_input_tensor

        wv_input_func = self.word_index_input_tensor if params.use_word_index else self.input_tensor

        if params.nn_model == 'cnn':
            return self.create_cnn_model(params, wv_input_func, pos_input_func)
        else:
            return self.create_lstm_model(params, wv_input_func, pos_input_func)


if __name__ == '__main__':
    mf = ModelFactory()
    model = mf.create(ExperimentParameters(nn_model='cnn', use_parse='concat', use_pos='embed', use_word_index=True))
    model.summary()
