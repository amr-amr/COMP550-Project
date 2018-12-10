from keras.layers import Dense, Input, CuDNNLSTM, Dropout, SpatialDropout1D, Bidirectional, Embedding, \
    Concatenate, Lambda, Convolution1D, MaxPooling1D, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras_extensions import ExperimentParameters
from keras import backend as K


class ModelFactory:

    @staticmethod
    def pos_input_tensor(params: ExperimentParameters):

        wv_input = Input(shape=(params.sent_dim, params.wv_dim), name='wv_input')
        pos_input = Input(shape=(params.sent_dim,), name='pos_input')

        embedding_layer = Embedding(params.pos_dict_len, params.pos_dim, input_length=params.sent_dim,
                                    embeddings_initializer='glorot_normal',
                                    name='POSEmbeddings')(pos_input)
        concatenate_layer = Concatenate(axis=2,
                                        name='wv_pos_concatenate')([wv_input, embedding_layer])

        return concatenate_layer, [wv_input, pos_input]

    @staticmethod
    def pos_one_hot_input_tensor(params: ExperimentParameters):

        wv_input = Input(shape=(params.sent_dim, params.wv_dim), name='wv_input')
        pos_input = Input(shape=(params.sent_dim,), dtype='uint8', name='pos_input')

        one_hot_layer = Lambda(K.one_hot,
                               arguments={'num_classes': params.pos_dim},
                               output_shape=(params.sent_dim, params.pos_dim))(pos_input)

        concatenate_layer = Concatenate(axis=2,
                                        name='wv_pos_concatenate')([wv_input, one_hot_layer])

        return concatenate_layer, [wv_input, pos_input]

    @staticmethod
    def input_tensor(params: ExperimentParameters):
        input_layer = Input(shape=(params.sent_dim, params.wv_dim), name='input')
        return input_layer, input_layer

    @staticmethod
    def create_lstm_model(params: ExperimentParameters, input_func):

        input_layer, inputs = input_func(params)

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
    def create_cnn_model(params: ExperimentParameters, input_func):

        # input_layer = Input(shape=(params.sent_dim, params.wv_dim), name='input')
        input_layer, inputs = input_func(params)

        filter_sizes = (3, 8)
        num_filters = 10
        hidden_dims = 50
        z = Dropout(0.5)(input_layer)

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

        z = Dropout(0.8)(z)
        z = Dense(hidden_dims, activation="relu")(z)
        model_output = Dense(1, activation="sigmoid")(z)

        # model = Model(input_layer, model_output)
        model = Model(inputs=inputs, outputs=model_output)
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
        return model

    def create(self, params: ExperimentParameters):

        if params.use_pos == 'embed':
            input_layer_func = self.pos_input_tensor
        else:
            input_layer_func = self.pos_one_hot_input_tensor if params.use_pos == 'one_hot' else self.input_tensor

        if params.nn_model == 'cnn':
            return self.create_cnn_model(params, input_layer_func)
        else:
            return self.create_lstm_model(params, input_layer_func)


if __name__ == '__main__':
    mf = ModelFactory()
    model = mf.create(ExperimentParameters(nn_model='cnn', use_pos='embed'))
    model.summary()
