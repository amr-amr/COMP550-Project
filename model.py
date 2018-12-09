from keras.layers import Dense, Input, CuDNNLSTM, Dropout, SpatialDropout1D, Bidirectional, Embedding, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras_modules import ExperimentParameters


class ModelFactory:

    @staticmethod
    def lstm_pos_input_tensor(params: ExperimentParameters):

        wv_input = Input(shape=(params.sent_dim, params.wv_dim), name='wv_nput')
        pos_input = Input(shape=(params.sent_dim,), name='pos_input')

        embedding_layer = Embedding(params.pos_dim, params.pos_dim, input_length=params.sent_dim,
                                    embeddings_initializer='glorot_normal',
                                    name='POSEmbeddings')(pos_input)
        concatenate_layer = Concatenate(axis=2,
                                        name='wv_pos_concatenate')([wv_input, embedding_layer])

        return concatenate_layer, [wv_input, pos_input]

    @staticmethod
    def lstm_input_tensor(params: ExperimentParameters):
        input_layer = Input(shape=(params.sent_dim, params.wv_dim + params.pos_dim), name='input')
        return input_layer, input_layer

    @staticmethod
    def create_lstm_model(params: ExperimentParameters, input_func):

        input_layers, inputs = input_func(params)

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

    def create(self, params: ExperimentParameters):

        if params.use_pos:
            return self.create_lstm_model(params, self.lstm_pos_input_tensor)
        else:
            return self.create_lstm_model(params, self.lstm_input_tensor)


if __name__ == '__main__':
    mf = ModelFactory()
    model = mf.create(ExperimentParameters())
    model.summary()
