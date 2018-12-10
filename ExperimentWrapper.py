from keras.utils import Sequence
import math
import gensim.downloader as gensim_api


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

class TextSequence(Sequence):

    def __init__(self, x_set, y_set, batch_size,
                 pretrained_wv='gensim-glove-100',
                 use_pos_tags=False, use_pos_embed=False, pos_tags=[],
                 use_parse=False, use_parse_embed=False, parse=[],
                 MAX_SENT_LEN=250, WV_DIM=100):

        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.pretrained_wv = pretrained_wv
        self.pretrained_wv_model = self.build_model()
        self.use_pos_tags = use_pos_tags
        self.use_pos_embed = use_pos_embed
        self.pos_tags = pos_tags
        self.use_parse = use_parse
        self.use_parse_embed = use_parse_embed
        self.parse = parse
        self.MAX_SENT_LEN = MAX_SENT_LEN
        self.WV_DIM = WV_DIM

    def build_model(self):
        if self.pretrained_wv == 'gensim-glove-100':
            return gensim_api.load("glove-wiki-gigaword-100")

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        # build batches
        batch_start = idx * self.batch_size
        batch_end = batch_start + self.batch_size
        batch_x = self.x[batch_start:batch_end]
        batch_y = self.y[batch_start:batch_end]

        # process batches
        processed_batch_x = np.array([self.x_process(x) for x in batch_x])
        processed_batch_inputs = [processed_batch_x]  # TODO: ensure you can return list of 1
        processed_batch_y = np.array([self.y_process(y) for y in batch_y])


        if self.use_pos_tags:
            # TODO: concatenate
            raise("Don't use pos_tags ya dummy. Not implemented")
            # pos_v = np.eye(self.POS_DIM)[i]


        elif self.use_pos_embed:
            batch_pos = self.pos_tags[batch_start:batch_end]
            processed_batch_pos = np.array([self.pos_process(pos) for pos in batch_pos])
            processed_batch_inputs = processed_batch_inputs.append(processed_batch_pos)

        if self.use_parse:
            #TODO: concatenate
            raise("Don't use parse ya dummy. Not implemented")

        elif self.use_parse_embed:
            batch_parse = self.parse[batch_start:batch_end]
            processed_batch_parse = np.array([self.parse_process(parse) for parse in batch_parse])
            processed_batch_inputs = processed_batch_inputs.append(processed_batch_parse)

        return processed_batch_inputs, processed_batch_y



    def x_process(self, text):
        wv_tensor = (np.random.random((self.MAX_SENT_LEN, self.WV_DIM)) - 0.5)/5
        oov_count = 0
        for i, w in zip(range(self.MAX_SENT_LEN), text.split()):
            try:
                wv_tensor[i] = self.build_pretrained_wv(w)
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


    def build_pretrained_wv(self, w):
        if self.pretrained_wv == 'gensim-glove-100':
            return self.pretrained_wv[w]


class ExperimentWrapper:
    # TODO: define kwargs
    def __init__(self, model_kwargs, sequence_kwargs, fit_kwargs,
                 model_filepath):
        self.model = ModelFactory().create(**model_kwargs)
        self.sequence_kwargs = sequence_kwargs
        self.fit_kwargs = fit_kwargs
        self.model_filepath = model_filepath


    def train_model(self, train_data_kwargs):
        # train
        training_generator = TextSequence(**train_data_kwargs, **self.sequence_kwargs)
        hist = self.model.fit_generator(training_generator, **self.fit_kwargs)

        # plot
        # TODO: save output somehow?
        history = pd.DataFrame(hist.history)
        plt.figure(figsize=(12, 12));
        plt.plot(history["loss"]);
        plt.plot(history["val_loss"]);
        plt.title("Loss with pretrained word vectors");
        plt.show();

        # save
        # TODO: check if filepath exists
        self.model.save(self.model_filepath)

    def evaluate_model(self, test_data_kwargs):
        eval_generator = TextSequence(**test_data_kwargs, **self.sequence_kwargs)
        # TODO: evaluate
