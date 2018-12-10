from data_generation.pos_dicts import PosDictionary
import datetime
import time


class ExperimentData:

    def __init__(self, x, x_pos, x_parse, y):
        self.y = y
        self.x = x
        self.x_pos = x_pos
        self.x_parse = x_parse

    @staticmethod
    def from_df(df, text_col='text', pos_col='nltk_pos', label_col='label'):
        return ExperimentData(df[text_col], df[pos_col], [], df[label_col])


class ExperimentParameters:

    def __init__(self, batch_size=512, wv_type='gensim-glove-100',
                 use_pos=None, use_parse=False, pos_dict_len=None, sent_dim=200, wv_dim=100,
                 pos_dim=None, epochs=20, dropout=0.1):
        self.batch_size = batch_size
        self.wv_type = wv_type
        self.use_pos = use_pos
        self.use_parse = use_parse
        self.sent_dim = sent_dim
        self.wv_dim = wv_dim
        self.pos_dict_len = PosDictionary.spacy_len if pos_dict_len is None else pos_dict_len
        self.pos_dim = self.pos_dict_len if pos_dim is None else pos_dim
        self.epochs = epochs
        self.dropout = dropout
        self.timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')

    def __str__(self) -> str:
        return "batch_size=%s wv_type=%s use_pos=%s use_parse=%s sent_dim=%d wv_dim=%d pos_dim=%d epochs=%d" \
               % (self.batch_size, self.wv_type, self.use_pos, self.use_parse, self.sent_dim,
                  self.wv_dim, self.pos_dim, self.epochs)

    # TODO: extension?
    def file_name(self) -> str:
        return '%s_%s' % (self.__str__().replace(' ', '').lower(), self.timestamp)
