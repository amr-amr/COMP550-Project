from data_generation.pos_dicts import PosDictionary


class ExperimentData:

    def __init__(self, x, x_pos, x_parse, y):
        self.y = y
        self.x = x
        self.x_pos = x_pos
        self.x_parse = x_parse


class ExperimentParameters:

    def __init__(self, batch_size=512, wv_type='gensim-glove-100',
                 use_pos=False, use_parse=False, pos_dict=None, sent_dim=200, wv_dim=100,
                 pos_dim=None, epochs=20):
        self.batch_size = batch_size
        self.wv_type = wv_type
        self.use_pos = use_pos
        self.use_parse = use_parse
        self.sent_dim = sent_dim
        self.wv_dim = wv_dim
        self.pos_dict_len = PosDictionary.spacy_len if pos_dict is None else pos_dict
        self.pos_dim = self.pos_dict_len if pos_dim is None else pos_dim
        self.epochs = epochs

    def __str__(self) -> str:
        return "batch_size=%s, wv_type=%s, use_pos=%s, use_parse=%s, sent_dim=%d, wv_dim=%d, pos_dim=%d, epochs=%d" \
               % (self.batch_size, self.wv_type, self.use_pos, self.use_parse, self.sent_dim,
                  self.wv_dim, self.pos_dim, self.epochs)
