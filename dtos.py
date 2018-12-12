"""
Comp 550 - Final Project - Fall 2018
Augmenting Word Embeddings using Additional Linguistic Information
Group 1 - Andrei Mircea (260585208) - Stefan Wapnick (id 260461342)

Github:                 https://github.com/amr-amr/COMP550-Project
Public Data folder:     https://drive.google.com/drive/folders/1Z0YrLC8KX81HgDlpj1OB4bCM6VGoAXmE?usp=sharing

Script Description:

"""
from caching import PosDictionary
import datetime
import time
import copy


class ExperimentData:

    def __init__(self, x, x_pos, x_parse, y, df):
        self.y = y
        self.x = x
        self.x_pos = x_pos
        self.x_parse = x_parse
        self.df = df

    @staticmethod
    def from_df(df, text_col='spacy_text', pos_col='spacy_pos', label_col='label'):
        return ExperimentData(df[text_col], df[pos_col], df['parse'], df[label_col], df)


class ExperimentParameters:

    def __init__(self, batch_size=128, train_wv=False,
                 use_pos=None, use_parse=None, pos_dict_len=None, sent_dim=300, wv_dim=100,
                 pos_dim=None, epochs=20, dropout=0.5, nn_model='lstm'):
        self.batch_size = batch_size
        self.train_wv = train_wv
        self.use_pos = use_pos
        self.use_parse = use_parse
        self.sent_dim = sent_dim
        self.wv_dim = wv_dim
        self.pos_dict_len = PosDictionary.spacy_len if pos_dict_len is None else pos_dict_len
        self.pos_dim = self.pos_dict_len if pos_dim is None else pos_dim
        self.epochs = epochs
        self.dropout = dropout
        self.nn_model = nn_model
        self.timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')

    def __str__(self) -> str:
        return "nn_model=%s batch_size=%s sent_dim=%d train_wv=%s dropout=%.2f use_pos=%s pos_dim=%d use_parse=%s" \
               % (self.nn_model, self.batch_size, self.sent_dim, self.train_wv, self.dropout,
                  self.use_pos, self.pos_dim, self.use_parse)

    def is_baseline(self):
        return self.use_pos is None and self.use_parse is None

    def get_baseline(self):
        return "%s_%d-sentdim_%s-trainwv" \
               % (self.nn_model, self.sent_dim, self.train_wv)

    def get_name(self) -> str:
        return "%s_%d-sentdim_%s-trainwv_%.2f-dropout_%s-%d-pos_%s-parse" \
               % (self.nn_model, self.sent_dim, self.train_wv, self.dropout,
                  self.use_pos, self.pos_dim, self.use_parse)

    def file_name(self) -> str:
        return self.get_name()
