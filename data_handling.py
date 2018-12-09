import os
import pandas as pd
import spacy
from tensorflow import keras

imdb = keras.datasets.imdb

class SpacyPosAndParse:
    def __init__(self, spacy_model="en_core_web_lg"):
        self.pos_dict = self.build_pos_dict()
        self.nlp = spacy.load(spacy_model)

    def build_pos_dict(self):
        pos_dict = {'ADJ': 0,
                    'ADP': 1,
                    'ADV': 2,
                    'AUX': 3,
                    'CONJ': 4,
                    'CCONJ': 4,
                    'DET': 5,
                    'INTJ': 6,
                    'NOUN': 7,
                    'NUM': 8,
                    'PART': 9,
                    'PRON': 10,
                    'PROPN': 11,
                    'PUNCT': 12,
                    'SCONJ': 13,
                    'SYM': 14,
                    'VERB': 15,
                    'X': 16}
        return pos_dict

    def parse_text(self, text):
        pos_tags = []
        parse_tree = []

        doc = self.nlp(text)
        for token in doc:
            # parse pos tags
            pos_tag = token.pos_
            try:
                i = self.pos_dict[pos_tag]
            except:
                i = 16
            pos_tags.append(i)

            # dependency parse
            dep_and_head = (token.dep_, token.head.i)
            parse_tree.append(dep_and_head)

        return pos_tags, parse_tree


def pre_compute_data(train_x, train_y, test_x, test_y):
    # create dataframes
    df_train = pd.DataFrame(columns=['text', 'label', 'pos', 'parse'])
    df_test = pd.DataFrame(columns=['text', 'label', 'pos', 'parse'])

    df_train['text'] = train_x
    df_train['label'] = train_y
    df_test['text'] = test_x
    df_test['label'] = test_y

    # compute pos tags and dependency parse
    sda = SpacyPosAndParse(spacy_model="en_core_web_sm")
    df_train[['pos', 'parse']] = df_train.apply(lambda row: sda.parse_text(row['text']), axis=1, result_type='expand')
    df_test[['pos', 'parse']] = df_test.apply(lambda row: sda.parse_text(row['text']), axis=1, result_type='expand')

    # save dataframes to pickle
    df_train_path = os.path.join('/', "df_imdb_train.pkl")
    df_test_path = os.path.join('/', "df_imdb_test.pkl")
    df_train.to_pickle(df_train_path)
    df_test.to_pickle(df_test_path)


def load_data():
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data()

    # convert from integers to text
    word_index = imdb.get_word_index()
    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2  # unknown
    word_index["<UNUSED>"] = 3

    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    def decode_review(text):
        return ' '.join([reverse_word_index.get(i, '?') for i in text])

    train_x = [decode_review(x) for x in train_data]
    test_x = [decode_review(x) for x in test_data]
    return (train_x, train_labels), (test_x, test_labels)
