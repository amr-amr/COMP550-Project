"""
Comp 550 - Final Project - Fall 2018
Augmenting Word Embeddings using Additional Linguistic Information
Group 1 - Andrei Mircea (260585208) - Stefan Wapnick (id 260461342)

Github:                 https://github.com/amr-amr/COMP550-Project
Public Data folder:     https://drive.google.com/drive/folders/1Z0YrLC8KX81HgDlpj1OB4bCM6VGoAXmE?usp=sharing

Script Description:

"""
import os
from time import time

import pandas as pd
import spacy
from keras.datasets import imdb
from nltk import pos_tag as pos_tagger
from caching import PosDictionary
from constants import DATA_DIRECTORY
from caching import WordIndexCache


class LinguisticDataExtractor:
    def __init__(self, spacy_model="en_core_web_lg"):
        self.spacy_pos_dict = PosDictionary.spacy
        self.nltk_pos_dict = PosDictionary.nltk
        self.nlp = spacy.load(spacy_model)
        self.processed_counter = 0

    def parse_text(self, text):
        spacy_pos_tags = []
        parse_tree = []
        spacy_text = []

        self.processed_counter += 1
        if self.processed_counter % 100 == 0:
            print('Processed count = %d' % self.processed_counter)

        doc = self.nlp(text)
        for token in doc:
            spacy_text.append(token.lemma_)
            # parse pos tags
            pos_tag = token.pos_
            try:
                i = self.spacy_pos_dict[pos_tag]
            except:
                print('except')
                i = 16
            spacy_pos_tags.append(i)

            # dependency parse
            dep_and_head = (token.dep_, token.head.i)
            parse_tree.append(dep_and_head)

        nltk_pos_tags = [self.nltk_pos_dict[pos] for (word, pos) in pos_tagger(spacy_text)]
        return spacy_text, spacy_pos_tags, nltk_pos_tags, parse_tree


def load_imdb_dataset():
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


def extract_linguistic_data(force_reload=False):
    test_df_file = os.path.join(DATA_DIRECTORY, 'df_test.pkl')
    train_df_file = os.path.join(DATA_DIRECTORY, 'df_train.pkl')

    if not force_reload and os.path.exists(test_df_file) and os.path.exists(train_df_file):
        print('Data %s, %s already exists' % (test_df_file, train_df_file))
        return pd.read_pickle('df_train.pkl'), pd.read_pickle('df_test.pkl')

    extractor = LinguisticDataExtractor('en_core_web_md')
    df_results = []
    for (x, y), partition in zip(load_imdb_dataset(), ('train', 'test')):
        df = pd.DataFrame(columns=['text', 'spacy_text', 'spacy_pos', 'nltk_pos', 'parse', 'label'])
        df['text'] = x
        df['label'] = y

        print('Starting to parse %s set' % partition)
        start = time()
        df[['spacy_text', 'spacy_pos', 'nltk_pos', 'parse']] = df.apply(lambda row: extractor.parse_text(row['text']),
                                                                        axis=1,
                                                                        result_type='expand')
        print('Took %d to parse %s set' % (time() - start, partition))
        output_file = os.path.join(DATA_DIRECTORY, 'df_%s.pkl' % partition)
        print('Saving %s set to %s' % (partition, output_file))
        df.to_pickle(output_file)
        df_results.append(df)

    return df_results[0], df_results[1]


def preprocess_dataset():
    df_train, df_test = extract_linguistic_data()

    if WordIndexCache.get_word_index() is not None:
        return df_train, df_test

    print('Assigning indices to all words...')
    all_text = list(df_train['spacy_text']) + list(df_test['spacy_text'])
    WordIndexCache.initialize(all_text)
    return df_train, df_test


if __name__ == '__main__':
    preprocess_dataset()
