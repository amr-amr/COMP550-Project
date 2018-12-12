import os
import pandas as pd
import spacy
from tensorflow import keras
from data_generation.pos_dicts import PosDictionary
from time import time
from nltk import pos_tag as pos_tagger
imdb = keras.datasets.imdb


class PosAndParseExtractor:
    def __init__(self, spacy_model="en_core_web_lg"):
        self.spacy_pos_dict = PosDictionary.spacy
        self.nltk_pos_dict = PosDictionary.nltk
        self.nlp = spacy.load(spacy_model)
        self.counter = 0

    def parse_text(self, text):
        spacy_pos_tags = []
        parse_tree = []

        if self.counter % 100 == 0:
            print(self.counter)

        doc = self.nlp(text)
        for token in doc:
            # parse pos tags
            pos_tag = token.pos_
            try:
                i = self.spacy_pos_dict[pos_tag]
            except:
                i = 16
            spacy_pos_tags.append(i)

            # dependency parse
            dep_and_head = (token.dep_, token.head.i)
            parse_tree.append(dep_and_head)

        nltk_pos_tags = [self.nltk_pos_dict[pos] for (word, pos) in pos_tagger(text.split())]

        self.counter = self.counter + 1
        return spacy_pos_tags, nltk_pos_tags, parse_tree


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


if __name__ == '__main__':
    extractor = PosAndParseExtractor("en_core_web_md")
    for (x, y), partition in zip(load_data(), ('train', 'test')):
        df = pd.DataFrame(columns=['text', 'spacy_pos', 'nltk_pos', 'parse', 'label'])
        df['text'] = x
        df['label'] = y

        print('Starting to parse %s partition' % partition)
        start = time()
        df[['spacy_pos', 'nltk_pos', 'parse']] = df.apply(lambda row: extractor.parse_text(row['text']), axis=1,
                                                          result_type='expand')
        print(time() - start)

        df.to_pickle('df_%s.pkl' % partition)
