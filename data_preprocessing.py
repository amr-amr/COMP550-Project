import os
from time import time

import pandas as pd
import spacy
from keras.datasets import imdb
from nltk import pos_tag as pos_tagger
from caching import PosDictionary


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


def download_imdb_dataset():
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


def preprocess_imdb_dataset(output_dir):
    extractor = LinguisticDataExtractor("en_core_web_md")
    for (x, y), partition in zip(download_imdb_dataset(), ('train', 'test')):
        df = pd.DataFrame(columns=['text', 'spacy_text', 'spacy_pos', 'nltk_pos', 'parse', 'label'])
        df['text'] = x
        df['label'] = y

        print('Starting to parse %s set' % partition)
        start = time()
        df[['spacy_text', 'spacy_pos', 'nltk_pos', 'parse']] = df.apply(lambda row: extractor.parse_text(row['text']),
                                                                        axis=1,
                                                                        result_type='expand')
        print('Took %d to parse %s set' % (time() - start, partition))
        output_file = os.path.join(output_dir, 'df_%s.pkl' % partition)
        print('Saving %s set to %s' % (partition, output_file))
        df.to_pickle(output_file)


if __name__ == '__main__':
    preprocess_imdb_dataset('./data')
