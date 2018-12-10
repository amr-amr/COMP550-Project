import pandas as pd
from tensorflow import keras
from keras.datasets import imdb
from data_generation.PosAndParseExtractor import PosAndParseExtractor
from time import time


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


extractor = PosAndParseExtractor("en_core_web_md")
for (x, y), partition in zip(load_data(), ('train', 'test')):

    df = pd.DataFrame(columns=['text', 'spacy_pos', 'nltk_pos', 'parse', 'label'])
    df['text'] = x
    df['label'] = y

    print('Starting to parse %s partition' % partition)
    start = time()
    df[['spacy_pos', 'nltk_pos', 'parse']] = df.apply(lambda row: extractor.parse_text(row['text']), axis=1, result_type='expand')
    print(time() - start)

    df.to_pickle('df_%s.pkl' % partition)

pkl_train = pd.read_pickle('df_train.pkl')
pkl_train.info(memory_usage='deep')
pkl_test = pd.read_pickle('df_test.pkl')
pkl_test.info(memory_usage='deep')